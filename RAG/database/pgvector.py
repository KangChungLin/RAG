from __future__ import annotations

import asyncio
import enum
import logging
import uuid
from datetime import datetime
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np
import sqlalchemy
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session, declarative_base

from schemas.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain.text_splitter import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from database._pgvector_data_models import CollectionStore


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
    is_separator_regex = False,
)


# class BaseModel(Base):
#     """Base model for the SQL stores."""

#     __abstract__ = True
#     uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]

def _documentStore_to_document(objs: List[Any]) -> List[Document]:
    if objs is None:
        return []
    results = []
    for obj in objs:
        if obj.publish_time is not None:
            publish_time = obj.publish_time.strftime("%Y-%m-%d")
        else:
            publish_time = None
        metadata = {"document_id": obj.uid,
                    "title": obj.title,
                    "category": obj.category,
                    "url": obj.url,
                    "publish_time":publish_time}
        results.append(Document(page_content = obj.document, metadata=metadata))
    return results


class PGVector(VectorStore):
    """`Postgres`/`PGVector` vector store.

    To use, you should have the ``pgvector`` python package installed.

    Args:
        connection_string: Postgres connection string.
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        collection_name: The name of the collection to use. (default: langchain)
            NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
        distance_strategy: The distance strategy to use. (default: COSINE)
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.

    Example:
        .. code-block:: python

            from langchain.vectorstores import PGVector
            from langchain.embeddings.openai import OpenAIEmbeddings

            CONNECTION_STRING = "postgresql+psycopg2://hwc@localhost:5432/test3"
            COLLECTION_NAME = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorestore = PGVector.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )


    """

    def __init__(
        self,
        # connection_string: str,
        embedding_function: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        # collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        # self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        # self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """
        # self._conn = self.connect()
        # self.create_vector_extension()
        from database._pgvector_data_models import (
            CollectionStore,
            DocumentStore,
            EmbeddingStore,
        )

        self.CollectionStore = CollectionStore
        self.DocumentStore = DocumentStore
        self.EmbeddingStore = EmbeddingStore
        # self.create_tables_if_not_exists()
        # self.create_collection()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def create_collection(self, session: Session, role_name: str = '', job: str = '') -> bool:
        if self.pre_delete_collection:
            self.delete_collection()
        # with Session(self._conn) as session:
        _, created =  self.CollectionStore.get_or_create(
            session, self.collection_name, role_name=role_name, job=job
        )
        return created

    def delete_collection(self, session: Session) -> bool:
        self.logger.debug("Trying to delete collection")
        # with Session(self._conn) as session:
        collection = self.get_collection(session)
        if not collection:
            self.logger.warning("Collection not found")
            return False
        session.delete(collection)
        session.commit()
        return True

    def delete(
        self,
        session: Session,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
        """
        # with Session(self._conn) as session:
        if ids is not None:
            self.logger.debug(
                "Trying to delete vectors by ids (represented by the model "
                "using the custom ids field)"
            )
            stmt = delete(self.DocumentStore).where(
                self.DocumentStore.uid.in_(ids)
            )
            session.execute(stmt)
        session.commit()

    def get_collection(self, session: Session) -> Optional["CollectionStore"]:
        return self.CollectionStore.get_by_name(session, self.collection_name)

    @classmethod
    def __from(
        cls,
        session: Session,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        # connection_string: Optional[str] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        store.add_embeddings(
            session=session,
            texts=texts, 
            embeddings=embeddings, 
            metadatas=metadatas, 
            ids=ids, 
            **kwargs
        )

        return store

    def add_embeddings(
        self,
        session: Session,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        
        for text, embedding, id in zip(texts, embeddings, ids):
            embedding_store = self.EmbeddingStore(
                embedding=embedding,
                paragraph=text,
                document_id=id,
            )
            session.add(embedding_store)
        session.commit()

        return ids

    def add_texts(
        self,
        session: Session,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        collection = self.get_collection(session)
        if not collection:
            raise ValueError("Collection not found")
        # insert to DocumentStore
        for text, metadata, id in zip(texts, metadatas, ids):
            documentStore = self.DocumentStore(
                collection_id=collection.uuid,
                document = text,
                uid = id,
                title=metadata.get('title',''),
                category=metadata.get('category',''),
                url=metadata.get('url',''),
                publish_time=metadata.get('publish_time', None),
            )
            session.add(documentStore)
        session.commit()

        # split too long texts
        id_dicts = [{"id": id} for id in ids]
        splits = text_splitter.create_documents(texts=texts, metadatas=id_dicts)
        paragraphs = [s.page_content for s in splits]
        document_ids = [s.metadata['id'] for s in splits]
        embeddings = self.embedding_function.embed_documents(paragraphs)

        return self.add_embeddings(session, texts=paragraphs, embeddings=embeddings, 
                                   ids=document_ids, **kwargs)
        
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        session: Session,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            session, embedding=embedding, k=k, filter=filter
        )
        return docs

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self.EmbeddingStore.embedding.cosine_distance
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self.EmbeddingStore.embedding.max_inner_product
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def similarity_search_with_score_by_vector(
        self,
        session: Session,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        results = self.__query_collection(session, embedding=embedding, k=k, filter=filter)

        return self._results_to_docs_and_scores(results)

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        docs = [
            (
                # _embeddingStore_to_document([result.EmbeddingStore])[0],
                Document(page_content=result.EmbeddingStore.paragraph, 
                         metadata={"document_id":result.EmbeddingStore.document_id}),
                result.distance if self.embedding_function is not None else None,
            )
            for result in results
        ]
        return docs

    def __query_collection(
        self,
        session: Session,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        # with Session(self._conn) as session:
        collection = self.get_collection(session)
        if not collection:
            raise ValueError("Collection not found")

        # filter_by = self.EmbeddingStore.collection_id == collection.uuid
        filter_by = self.DocumentStore.collection_id == collection.uuid

        if filter is not None:
            publish_time = filter.get('publish_time', None)
            if publish_time is not None and isinstance(publish_time, datetime):
                 filter_by = sqlalchemy.and_(filter_by, self.DocumentStore.publish_time>=publish_time)
        #     filter_clauses = []
        #     for key, value in filter.items():
        #         IN = "in"
        #         if isinstance(value, dict) and IN in map(str.lower, value):
        #             value_case_insensitive = {
        #                 k.lower(): v for k, v in value.items()
        #             }
        #             filter_by_metadata = self.EmbeddingStore.cmetadata[
        #                 key
        #             ].astext.in_(value_case_insensitive[IN])
        #             filter_clauses.append(filter_by_metadata)
        #         else:
        #             filter_by_metadata = self.EmbeddingStore.cmetadata[
        #                 key
        #             ].astext == str(value)
        #             filter_clauses.append(filter_by_metadata)

        #     filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

        # _type = self.EmbeddingStore

        results: List[Any] = (
            session.query(
                self.EmbeddingStore,
                self.distance_strategy(embedding).label("distance"),  # type: ignore
            )
            .filter(filter_by)
            .order_by(sqlalchemy.asc("distance"))
            .join(
                self.DocumentStore,
                self.EmbeddingStore.document_id == self.DocumentStore.uid,
            )
            .limit(k)
            .all()
        )
        return results

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return _results_to_docs(docs_and_scores)

    @classmethod
    def from_texts(
        cls: Type[PGVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        """Construct PGVector wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.

        Example:
            .. code-block:: python

                from langchain.vectorstores import PGVector
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = PGVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        results = self.__query_collection(embedding=embedding, k=fetch_k, filter=filter)

        embedding_list = [result.EmbeddingStore.embedding for result in results]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

        return _results_to_docs(docs_and_scores)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def search_documents(self, session: Session, title: str = None, category: str = None) -> List[Any]:
        collection = self.get_collection(session)
        if not collection:
            raise ValueError("Collection not found")
        result = self.DocumentStore.search_documents(session, collection.uuid, title, category)
        return _documentStore_to_document(result)
        
    def update_documentsStore(self, session: Session, uid: str, content: str=None, metadata: dict={}):
        documentsStore = session.query(self.DocumentStore).filter(self.DocumentStore.uid == uid)
        if content is None:
            if len(metadata)>0:
                documentsStore.update(metadata)
                session.commit()
        else:
            update_data = metadata
            update_data['document'] = content
            documentsStore.update(update_data)
            session.commit()

            splits = text_splitter.create_documents(texts=[content])
            paragraphs = [s.page_content for s in splits]
            embeddings = self.embedding_function.embed_documents(paragraphs)

            session.query(self.EmbeddingStore).filter(self.EmbeddingStore.document_id == uid).delete()
            session.commit()
            self.add_embeddings(session, texts=paragraphs, embeddings=embeddings, 
                                ids=[uid for _ in range(len(paragraphs))])
            
    def get_document(self, session: Session, document_id: str) -> Optional[Document]:
        doc = self.DocumentStore.get_by_uid(session, document_id)
        if doc is None:
            return None
        
        collection = self.get_collection(session)
        if collection is None:
            return None
        if doc.collection_id != collection.uuid:
            return None
        
        return _documentStore_to_document([doc])[0]