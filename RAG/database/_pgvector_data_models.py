from typing import Optional, Tuple, List, Any

import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, relationship
import uuid

from database.pgvector import Base


class CollectionStore(Base):
    """Collection store."""

    __tablename__ = "langchain_pg_collection"

    name = sqlalchemy.Column(sqlalchemy.String)
    # cmetadata = sqlalchemy.Column(JSON)
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_name = sqlalchemy.Column(sqlalchemy.String)
    job = sqlalchemy.Column(sqlalchemy.String)

    documents = relationship(
        "DocumentStore",
        back_populates="collection",
        passive_deletes=True,
    )

    @classmethod
    def get_by_name(cls, session: Session, name: str) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.name == name).first()  # type: ignore

    @classmethod
    def get_or_create(
        cls,
        session: Session,
        name: str,
        role_name: str = '',
        job: str = ''
        # cmetadata: Optional[dict] = None,
    ) -> Tuple["CollectionStore", bool]:
        """
        Get or create a collection.
        Returns [Collection, bool] where the bool is True if the collection was created.
        """
        created = False
        collection = cls.get_by_name(session, name)
        if collection:
            return collection, created

        collection = cls(name=name, role_name=role_name, job=job)
        session.add(collection)
        session.commit()
        created = True
        return collection, created

class DocumentStore(Base):
    """Document store."""

    __tablename__ = "langchain_pg_document"

    collection_id = sqlalchemy.Column(
        UUID(as_uuid=True),
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.uuid",
            ondelete="CASCADE",
        ),
    )
    collection = relationship("CollectionStore", back_populates="documents")
    embeddings = relationship("EmbeddingStore", back_populates="document", passive_deletes=True)

    document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    uid = sqlalchemy.Column(sqlalchemy.String, primary_key=True)
    title = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    category = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    url = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    publish_time = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)


    @classmethod
    def get_by_uid(cls, session: Session, uid: str) -> Optional["DocumentStore"]:
        return session.query(cls).filter(cls.uid == uid).first()
    
    @classmethod
    def search_documents(cls, session: Session, collection_id, title: str, category: str) -> List["DocumentStore"]:
        if collection_id is None:
            return []
        query = session.query(cls).filter(cls.collection_id == collection_id)
        if title is not None:
            query = query.filter(cls.title.like(f'%{title}%'))
        if category is not None:
            query = query.filter(cls.category == category)
        return query.limit(100).all()


class EmbeddingStore(Base):
    """Embedding store."""

    __tablename__ = "langchain_pg_embedding"

    document_id = sqlalchemy.Column(
        sqlalchemy.String,
        sqlalchemy.ForeignKey(
            f"{DocumentStore.__tablename__}.uid",
            ondelete="CASCADE",
        ),
    )
    document = relationship("DocumentStore", back_populates="embeddings")

    embedding: Vector = sqlalchemy.Column(Vector(None))
    paragraph = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
