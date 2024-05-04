from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv
from database.pgvector import PGVector
from database.connect import get_db
from sqlalchemy.orm import Session
from datetime import datetime
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, Depends, File, UploadFile, Form
from loguru import logger
from io import BytesIO
from prompt import answer_query_with_context
from log_setting import configure_logger
from schemas.api import (
    searchReq, 
    searchResp, 
    collectionReq, 
    commonResp,
    addReq,
    docReq,
    editReq,
    askReq,
    askResp,
    similarityResp)

load_dotenv()

tags_metadata = [
    {
        "name": "collection",
        "description": "Manage collections",
    },
    {
        "name": "document",
        "description": "Manage documents",
    },
    {
        "name": "RAG",
        "description": "Retrieval and generation",
    }
]

app = FastAPI(openapi_tags=tags_metadata)

configure_logger()

use_local_model = os.getenv('USE_LOCAL_MODEL') == "True"

# 看是否要用OpenAI API
if use_local_model:
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
else:
    embeddings = OpenAIEmbeddings()

@app.post("/api/collection/create", response_model=commonResp, 
          tags=["collection"], description='新增collection')
def create_collection(req: collectionReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    created = pgvector.create_collection(db, role_name=req.role_name, job=req.job)
    if created:
        return {"rcode":"0000"}
    else: # 已存在該collection_name
        return {"rcode":"0001"}
    
@app.post("/api/collection/delete", response_model=commonResp, 
          tags=["collection"], description='刪除collection')
def delete_collection(req: collectionReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    deleted = pgvector.delete_collection(db)
    if deleted:
        return {"rcode":"0000"}
    else: # 無該collection_name
        return {"rcode":"0001"}


@app.post("/api/document/search", response_model=searchResp, 
          tags=["document"], description='文件標題搜尋')
def search(req: searchReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    results = pgvector.search_documents(db, title=req.title, category=req.category)
    return {"docs":results}

@app.post("/api/document/add", response_model=commonResp, 
          tags=["document"], description='新增文件')
def add(req: addReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    metadata = {'title': req.title}
    if req.category is not None:
        metadata['category'] = req.category
    if req.url is not None:
        metadata['url'] = req.url
    if req.publish_time is not None:
        date_format = '%Y-%m-%d'
        try:
            publish_time = datetime.strptime(req.publish_time, date_format)
            metadata['publish_time'] = publish_time
        except:
            pass
    pgvector.add_texts(db, texts=[req.content], metadatas=[metadata])
    return {"rcode":"0000"}

@app.post("/api/document/get", response_model=searchResp, 
          tags=["document"], description='文件id搜尋')
def get_doc(req: docReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    doc = pgvector.get_document(db, req.uid)
    if doc is None:
        return {"docs":[]}
    return {"docs":[doc]}

@app.post("/api/document/delete", response_model=commonResp, 
          tags=["document"], description='刪除文件')
def delete(req: docReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    pgvector.delete(db, [req.uid])
    return {"rcode":"0000"}

@app.post("/api/document/edit", response_model=commonResp, 
          tags=["document"], description='編輯文件的內容或其他資訊')
def edit(req: editReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    metadata = {}
    if req.title is not None:
        metadata['title'] = req.title
    if req.category is not None:
        metadata['category'] = req.category
    if req.url is not None:
        metadata['url'] = req.url
    if req.publish_time is not None:
        date_format = '%Y-%m-%d'
        try:
            publish_time = datetime.strptime(req.publish_time, date_format)
            metadata['publish_time'] = publish_time
        except:
            pass
    pgvector.update_documentsStore(db, uid=req.uid, content=req.content, metadata=metadata)
    return {"rcode":"0000"}

@app.post("/api/document/ask", response_model=askResp, 
          tags=["RAG"], description='LLM根據搜尋到的相似資料來回答')
def ask(req: askReq, db: Session = Depends(get_db)):
    if use_local_model:
        return {"result": "此功能無法使用"}
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    docs_with_score = pgvector.similarity_search_with_score(db, req.question, k=req.k)
    docs_content = [doc[0].page_content for doc in docs_with_score]
    result = answer_query_with_context(req.question, docs_content)
    return {"result": result}

@app.post("/api/document/similarity_search", response_model=similarityResp, 
          tags=["RAG"], description='語意相似度搜尋')
def similarity_search(req: askReq, db: Session = Depends(get_db)):
    pgvector = PGVector(embedding_function=embeddings, collection_name=req.collection_name)
    filter_by = {}
    if req.publish_time is not None:
        try:
            publish_time = datetime.strptime(req.publish_time, "%Y-%m-%d")
            filter_by['publish_time'] = publish_time
        except:
            logger.info(f'can not parse "{req.publish_time}"')
    docs_with_score = pgvector.similarity_search_with_score(db, req.question, k=req.k, filter=filter_by)
    result = [{"doc": doc[0], "score": doc[1]} for doc in docs_with_score]
    return {"result": result}


@app.post("/api/document/batch_upload", response_model=commonResp, 
          tags=["document"], description='匯入資料 (限excel或csv)')
async def batch_upload(collection_name: str = Form(...), file: UploadFile = File(...), 
                       db: Session = Depends(get_db)):
    try:
        _, file_extension = os.path.splitext(file.filename)
        content = await file.read()
        data = BytesIO(content)
        if file_extension == '.csv':
            file_df = pd.read_csv(data)
        elif file_extension == '.xlsx' or file_extension == '.xls':
            file_df = pd.read_excel(data)
        else:
            logger.error('error file extension')
            return {"rcode":"1002"} #副檔名錯誤
        file_df = file_df.replace({np.nan:None})
    except Exception as e:
        logger.error(f'batch_upload error: {e}')
        # 檔案格式錯誤
        return {"rcode":"1001"}
    finally:
        await file.close()
    
    if file_df.shape[0]==0:
        # 無資料
        return {"rcode":"1002"}
    
    if 'title' not in file_df.columns or 'content' not in file_df.columns:
        # 缺少重要欄位
        return {"rcode":"1003"}
    
    
    texts = file_df['content'].values.tolist()

    if 'publish_time' in file_df.columns:
        file_df['publish_time'] = file_df['publish_time'].astype(str)
        file_df['publish_time'] = pd.to_datetime(file_df['publish_time'], errors='coerce')
        file_df = file_df.replace({pd.NaT: None})
    metadatas = file_df.drop('content', axis=1).to_dict(orient='records')

    pgvector = PGVector(embedding_function=embeddings, collection_name=collection_name)
    pgvector.add_texts(db, texts=texts, metadatas=metadatas)
    return {"rcode":"0000"}