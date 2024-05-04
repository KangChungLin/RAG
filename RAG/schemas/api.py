from pydantic import BaseModel
from typing import Optional, List
from schemas.document import Document

class ReqBase(BaseModel):
    collection_name: str

class searchReq(ReqBase):
    title: Optional[str] = None
    category: Optional[str] = None

class searchResp(BaseModel):
    docs: List[Document] = []

class searchReq(ReqBase):
    title: Optional[str] = None
    category: Optional[str] = None

class collectionReq(ReqBase):
    role_name: str = ''
    job: str = ''

class commonResp(BaseModel):
    rcode: str = '0000'

class addReq(ReqBase):
    title: str
    content: str
    category: Optional[str] = None
    url: Optional[str] = None
    publish_time: Optional[str] = None

class docReq(ReqBase):
    uid: str

class editReq(ReqBase):
    uid: str
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    url: Optional[str] = None
    publish_time: Optional[str] = None

class askReq(ReqBase):
    question: str
    k: int = 3
    publish_time: Optional[str] = None

class askResp(BaseModel):
    result: str

class similarity(BaseModel):
    doc: Document
    score: float

class similarityResp(BaseModel):
    result: List[similarity]

