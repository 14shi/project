import torch
import base64, io
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from configs import EMBEDDING_MODEL_NAME, DB_DIR, EMBEDDING_MODEL_INFOS
from utils.emb import load_from_name
from utils.db import (
    get_collections,
    create_collection,
    recreate_collection,
    get_collection_points,
    del_collection_points,
    retrieve_image_text,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name(
    EMBEDDING_MODEL_NAME,
    device,
    download_root="./ckpts",
    **EMBEDDING_MODEL_INFOS.get(EMBEDDING_MODEL_NAME, {}),
)
qd_client = QdrantClient(path=DB_DIR)


app = FastAPI(
    title="CLip server",
    description="",
    version="0.1.0",
)

#服务器要被不同域名的前端页面访问，就需要使用CORS中间件来允许跨域请求。
#它会在请求到达应用的路由处理函数之前和响应返回给客户端之前进行一些处理。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#collection_name为查询参数 
#将客户端发送的HTTP请求中的查询字符串参数映射到路由函数的参数上。
@app.get("/recreate_collection")
def recreate_collection_server(collection_name: str):
    recreate_collection(qd_client, collection_name)


@app.get("/get_collections")
def get_collections_server():
    collection_names = get_collections(qd_client)
    return collection_names


class Data(BaseModel):
    text_files: list[str] = None
    image_files: list[str] = None
    recreate_collection: bool = False
    batch_size: int = 24
    chunk_size: int = 128

#request.model_dump() 方法会把 request 对象转换为一个字典，
#  ** 操作符会将这个字典解包，
# 将字典中的键值对作为关键字参数传递给 create_collection 函数
@app.post("/create_collection")
def create_collection_server(request: Data):
    create_collection(model, preprocess, qd_client, **request.model_dump())


class QueryData(BaseModel):
    query: str
    query_type: str = "text"
    retrieve_type: str = "full"
    top_k: int = 12
    text_thres: float = None
    image_thres: float = None
    use_llm_summary: bool = False


@app.post("/retrieve_image_text")
def retrieve_image_text_server(request: QueryData):
    params = request.model_dump()
    if params["query_type"] == "image":
        query = Image.open(io.BytesIO(base64.b64decode(params["query"])))
        # 1. `params["query"]` 取出客户端传递的图像数据，这个数据是经过 Base64 编码的字符串。
        # 2. `base64.b64decode(params["query"])` 对 Base64 编码的字符串进行解码，得到二进制图像数据。
        # 3. `io.BytesIO(...)` 将二进制图像数据包装成一个类文件对象，以便后续可以像操作文件一样操作它。
        # 4. `Image.open(...)` 使用 `PIL` 库的 `Image` 类打开这个类文件对象，得到一个 `PIL.Image` 对象。
        # 最终，`query` 变量存储的是一个 `PIL.Image` 对象，表示解码后的图像。
        params["query"] = query
    retrieve_texts, retrieve_images = retrieve_image_text(
        model, preprocess, qd_client, **params
    )

    return (retrieve_texts, retrieve_images)


@app.get("/get_collection_points")
def get_collection_points_server(collection_name: str) -> list[list]:

    return get_collection_points(qd_client, collection_name)


class DelDate(BaseModel):
    collection_name: str
    point_ids: list[int]


@app.post("/del_collection_points")
def del_collection_points_server(request: DelDate):

    del_collection_points(qd_client, **request.model_dump())


## uvicorn server:app --host 0.0.0.0 --port 8000 --reload
