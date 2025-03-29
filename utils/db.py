import torch
from PIL import Image
from qdrant_client import models, QdrantClient

from configs import EMBEDDING_DIMS, EMBEDDING_MODEL_NAME
from utils.llm import summary
from utils.emb import (
    get_image_embedding,
    get_text_embedding,
    fetch_text_embeddings,
    fetch_image_embeddings,
    load_text_split,
)


def get_collections(qd_client: QdrantClient):

    collections = qd_client.get_collections()
    collection_names = [i.name for i in collections.collections]

    return collection_names


def recreate_collection(qd_client: QdrantClient, collection_name: str):

    if qd_client.collection_exists(collection_name):
        qd_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMS,
                distance=models.Distance.COSINE,
            ),
        )


"""
在Qdrant向量数据库中创建集合，并将文本或图像数据的嵌入向量存储到集合中。
"""
def create_collection(
    model,
    preprocess,
    qd_client: QdrantClient,
    text_files: list[str] = None,
    image_files: list[str] = None,
    batch_size: int = 24,
    chunk_size: int = 128,
    recreate_collection: bool = False,
):
    assert not (text_files and image_files) or not (text_files or image_files)
    #使用 assert 语句确保要么只提供文本文件，要么只提供图像文件，不能同时提供两者。
    
    if text_files is not None:
        text_chunks = []
        for file in text_files:
            chunks = load_text_split(file, chunk_size=chunk_size)
            text_chunks.extend(chunks)
        data = fetch_text_embeddings(
            model, text_chunks, bs=batch_size, emb_model=EMBEDDING_MODEL_NAME
        )
    elif image_files := image_files:
        data = fetch_image_embeddings(
            model,
            preprocess,
            image_files,
            batch_size=batch_size,
            emb_model=EMBEDDING_MODEL_NAME,
        )

    if not data:
        return

    collection_name = data[0]["metadata"]["type"]
    print(f"》》》正在新建的collection：{collection_name}")
    if not qd_client.collection_exists(collection_name):
        qd_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMS,
                distance=models.Distance.COSINE,
            ),
        )
    elif recreate_collection:
        qd_client.create_collection_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMS,
                distance=models.Distance.COSINE,
            ),
        )

    records = [
        models.Record(id=idx, vector=d["vec"], payload=d["metadata"])
        for idx, d in enumerate(data)
    ]

    qd_client.upload_points(
        collection_name=collection_name, points=records, batch_size=128, parallel=12
    )
    print("====finish====")


def get_collection_points(qd_client: QdrantClient, collection_name: str):
    if not qd_client.collection_exists(collection_name):
        return []
    points = qd_client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="embedding_model_name",
                    match=models.MatchValue(value=EMBEDDING_MODEL_NAME),
                ),
            ]
        ),
        limit=1000000,
        with_payload=True,
        with_vectors=False,
    )
    item = "text" if collection_name == "text" else "image_path"
    id_texts = [[i.id, i.payload[item]] for i in points[0]]

    return id_texts


def del_collection_points(
    qd_client: QdrantClient, collection_name: str, point_ids: list[int]
):
    qd_client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(
            points=point_ids,
        ),
    )


def retrieve_image_text(
    model,
    preprocess,
    qd_client: QdrantClient,
    query: str | Image.Image,
    query_type="text",
    retrieve_type="full",
    top_k=12,
    text_thres=None,
    image_thres=None,
    use_llm_summary=False,
) -> tuple[list]:
    # 根据查询类型生成查询向量
    if query_type == "text":
        # 如果使用大语言模型进行文本总结，则对查询文本进行总结
        if use_llm_summary:
            query = summary(query)[1]
        # 调用 get_text_embedding 函数生成文本查询向量
        query_vec = get_text_embedding(model, query)
    else:
        # 调用 get_image_embedding 函数生成图像查询向量
        query_vec = get_image_embedding(model, preprocess, query)

    # 初始化检索到的文本列表
    retrieve_texts = []
    # 如果检索类型包含文本或全部
    if retrieve_type in ["text", "full"]:
        # 设置文本集合名称
        collection_name = "text"
        print(f">>> 正在检索 {collection_name} collection")
        # 检查文本集合是否存在
        if qd_client.collection_exists(collection_name):
            # 从文本集合中查询相关点
            retrieve_texts = qd_client.query_points(
                collection_name=collection_name,
                query=query_vec,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="embedding_model_name",
                            match=models.MatchValue(value=EMBEDDING_MODEL_NAME),
                        ),
                    ]
                ),
                limit=top_k,
                score_threshold=text_thres,
            )
            # 提取查询结果中的分数和文本内容
            retrieve_texts = [
                [i.score, i.payload["text"]] for i in retrieve_texts.points
            ]
        else:
            # 如果文本集合不存在，返回空列表
            retrieve_texts = []
        print(f"    数量：{len(retrieve_texts)}")

    # 初始化检索到的图像列表
    retrieve_images = []
    # 如果检索类型包含图像或全部
    if retrieve_type in ["image", "full"]:
        # 设置图像集合名称
        collection_name = "image"
        print(f">>> 正在检索图片 {collection_name} collection")
        # 检查图像集合是否存在
        if qd_client.collection_exists(collection_name):
            # 从图像集合中查询相关点
            retrieve_images = qd_client.query_points(
                collection_name=collection_name,
                query=query_vec,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="embedding_model_name",
                            match=models.MatchValue(value=EMBEDDING_MODEL_NAME),
                        ),
                    ]
                ),
                limit=top_k,
                score_threshold=image_thres,
            )
            # 提取查询结果中的分数和图像路径
            retrieve_images = [
                [i.score, i.payload["image_path"]] for i in retrieve_images.points
            ]
        else:
            # 如果图像集合不存在，返回空列表
            retrieve_images = []
        print(f"    数量：{len(retrieve_images)}")

    # 清空 PyTorch CUDA 缓存
    torch.cuda.empty_cache()

    # 返回检索到的文本和图像列表
    return retrieve_texts, retrieve_images
