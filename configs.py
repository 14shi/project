model_dims = {
    "RN50": 1024,
    "ViT-B-16": 512,
    "ViT-L-14": 512,
    "ViT-L-14-336": 512,
    "ViT-H-14": 1024,
    "clip_cn_vit-b-16-panda-sft.pt": 512,
}

EMBEDDING_MODEL_INFOS = {
    "clip_cn_vit-b-16-panda-sft.pt": {
        "vision_model_name": "ViT-B-16",
        "text_model_name": "RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224,
    }
}


# db
DB_DIR = "database/db/qdrant"


# embedding setting
# EMBEDDING_MODEL_NAME = "ViT-L-14"   ## 改了模型的话，得重新上传数据
EMBEDDING_MODEL_NAME = "clip_cn_vit-b-16-panda-sft.pt"
EMBEDDING_DIMS = model_dims[EMBEDDING_MODEL_NAME]
BATCH_SIZE = 8
CHUNK_SIZE = 128

# text summary setting
from dotenv import load_dotenv
import os

# 加载.env文件中的环境变量
load_dotenv()

# 使用环境变量
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
SILICONFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"
