import os
import torch
import cn_clip.clip as clip
from PIL import Image
from tqdm import tqdm
#tqdm 是一个用于在Python中创建进度条的库，它可以让你直观地看到循环或迭代过程的进度
from utils.llm import summary
from cn_clip.clip.utils import (
    create_model,
    image_transform,
    _MODELS,
    _download,
    _MODEL_INFO,
)

from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.chinese_splitter import ChineseRecursiveTextSplitter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#根据模型名称或文件路径加载模型和对应的图像预处理函数。
#它支持预定义模型的下载和本地模型文件的加载
def load_from_name(name: str, device=DEVICE, download_root: str = "ckpts", **kwargs):

    print("正在加载模型中....")
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root or os.path.expanduser("~/.cache/clip")
        )
        model_struct = _MODEL_INFO[name]["struct"]
        input_resolution = _MODEL_INFO[name]["input_resolution"]
    elif os.path.isfile(name):
        model_path = name
        model_struct = f'{kwargs["vision_model_name"]}@{kwargs["text_model_name"]}'
        input_resolution = kwargs["input_resolution"]
    elif os.path.isfile(download_root + "/" + name):
        model_path = download_root + "/" + name
        model_struct = f'{kwargs["vision_model_name"]}@{kwargs["text_model_name"]}'
        input_resolution = kwargs["input_resolution"]
    else:
        raise RuntimeError(f"Model {name} not found; available models = {_MODELS}")

    with open(model_path, "rb") as opened_file:
        # loading saved checkpoint
        checkpoint = torch.load(opened_file, map_location="cpu")

    model = create_model(model_struct, checkpoint)
    if str(device) == "cpu":
        model.float()
    else:
        model.to(device)
    model.eval()
    preprocess = image_transform(input_resolution)

    return model, preprocess


class ImageDataset(Dataset):
    """Images dataset"""

    def __init__(self, image_list, transform):
        """
        Args:
            image_list: List of image paths.
            transform : Transform to be applied on a sample.
        """
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        data = {"image": image, "image_path": image_path}
        return data


class TextDataset(Dataset):
    def __init__(self, texts: str | list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        keywords = summary(text)
        return {"texts": text, "keywords": keywords}


def load_text_split(file_path: str, chunk_size=128, chunk_overlap=0) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_image_embedding(model, preprocess, image: str | Image.Image):

    if isinstance(image, str):
        image = Image.open(image)
   # 检查 image 是否为字符串类型，如果是，则认为它是图像文件路径

    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_embedding = model.encode_image(image.to(DEVICE))
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
       # 对图像嵌入向量进行归一化处理，使其模长为 1。

    return image_embedding.squeeze().cpu().numpy().tolist()


def get_text_embedding(model, text: str):
    with torch.no_grad():
        text_embedding = model.encode_text(clip.tokenize(text).to(DEVICE))
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
# 对文本嵌入向量进行归一化处理，使其模长为 1。
    return text_embedding.squeeze().cpu().numpy().tolist()

#批量提取图像的嵌入向量和相关元数据
def fetch_image_embeddings(
    model, preprocess, images: str | list[str], batch_size: int = 8, emb_model=None
):
    print("正在预处理图片...")
    images = [images] if isinstance(images, str) else images
    dataset = ImageDataset(images, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#DataLoader 是PyTorch中的一个实用工具类，它用于将数据集包装成一个可迭代对象，
# 使其能够批量加载数据，提高数据加载的效率。
# 在循环中， dataloader 会按批次返回数据，每次返回一个包含8个样本的批次，
# 这样可以减少内存的使用，提高数据处理的效率
    print("正在提取特征中...", end=" ")
    image_paths = []
    image_embeddings = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            image_path = data["image_path"]
            X = data["image"].to(DEVICE)
            image_embedding = model.encode_image(X)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

            image_paths.extend(image_path)
            image_embeddings.extend(image_embedding.cpu().numpy().tolist())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = [
        {
            "vec": vec,
            "metadata": {
                "type": "image",
                "image_path": ip,
                "embedding_model_name": emb_model,
            },
        }
        for vec, ip in zip(image_embeddings, image_paths)
    #zip 函数是Python的内置函数，它能够接受多个可迭代对象（像列表、元组等）作为参数，
    # 然后返回一个迭代器，该迭代器会生成由每个可迭代对象中相同索引位置的元素组成的元组
    #zip将两个列表中相同索引位置的元素配对，从而在循环里同时访问图像的嵌入向量和对应的文件路径。
    ]
    #data里面是列表推导式
    return data

#批量提取文本的嵌入向量和相关元数据
def fetch_text_embeddings(model, texts: str | list[str], bs: int = 24, emb_model=None):

    print("正在提取关键词中...")
    text_keywords = []
    mw = 1
    with ThreadPoolExecutor(max_workers=mw) as executor:
        for i in tqdm(range(0, len(texts), mw)):
            futures = [executor.submit(summary, text) for text in texts[i : i + mw]]
             # 向线程池提交任务，调用 定义的summary 函数提取每个文本的关键词
            for f in as_completed(futures):
                # 当任务完成时，获取任务的结果
                text, kws = f.result()
                text_keywords.append([text, kws])

    print("正在提取特征中...")
    text_embeddings, tmp_texts, tmp_keywords = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(text_keywords), bs)):
            tmpt = [j[0] for j in text_keywords[i : i + bs]]
            #提取当前批次的文本
            tmpk = [j[1] for j in text_keywords[i : i + bs]]
            #提取当前批次的关键词
            text_embedding = model.encode_text(clip.tokenize(tmpk).to(DEVICE))
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            #对关键词进行编码得到文本的嵌入向量
            text_embeddings.extend(text_embedding.cpu().numpy().tolist())
            tmp_keywords.extend(tmpk)
            tmp_texts.extend(tmpt)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = [
        {
            "vec": vec,
            "metadata": {
                "type": "text",
                "keywords": kw,
                "text": text,
                "embedding_model_name": emb_model,
            },
        }
        for vec, kw, text in zip(text_embeddings, tmp_keywords, tmp_texts)
    ]
   # 使用列表推导式将文本的嵌入向量、关键词和文本组合成字典列表
    # 每个字典包含一个向量和相关的元数据
    return data


if __name__ == "__main__":

    # print(f"Device used: {device}")
    # model, preprocess = clip.load("ViT-B/32", device, jit=False)
    # model.eval()
    # fetch_save_image_embeddings(model, preprocess, "images")

    model, preprocess = load_from_name("RN50", "cpu", download_root="./ckpts")
    model.eval()

    cleaned_image_list = [
        r"D:\notebooks\Code\d-files\D213\easy-image-search-clib\database\images\bird1.jpg",
        r"D:\notebooks\Code\d-files\D213\easy-image-search-clib\database\images\bird2.jpg",
        r"D:\notebooks\Code\d-files\D213\easy-image-search-clib\database\images\bird3.jpg",
    ]
    dataset = ImageDataset(cleaned_image_list, preprocess)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for data in tqdm(dataloader):
        print(data["image"].shape)
        print(len(data["image_path"]))
