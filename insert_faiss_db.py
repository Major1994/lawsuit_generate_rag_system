import numpy as np
import faiss
import json
import pickle
from sentence_transformers import SentenceTransformer
import torch
import gc
from tqdm import tqdm
# ================== 配置 ==================
MODEL_PATH = "./Qwen3-Embedding-8B"
DATA_PATHS = ["data/test_data.jsonl"]# ["data/train_data.jsonl", "data/rest_data.jsonl"]
OUTPUT_INDEX = "law_db_native.index"
OUTPUT_METADATA = "law_db_native_meta.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # 根据显存调整：1, 2, 4...

# ================== 加载模型 ==================
print("加载嵌入模型...")
model = SentenceTransformer(MODEL_PATH)
model = model.to(DEVICE)
# model = model.half()  # 使用 float16，减少显存占用

# ================== 加载数据（同前）==================
def load_data(paths):
    lines = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line.strip()))
    return lines

print("加载法律数据...")
lines = load_data(DATA_PATHS)

def accusation_json2str(data):
    accusation_list=data["罪名"]
    result="判决结果："
    result+="罪名:"+"，".join(accusation_list)+"，"
    result+="罚金:"+str(data["罚金"])+"元，"
    result+="罪犯:"+",".join(data["犯罪嫌疑人"])+"，"
    if data["是否死刑"]:
        result+="刑期：死刑"
    elif data["是否无期"]:
        result+="刑期：无期徒刑"
    else:
        result+="刑期：有期徒刑"+str(data["有期徒刑"])+"个月"
    return result

texts = []
metadata_list = []
for data in lines:
    text = data["input"]
    if len(text) > 2000:
        continue

    result_str = accusation_json2str(data["output"])

    for accusation in data["output"]["罪名"]:
        texts.append(text)
        metadata_list.append({
            "result": result_str,
            "category": accusation
        })

print(f"共 {len(texts)} 个文本待编码")

# ================== 分批生成嵌入 ==================
def encode_in_batches(model, texts, batch_size=4, device="cuda"):
    all_embeddings = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        total=(len(texts) - 1) // batch_size + 1,
        unit="batch"
    ):
        batch = texts[i:i+batch_size]
        # print(f"编码 batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        with torch.no_grad():  # 禁用梯度，节省内存
            emb = model.encode(batch, convert_to_numpy=True, device=device)
        all_embeddings.append(emb)
        # 可选：每批后清理
        # torch.cuda.empty_cache()
        # gc.collect()
    return np.concatenate(all_embeddings, axis=0)

print("开始分批生成嵌入...")
embeddings = encode_in_batches(model, texts, batch_size=BATCH_SIZE, device=DEVICE)
print(f"嵌入形状: {embeddings.shape}")

# ================== 构建 FAISS 索引 ==================
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype('float32'))  # FAISS 要求 float32

# ================== 保存 ==================
faiss.write_index(index, OUTPUT_INDEX)
with open(OUTPUT_METADATA, "wb") as f:
    pickle.dump({"texts": texts, "metadata_list": metadata_list}, f)

print("构建完成")
