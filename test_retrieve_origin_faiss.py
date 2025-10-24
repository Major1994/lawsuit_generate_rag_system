# search.py

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch

# ================== 配置 ==================
MODEL_PATH = "./Qwen3-Embedding-8B"
INDEX_FILE = "law_db_native.index"
META_FILE = "law_db_native_meta.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3

# ================== 加载模型 ==================
model = SentenceTransformer(MODEL_PATH)
model = model.to(DEVICE)

# ================== 加载 FAISS 索引 ==================
print("加载 FAISS 索引...")
index = faiss.read_index(INDEX_FILE)

# ================== 加载元数据 ==================
print("加载元数据...")
with open(META_FILE, "rb") as f:
    data = pickle.load(f)
texts = data["texts"]
metadata_list = data["metadata_list"]

# ================== 检索函数 ==================
def search(query: str, k: int = 3):
    # 生成查询向量
    query_vec = model.encode([query], convert_to_numpy=True, device=DEVICE)
    
    # FAISS 要求是 float32
    query_vec = np.array(query_vec, dtype=np.float32)
    
    # 搜索
    distances, indices = index.search(query_vec, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # 无结果
        results.append({
            "text": texts[idx],
            "metadata": metadata_list[idx],
            "distance": distances[0][i]  # L2 距离，越小越相似
        })
    return results

# ================== 测试 ==================
query = "被告人非法捕猎果子狸和野猫"
print(f"查询: {query}")
results = search(query, TOP_K)

for i, res in enumerate(results):
    print(f"\n--- 匹配 {i+1} ---")
    print("案情:", res["text"][:200] + "...")
    print("罪名:", res["metadata"]["category"])
    print("判决:", res["metadata"]["result"])
    print("距离:", res["distance"])
