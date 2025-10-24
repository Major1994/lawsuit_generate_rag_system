# test_retrieval.py
from langchain_community.vectorstores import FAISS
import os

from langchain_core.embeddings import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer


class CustomEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

def main():
    DB_DIR = "law_db"

    print("加载嵌入模型和 FAISS 数据库...")
    embeddings = CustomEmbeddings('./Qwen3-Embedding-0.6B')
    
    # 使用 allow_dangerous_deserialization=True 是因为保存的是自定义对象
    # 更安全做法：使用 `pickle` 外的序列化方式（如 Chroma + HuggingFace）
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    query = "被告人非法捕猎野生动物，被查获"
    print(f"\n🔍 查询: {query}\n")

    results = db.similarity_search(query, k=3)

    for i, r in enumerate(results, 1):
        print(f"--- 匹配 {i} ---")
        print("案情:", r.page_content[:200] + ("..." if len(r.page_content) > 200 else ""))
        print("罪名:", r.metadata["category"])
        print("判决:", r.metadata["result"])
        print()


if __name__ == "__main__":
    main()
