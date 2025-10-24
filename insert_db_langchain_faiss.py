# main.py

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json

# 初始化嵌入模型
class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档列表转换为嵌入向量列表"""
        # 使用 SentenceTransformer 的 encode 方法（自动处理 tokenization）
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """将查询文本转换为嵌入向量"""
        embedding = model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()
embeddings = CustomEmbeddings()

# 加载数据
def load_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

lines1 = load_lines("data/train_data.jsonl")
lines2 = load_lines("data/rest_data.jsonl")
lines = lines1 + lines2

# 构建文档
documents = []
for data in lines:
    text = data["input"]
    # 可选：用 tokenizer 更精确控制长度
    if len(text) > 2000:
        continue

    result_str = predict.accusation_json2str(data["output"])  # 假设返回字符串
    accusation_list = data["output"]["罪名"]

    for accusation in accusation_list:
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "result": result_str,
                    "category": accusation
                }
            )
        )

print(f"共处理 {len(documents)} 个文档，开始构建 FAISS 索引...")

# 构建向量数据库
db = FAISS.from_documents(documents, embeddings)

# 正确保存
db.save_local("law_db")
print("FAISS 数据库已保存到 'law_db' 目录")
##############################################################################
# test_retrieval.py
from langchain_community.vectorstores import FAISS
from Embedding import CustomEmbeddings

embeddings = CustomEmbeddings()
db = FAISS.load_local("law_db", embeddings, allow_dangerous_deserialization=True)

query = "被告人非法捕猎野生动物，被查获"
results = db.similarity_search(query, k=3)

for r in results:
    print("案情:", r.page_content[:200] + "...")
    print("罪名:", r.metadata["category"])
    print("判决:", r.metadata["result"])
    print("---")
