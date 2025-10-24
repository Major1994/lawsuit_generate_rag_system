# main.py

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from json_output import parser  # 假设 parser 是 StructuredOutputParser
import torch
from tqdm import tqdm
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI
import random
import os
import dashscope

# ================== 配置 ==================
DASHSCOPE_API_KEY = "sk-af51f271a37748439c658f3882f4d969"
EMBEDDING_MODEL_PATH = "/root/autodl-tmp/Qwen3-Embedding-8B"
VECTOR_DB_PATH = "law_db_native"  # 对应之前保存的 native FAISS
CASE_NUM = 3

# ================== 初始化大模型 ==================
dashscope.api_key = DASHSCOPE_API_KEY

def think(prompt, stream=False):
    """调用 Qwen3 生成司法解释"""
    messages = [
        {'role': 'system', 'content': '你是一个资深的法律顾问，擅长撰写正式、严谨的司法解释。'},
        {'role': 'user', 'content': prompt}
    ]
    response = dashscope.Generation.call(
        model="qwen3-30b-a3b",
        messages=messages,
        result_format='message',
        stream=stream,
        enable_thinking=False
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(f"API 调用失败: {response.message}")

# ================== 加载向量数据库（推荐方式）==================
def load_vector_db():
    print("加载嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}  # 注意：内积相似度需归一化
    )
    print("加载向量数据库...")
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True  # 允许加载 pickle
    )
    return db

db = load_vector_db()

# ================== OutputFixingParser 初始化 ==================
try:
    from langchain_core.output_parsers import OutputFixingParser
except ImportError:
    from langchain.output_parsers import OutputFixingParser

llm_for_fix = ChatOpenAI(
    model="qwen2.5-32b-instruct",
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=DASHSCOPE_API_KEY,
    temperature=0.3
)
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_for_fix)

# ================== 罪名预测模块（假设 predict_accusation.py 提供）==================
# 注意：确保 predict_accusation.py 中有 predict_accusation(text) 函数
from predict_accusation import predict_accusation

# ================== 辅助函数 ==================
def accusation_json2str(data):
    """将判决 JSON 转为自然语言字符串"""
    result = "判决结果："
    result += "罪名：" + "、".join(data.get("罪名", ["未知"])) + "；"
    result += "罚金：" + str(data.get("罚金", 0)) + "元；"
    result += "罪犯：" + "、".join(data.get("犯罪嫌疑人", ["未知"])) + "；"
    if data.get("是否死刑", False):
        result += "刑期：死刑"
    elif data.get("是否无期", False):
        result += "刑期：无期徒刑"
    else:
        result += f"刑期：有期徒刑{data.get('有期徒刑', 0)}个月"
    return result

def parser_documents(documents):
    """格式化检索到的案例，用于 prompt"""
    result = ""
    for i, doc in enumerate(documents):
        text = doc.page_content + " 判决结果：" + doc.metadata["result"] + "\n\n"
        result += f"案例{i}：{text}"
    return result

def merge_result(all_result):
    """合并多次推理结果：布尔值取或，数值取平均"""
    death = any(r["是否死刑"] for r in all_result)
    life_imprisonment = any(r["是否无期"] for r in all_result)
    
    months = [r["有期徒刑"] for r in all_result if not r["是否死刑"] and not r["是否无期"]]
    fines = [r["罚金"] for r in all_result]
    
    avg_month = int(sum(months) / len(months)) if months else 0
    avg_fine = int(sum(fines) / len(fines)) if fines else 0

    # 返回第一个结果并更新
    final = all_result[0].copy()
    final["是否死刑"] = death
    final["是否无期"] = life_imprisonment
    final["有期徒刑"] = avg_month
    final["罚金"] = avg_fine
    final["罪名"] = list(set(acc for r in all_result for acc in r["罪名"]))  # 去重合并罪名
    return final

# ================== 主预测函数 ==================
def predict(text, num=3):
    """
    输入案情，输出判决 + 相似案例
    """
    print("🔍 正在预测罪名...")
    predict_label = predict_accusation(text)
    print(f"✅ 预测罪名: {predict_label}")

    print("🔍 正在检索相似案例...")
    similar_docs = db.similarity_search(
        text,
        k=10,
        filter={"category": predict_label}
    )[:3]

    print(similar_docs)
    
    if not similar_docs:
        print("⚠️ 未找到相似案例，使用通用案例...")
        similar_docs = db.similarity_search(text, k=3)

    case_list = parser_documents(similar_docs)

    print(f"📊 基于 {len(similar_docs)} 个相似案例进行推理...")
    all_result = []
    for i in tqdm(range(num), desc="🧠 生成判决推理"):
        prompt = case_list + f"\n请根据上述案例，对以下案件做出判决：\n{text}\n{parser.get_format_instructions()}"
        try:
            response = llm_for_fix.invoke(prompt)
            result = fix_parser.parse(response.content)
            result["罪名"] = [predict_label]
            all_result.append(result)
        except Exception as e:
            print(f"❌ 第{i+1}次推理失败: {e}")
            continue

    if not all_result:
        raise Exception("所有推理均失败")

    final_result = merge_result(all_result)
    return final_result, case_list

# ================== 主程序 ==================
if __name__ == "__main__":
    test_text = (
        "公诉机关起诉书指控并审理经查明：2017年8月24日10时许，被告人刘某在本市丰台区木樨园长途客运站出站闸机处 "
        "不配合检查工作，用随身携带的小推车冲撞民警、用雨伞击打民警头部，在民警及辅警对其进行控制时，"
        "对民警及辅警进行辱骂、抓挠、撕扯，造成民警张1某右前臂、双下肢多处挫伤，辅警龙某右上肢皮肤挫伤。"
        "经鉴定，张1某、龙某二人的损伤程度均为轻微伤。\n"
        "公诉机关建议判处被告人刘某××至一年。被告人刘某对指控事实、罪名及量刑建议没有异议且签字具结，"
        "在开庭审理过程中亦无异议。"
    )

    print("🚀 开始预测...\n")
    result_json, case_list = predict(test_text, num=3)

    print("\n" + "="*50)
    print("📌 相似案例：")
    print(case_list)

    result_str = accusation_json2str(result_json)
    print("\n" + "="*50)
    print("⚖️  判决结果：")
    print(result_str)

    print("\n" + "="*50)
    print("📜 生成司法解释...")
    explanation_prompt = test_text + "\n" + result_str + "\n\n请根据上述案情和判决结果，给出合理的司法解释。"
    explanation = think(explanation_prompt, stream=False)
    print("司法解释：")
    print(explanation)
