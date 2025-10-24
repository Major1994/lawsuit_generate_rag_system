from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

import torch
from tqdm import tqdm
import pickle

import random
import json

with open("accusation_id",encoding="utf-8") as f:
    accusation_id=json.load(f)
id_accusation=dict([ [int(s2),s1] for s1,s2 in accusation_id.items()])

model_name="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#加载训练好的模型模型
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True,num_labels=len(accusation_id)).cuda()

#训练好的lora挂在上去
model = PeftModel.from_pretrained(model, "qwen_lora_model")

#训练好的分类权重,替换原来的
score_weights = torch.load("classify_weights.pt") 
model.score.load_state_dict(score_weights)
model=model.eval()

text="济南市长清区人民检察院指控，2014年2月15日，被告人徐某甲与其父徐某乙、母亲王某某、妻子朱某某和儿子徐某丙五人到济南市长清区孝里镇孝堂山庙会赶会，徐某乙将骑去的摩托三轮车停放在长清区孝里镇富群超市门前的广场，13时许，徐某乙准备驾车离开，因被害人张某甲向其收取停车费，被告人徐某甲同张某甲发生争执，继而相互殴打，徐某甲用拳头将张某甲鼻部打伤，致其鼻骨粉碎性骨折，经鉴定其损伤构成轻伤二级。2014年2月24日，经公安机关电话传唤，被告人徐某甲主动投案，并如实供述其犯罪事实，属自首。2014年3月6日，被告人徐某甲一方与被害人张某甲就民事赔偿问题达成调解协议，共赔偿张某甲各项经济损失35000元。"

tokens = tokenizer.encode(text,truncation=True,return_tensors="pt").to(model.device)
print(tokens)
result=model(tokens).logits
print(result)

p=torch.softmax(result,dim=-1)
print(p)

index=int(torch.argmax(p[0]))
print(index)

predict_label=id_accusation[index]
print(predict_label)
