
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

import json
import os


def data_collator(features) -> dict:
    max_seq_length=-1
    input_list = []
    labels_list = []
    max_seq_length=-1
    for feature in  features:
        ids= feature["input_ids"]["input"]
        if len(ids)>1000:
            continue
        max_seq_length=max(len(ids),max_seq_length)
    
    for feature in  features:
        ids= feature["input_ids"]["input"]
        if len(ids)>1000:
            continue
        label=feature["input_ids"]["output"]
         #数据长度的补齐：在batch中数据左边补0对齐
        ids = (max_seq_length-len(ids))*[0]+ids
        input_list.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor([label]))
    input_ids = torch.stack(input_list)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

with open("accusation_id",encoding="utf-8") as f:
    accusation_id=json.load(f)
model_name="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_8bit=False, trust_remote_code=True, device_map="auto",num_labels=len(accusation_id))
 
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
 
# LoRA配置
lora_config = LoraConfig(
    r=16,                      # LoRA矩阵的秩
    lora_alpha=16,            # LoRA缩放因子
    target_modules=["q_proj", "v_proj"],  # 要应用LoRA的模块
    lora_dropout=0.1,         # Dropout概率
    bias="none",              # 是否训练偏置
)
 
# 训练参数配置
training_args = TrainingArguments(
    output_dir="./qwen_lora_train",        # 输出目录
    learning_rate=2e-4,                     # 学习率
    per_device_train_batch_size=4,          # 训练批次大小
    gradient_accumulation_steps=4,          # 梯度累积步数
    num_train_epochs=3,                     # 训练轮次
    weight_decay=0.01,                      # 权重衰减
    logging_dir="./logs",                   # 日志目录
    logging_steps=10,                       # 日志记录频率
    save_strategy="steps",                  # 保存策略
    save_steps=30,                        # 保存频率
    fp16=True,                            # 使用混合精度训练
    report_to="tensorboard",
    save_total_limit=2,
)
model.config.pad_token_id=tokenizer.pad_token_id

with open("data/train_data_classify_token",encoding="utf-8")  as f:
    dataset=[ {"input_ids":json.loads(s)} for s in f.readlines()]

#model + lora
model = get_peft_model(model, lora_config) 
print (model)

#使用lora的时候，它会默认把原模型的所有参数都冻结住,重新恢复可训练
for name, param in model.score.named_parameters():
    param.requires_grad=True
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 启动训练
trainer.train()

# save_pretrained 只保存LoRA模型
model.save_pretrained("./qwen_lora_model")
#分类层 需单独保存
torch.save(model.score.state_dict(), "classify_weights.pt")
