from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os

# 获取脚本所在目录，并创建模型缓存路径
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "../models")

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen3Guard-Gen-0.6B", cache_dir=cache_path, revision="master")
# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)


def predict(message,model,tokenizer):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    text = tokenizer.apply_chat_template(
        message,tokenize=False,add_generation_prompt=True)

    model_inputs = tokenizer([text],return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

test_texts = {
    'instruction': "你是一个专业助手，你需要根据用户的问题，给出带有思考的回答。",
    'input': "傻逼"
}
instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "assistant", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages,model,tokenizer)
print(response)