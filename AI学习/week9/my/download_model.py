from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os
from mlx_lm import load , generate

# 获取脚本所在目录，并创建模型缓存路径
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "../models")

# 在modelscope上下载Qwen模型到本地目录下
# model_dir = snapshot_download("Qwen/Qwen3-0.6B", cache_dir=cache_path, revision="master")
# # Transformers加载模型权重
# tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)

# todo mac电脑单独支持的MLX版本. 使用方式与通用版本的不同.不支持pytorch 和 transformers
# model_dir = snapshot_download("Qwen/Qwen3-1.7B-MLX-8bit", cache_dir=cache_path, revision="master")
model, tokenizer = load(
    "../models/Qwen/Qwen3-1___7B-MLX-8bit"  # 明确指定使用Apple的GPU加速
)
prompt = "请介绍一下机器学习"
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=150
)

print("模型输出:", response)

# while True:
#     try:
#         question = input("👤 请输入您的问题: ").strip()
#
#         if question.lower() in ['quit', 'exit', '退出']:
#             print("👋 再见！")
#             break
#
#         if not question:
#             continue
#
#         response = generate(
#             model=model,
#             tokenizer=tokenizer,
#             prompt=question,
#             max_tokens=150
#         )
#
#         print(f"🤖 {response}")
#
#
#     except KeyboardInterrupt:
#         print("\n👋 再见！")
#         break
#     except Exception as e:
#         print(f"❌ 出现错误: {e}")