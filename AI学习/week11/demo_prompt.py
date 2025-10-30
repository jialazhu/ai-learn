import os
import warnings
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 设置HF镜像
# 设置环境变量禁止联网
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 关闭TF警告
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 关闭tokenizer并行警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error" # 关闭transformers警告
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#模型和分词器初始化
model_name = "Qwen/Qwen1.5-1.8B-Chat"
# model_name = "../week9/models/Qwen/Qwen3-0.6B"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True
                                                 , torch_dtype= torch.float16, device_map="cuda")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

#构建pipeline简化模型调用流程
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id = tokenizer.eos_token_id, # 设置pad_token 为eos_token 处理批处理长度不一致情况
    return_full_text=False # 返回完整的文本
)

problem = """
在一个果园当中,第一天采摘了总苹果数的1/5,第二天采摘了剩下苹果数的1/4,第三天采摘了剩下苹果数的1/3.
采摘完三天后,果园还剩下360个苹果.
请问,果园里面原来一共多少个苹果?
"""
print("="*50)
print("原始问题")
print( problem.strip())
print("="*50)

# demo1 零样本提示 zero-shot prompt
zero_shot_prompt = f"""问题:{problem}
要求解答,提示:设置未知数.列方程.求解答案
"""

message = [
    {
        "role":"user","content":zero_shot_prompt
    }
]

print("\n--------零样本提示-----------\n")

try:
    result_zero_shot = pipe(
        message,
        max_new_tokens= 400,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
        top_p = 0.95
    )
    print(result_zero_shot[0]["generated_text"])
    print("\n" + "-"*50)
except Exception as e:
    print(f"模型零样本提示调用失败: {e}")


# demo2 少样本提示 few-shot cot
few_shot_cot_prompt = f"""
要求你学习以下思维逆推范例:
「范例」
问题: 仓库第一天运走货物总数的1/3, 第二天运走货物数的1/2,最后剩下250吨,问原来一共有多少吨?
解答: 逆向推理
- 第二天运走后还剩下250吨
- 第二天之前: 250 / (1 - 1/2) = 500吨
- 第一天之前: 500 / (1 - 1/3) = 750吨
答案: 原来一共有750吨
---请你用相同的格式解决以下问题:
{problem}
"""

message = [
    {
        "role":"user","content":few_shot_cot_prompt
    }
]

print("\n--------少样本提示-----------\n")
try:
    result_few_shot = pipe(
        message,
        max_new_tokens= 400,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
        top_p = 0.95
    )
    print(result_few_shot[0]["generated_text"])
    print("\n" + "-"*50)
except:
    print("模型少样本提示调用失败")


#demo3 引导式的逆向推理
guide_prompt = f"""
问题:{problem}
从结果倒推:
1.最后还剩下360个苹果
2.第三天采摘之前: 360 / (1 - 1/3) = a
3.第二天之前: a / (1 - 1/4) = b
4.第一天之前: b / (1 - 1/5) = c
其中c就是要求得的答案.请你计算每步的具体数值,并得到最终答案c.
"""

message = [
    {
        "role":"user","content":guide_prompt
    }
]
print("\n--------引导式逆向推理-----------\n")
try:
    result_guide = pipe(
        message,
        max_new_tokens= 400,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
        top_p = 0.95
    )
    print(result_guide[0]["generated_text"])
    print("\n" + "-"*50)
except:
    print("模型引导式逆向推理调用失败")


# demo4 直接给出正确答案
direct_prompt = f"""
问题:{problem}
直接给出答案:
答案: 原来一共有900个苹果
"""
message = [
    {
        "role":"user","content":direct_prompt
    }
]
print("\n--------直接给出正确答案-----------\n")
try:
    result_direct = pipe(
        message,
        max_new_tokens= 400,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
        top_p = 0.95
    )
    print(result_direct[0]["generated_text"])
    print("\n" + "-"*50)
except:
    print("模型直接给出正确答案调用失败")









