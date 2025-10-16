import math
from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import warnings
warnings.filterwarnings("ignore", message=".*is not compiled with.*")
import random
import numpy as np
import argparse

# ========== 第一部分：数据预处理练习 ==========

QWEN_CHAT_TEMPLATE = (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

@dataclass
class Collator:
    """
    数据整理器，用于将批次中的样本填充（padding）到相同的长度。
    填充操作应用于 input_ids, attention_mask 和 labels。
    input_ids 和 labels 用 pad_token_id 填充，attention_mask 用 0 填充。

    Args:
        pad_token_id (int): tokenizer 的填充 token ID。
    """
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids, attn_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - x["input_ids"].size(0)
            input_ids.append(torch.nn.functional.pad(x["input_ids"], (0, pad_len), value=self.pad_token_id))
            attn_mask.append(torch.nn.functional.pad(x["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(x["labels"], (0, pad_len), value=-100))
        return {
            "input_ids": torch.stack(input_ids),  # 批次中的所有 input_ids 堆叠成一个张量
            "attention_mask": torch.stack(attn_mask),  # 批次中的所有 attention_mask 堆叠成一个张量
            "labels": torch.stack(labels),  # 批次中的所有 labels 堆叠成一个张量
        }

class SFTDataset(Dataset):
    """
    练习1：完成SFT数据集的实现
    要求：
    1. 实现__init__方法，接收原始数据并存储
    2. 实现__len__方法
    3. 实现__getitem__方法，返回处理后的样本
    """
    
    def __init__(self, data, tokenizer, max_length=256):
        """
        初始化数据集
        Args:
            data: 原始数据列表，每个元素包含instruction和output
            tokenizer: 分词器
            max_length: 最大序列长度（建议使用256避免维度问题）
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # TODO: 在这里添加你的初始化代码
        # 提示：可以预处理数据以提高效率，或者保持简单在__getitem__中处理
        self.examples = []
        for d in self.data:
            instruction = d["instruction"]
            output = d["output"]

            prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)
            response_ids = self.tokenizer(output + self.tokenizer.eos_token, add_special_tokens=False)

            input_ids = prompt_ids["input_ids"] + response_ids["input_ids"]
            attn_mask = prompt_ids["attention_mask"] + response_ids["attention_mask"]

            labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"]

            # 截断
            input_ids = input_ids[:self.max_length]
            attn_mask = attn_mask[:self.max_length]
            labels = labels[:self.max_length]
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        """返回数据集大小"""
        # TODO: 实现这个方法
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        要求：
        1. 获取instruction和output
        2. 使用QWEN_CHAT_TEMPLATE格式化
        3. 分别对instruction和output进行分词
        4. 构造labels，instruction部分设为-100
        5. 处理长度截断
        """
        # 对话模板

        
        # TODO: 实现数据预处理逻辑
        # 提示：参考以下步骤
        # 1. 获取当前样本的instruction和output
        # 2. 使用模板格式化instruction
        # 3. 分别对格式化后的instruction和output进行分词
        # 4. 拼接input_ids和attention_mask
        # 5. 构造labels：instruction部分为-100，output部分为对应的token ids
        # 6. 处理长度截断（重要：确保不超过max_length）
        # 7. 确保所有张量长度一致
        
        # 示例实现提示：
        # instruction = self.data[idx]["instruction"]
        # output = self.data[idx]["output"]
        # formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
        # prompt_ids = self.tokenizer(formatted_prompt, add_special_tokens=False)
        # response_ids = self.tokenizer(output + self.tokenizer.eos_token, add_special_tokens=False)
        # ... 继续实现
        return self.examples[idx]

# ========== 第二部分：训练函数练习 ==========
def train_one_epoch(model, dataloader, optimizer, device):
    """
    练习2：实现一个训练轮次
    要求：
    1. 设置模型为训练模式
    2. 遍历dataloader
    3. 计算损失并反向传播
    4. 返回平均损失
    """
    model.train()
    total_loss = 0
    
    # TODO: 实现训练循环
    # 提示：
    # 1. 使用for循环遍历dataloader
    # 2. 将数据移动到指定设备
    # 3. 清零梯度
    # 4. 前向传播并计算损失
    # 5. 反向传播
    # 6. 更新参数
    # 7. 累加损失
    # 8. 添加异常处理以避免维度不匹配错误
    
    # 示例实现提示：
    # for batch in dataloader:
    #     try:
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         labels = batch["labels"].to(device)
    #         optimizer.zero_grad()
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     except RuntimeError as e:
    #         print(f"批次处理出错: {e}")
    #         continue

    for batch in dataloader:
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        except RuntimeError as e:
            print(f"批次处理出错: {e}")
            continue
    return total_loss / max(len(dataloader), 1)

# ========== 第三部分：推理函数练习 ==========
def generate_response(model, tokenizer, instruction, device, max_length):
    """
    练习3：实现模型推理函数
    要求：
    1. 使用对话模板格式化输入
    2. 进行分词
    3. 生成回复
    4. 解码并返回结果
    """
    model.eval()
    
    # TODO: 实现推理逻辑
    # 提示：
    # 1. 使用模板格式化instruction
    # 2. 使用tokenizer进行分词
    # 3. 将输入移动到设备上
    # 4. 使用model.generate()生成回复
    # 5. 解码生成的token ids
    
    # 示例实现提示：
    # with torch.no_grad():
    #     formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
    #     inputs = tokenizer(formatted_prompt, return_tensors="pt")
    #     input_ids = inputs["input_ids"].to(device)
    #     attention_mask = inputs["attention_mask"].to(device)
    #     generated_ids = model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         max_new_tokens=max_length,
    #         do_sample=True,
    #         temperature=0.7,
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    #     new_tokens = generated_ids[0][input_ids.shape[1]:]
    #     response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #     return response

    with torch.no_grad():
        print(f"\n{'=' * 20} 开始推理演示 {'=' * 20}")
        formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        print(f"输入序列长度: {len(inputs['input_ids'][0])}")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = generated_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response

# ========== 第四部分：主函数练习 ==========
def main():
    """
    练习4：完成主函数的实现
    要求：
    1. 加载模型和分词器
    2. 准备训练数据
    3. 创建数据集和数据加载器
    4. 设置优化器
    5. 进行训练
    6. 测试推理效果
    """
    
    # TODO: 实现主函数逻辑
    # 提示：
    # 1. 加载Qwen模型和分词器（建议使用较小的模型如Qwen2.5-0.5B-Instruct）
    # 2. 准备一些简单的训练数据（至少3个样本）
    # 3. 创建SFTDataset实例（使用max_length=256）
    # 4. 创建DataLoader
    # 5. 设置AdamW优化器
    # 6. 调用train_one_epoch进行训练
    # 7. 使用generate_response测试模型效果
    
    # 示例实现提示：
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, 
    #     torch_dtype=torch.float32,
    #     trust_remote_code=True
    # )
    # model = model.to(device)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # ... 继续实现
    
    # 示例训练数据格式
    training_data = [
        {
            "instruction": "用两句话解释什么是机器学习",
            "output": "机器学习是人工智能的一个分支，它让计算机能够从数据中自动学习规律。通过算法训练，机器可以识别模式并做出预测或决策。"
        },
        {
            "instruction": "什么是深度学习？",
            "output": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。它能够自动提取数据的特征，在图像识别、自然语言处理等领域表现优异。"
        },
        {
            "instruction": "解释一下监督学习和无监督学习的区别",
            "output": "监督学习使用带标签的训练数据，目标是学习输入和输出之间的映射关系。无监督学习则处理没有标签的数据，主要任务是发现数据中的隐藏模式或结构。"
        }
    ]
    parser = argparse.ArgumentParser(description="SFT Loss Masking Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)  # 对于大模型，先用较少 Epoch 进行快速验证
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)  # 增加最大序列长度以适应Qwen模型
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 设置随机种子，以确保实验的可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(args.seed)

    # 设备选择
    device = "mps" if torch.mps.is_available() else "cpu"
    device = "cpu"
    print(f"当前运行设备: {device.upper()}")

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Qwen tokenizer 的 eos_token 就是其 <|im_end|>，可直接作为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer 加载完成。pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")

    dataset = SFTDataset(training_data, tokenizer, max_length=args.max_length)
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    print(f"\n加载模型: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16 if device == "mps" else torch.float32)

    model.to(device)
    model.train()  # 将模型设置为训练模式
    print(f"模型已加载到 {device}, 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    optim = AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, loader, optim, device)
        try:
            ppl = math.exp(avg_loss)  # 计算困惑度 (Perplexity)
        except OverflowError:  # 处理损失过大导致 ppl 溢出的情况
            ppl = float('inf')
        print(f"Epoch {epoch + 1} 结束: 平均损失={avg_loss:.4f}, 困惑度 (PPL)={ppl:.2f}")

    target = "请用一句话说明深度学习的优势"  # 目标问题
    generate = generate_response(model, tokenizer, target,device, args.max_length)
    print(f"生成结果: {generate}")





if __name__ == "__main__":
    main()