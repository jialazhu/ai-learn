#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗助手集成脚本
基于 Qwen3-1.7B 医疗微调模型，提供多种医疗场景的智能助手功能
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
from datetime import datetime
import os

# 医疗专业提示词模板
MEDICAL_PROMPTS = {
}

# 常见医疗场景
MEDICAL_SCENARIOS = {
}

# 预设问题示例
SAMPLE_QUESTIONS = {
    "中医": [
        "创建了IgA肾病从虚、瘀、风湿辨治体系并提出IgA肾病五型辨证治疗新方案的哪位著名中医？？",
        "清朝哪本书提出了温病和时疫的防治原则及方法，形成了中医药防治瘟疫（传染病）的理论和实践体系？",
        "轻可去实，故疗伤寒，为解肌第一“描述的是哪一味中药？"
    ],
    "犯罪": [
        "凌迟这种刑罚在中国正式被废除是哪一年？",
        "在2015年熊谷市连环杀人案中，凶手使用的武器是什么？",
        "‘细蓝线’一词是仿照哪场战争中形容英国步兵的‘细红线’而造出的？"
    ],
    "漫画": [
        "漫画《我的朋友世界第一可爱》的第1本单行本在台湾由哪个出版社发售？",
        "《驱魔少年》漫画中亚连·沃克最初使用的武器是什么？",
        "拉斐尔这个角色首次登场的《忍者龟》漫画是哪一年发行的？"
    ],
    "电影": [
        "在洛迦诺电影节中，被国际电影制片人协会认可的最高荣誉奖项是什么？",
        "在电影《玩具总动员2》中，谁为角色翠丝配音？",
        "成龙国际动作电影周首次独立举行是在哪一年？"
    ]
}

class MedicalAssistant:
    def __init__(self, checkpoint_path="../output/Qwen3-0.6B-chinese/checkpoint-1350"):
        """初始化问答小助手"""
        self.checkpoint_path = checkpoint_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def _select_device_and_dtype(self):
        """选择设备和数据类型"""
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 12:
                    raise RuntimeError("Unsupported CUDA capability for current PyTorch")
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32
    
    def load_model(self):
        """加载模型和分词器"""
        print("正在加载问答小助手模型...")
        
        # 检查路径是否存在
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, 
            use_fast=False, 
            trust_remote_code=True,
            local_files_only=True  # 只使用本地文件
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, 
            torch_dtype=self.dtype,
            local_files_only=True  # 只使用本地文件
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成！使用设备: {self.device}")
    
    def predict(self, messages, max_new_tokens=512):
        """执行预测"""
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        # 只解码新生成部分
        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response
    
    def ask_question(self, question, scenario_choice="中华文化", sub_choice="中国神话", max_tokens=512):
        """询问医疗问题"""
        if scenario_choice not in MEDICAL_PROMPTS:
            scenario_type = "中华文化"
        content = f"{MEDICAL_PROMPTS[scenario_choice]},研究{sub_choice},你需要根据用户的问题，给出答案。"
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": question}
        ]
        
        # 记录对话历史
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_choice,
            "question": question,
            "response": None
        })
        
        response = self.predict(messages, max_new_tokens=max_tokens)
        
        # 更新对话历史
        self.conversation_history[-1]["response"] = response
        
        return response
    
    def show_scenarios(self):
        """显示可用的医疗场景"""
        print("\n🏥 问答小助手 - 可用场景:")
        print("=" * 50)
        for key, _ in MEDICAL_PROMPTS.items():
            print(f"{key:2}")
        print("=" * 50)
    def show_sub_scenarios(self, scenario_type):
        """显示可用的子场景"""
        print("\n🏥 问答小助手 - 可用子场景:")
        print("=" * 50)
        for value in list(MEDICAL_SCENARIOS[scenario_type]):
            print(f"{value}")
        print("=" * 50)
    
    def show_sample_questions(self, scenario_type):
        """显示示例问题"""
        if scenario_type in SAMPLE_QUESTIONS:
            print(f"\n📋 {MEDICAL_SCENARIOS.get(scenario_type, '中华文化')} - 示例问题:")
            print("-" * 40)
            for i, question in enumerate(SAMPLE_QUESTIONS[scenario_type], 1):
                print(f"{i}. {question}")
            print("-" * 40)
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🤖 问答小助手已启动！")
        print("输入 'help' 查看帮助，输入 'quit' 退出")

        keep = False
        scenario_choice = None
        sub_choice = None
        while True:
            try:
                if not keep:
                    # 显示场景选择
                    self.show_scenarios()

                    # 选择场景
                    scenario_choice = input("\n请选择问答场景: ").strip()
                    if scenario_choice == 'quit':
                        break
                    elif scenario_choice == 'help':
                        self.show_help()
                        continue
                    elif scenario_choice not in list(MEDICAL_PROMPTS.keys()):
                        print("❌ 无效选择，请重新输入")
                        continue

                    self.show_sub_scenarios(scenario_choice)

                    sub_choice = input(f"\n请选择{scenario_choice}子场景: ").strip()
                    if sub_choice == 'quit':
                        break
                    elif sub_choice == 'help':
                        self.show_help()
                        continue
                    elif sub_choice not in list(MEDICAL_SCENARIOS[scenario_choice]):
                        print("❌ 无效选择，请重新输入")
                        continue
                
                # 显示示例问题
                self.show_sample_questions(sub_choice)
                
                # 获取用户问题
                # question = input(f"\n请输入您的{MEDICAL_SCENARIOS[scenario_choice]}问题: ").strip()
                question = input(f"\n请输入您的问题: ").strip()
                if not question:
                    print("❌ 问题不能为空")
                    continue
                
                # 生成回答
                print("\n🔄 正在分析您的问题...")
                start_time = time.time()
                
                response = self.ask_question(question, scenario_choice,sub_choice)
                
                end_time = time.time()
                
                # 显示回答
                elapsed_time = end_time - start_time
                print(f"\n💡 问答小助手回答 (耗时: {elapsed_time:.2f}秒):")
                print("=" * 60)
                print(response)
                print("=" * 60)
                
                # 询问是否继续
                continue_choice = input("\n是否继续咨询？(y/n),如果要重新选择场景请输入 again : ").strip().lower()
                if continue_choice == 'again':
                    keep = False
                    scenario_choice = None
                    sub_choice = None
                    continue
                elif continue_choice in ['n', 'no', '否']:
                    keep = False
                    scenario_choice = None
                    sub_choice = None
                    break
                else:
                    keep = True
                    
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用问答小助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                continue
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 问答小助手使用帮助:")
        print("=" * 50)
        print("1. 选择医疗场景 (1-10)")
        print("2. 输入您的医疗问题")
        print("3. 获得专业的医疗建议")
        print("\n💡 提示:")
        print("- 本助手仅提供参考建议，不能替代专业医疗诊断")
        print("- 紧急情况请立即就医")
        print("- 输入 'quit' 退出程序")
        print("=" * 50)
    
    def save_conversation(self, filename=None):
        """保存对话历史"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_conversation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"💾 对话历史已保存到: {filename}")
    
    def batch_questions(self, questions_file):
        """批量处理问题"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            print(f"📝 开始批量处理 {len(questions)} 个问题...")
            
            results = []
            for i, q in enumerate(questions, 1):
                print(f"\n处理第 {i}/{len(questions)} 个问题...")
                response = self.ask_question(
                    q.get('question', ''), 
                    q.get('scenario', 'diagnosis'),
                    q.get('max_tokens', 512)
                )
                
                results.append({
                    "question": q.get('question', ''),
                    "scenario": q.get('scenario', 'diagnosis'),
                    "response": response
                })
            
            # 保存结果
            output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 批量处理完成！结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 批量处理失败: {str(e)}")

def load_category(origin_path):
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            primary_category = data["category"]
            sec_category = data["sec_category"]
            MEDICAL_PROMPTS[primary_category] = f"你是一个{primary_category}学派学者"
            MEDICAL_SCENARIOS[primary_category] = sec_category


def main():
    parser = argparse.ArgumentParser(description="问答小助手 - 基于Qwen3-0.6B的智能百科咨询系统")
    parser.add_argument("--checkpoint", "-c", type=str, 
                       default="../output/Qwen3-0.6B-chinese-v1/checkpoint-1350",
                       help="模型检查点路径")
    parser.add_argument("--question", "-q", type=str, 
                       help="直接询问问题（需要配合 --scenario 使用）")
    parser.add_argument("--scenario", "-s", type=str, 
                       default="diagnosis", 
                       choices=list(MEDICAL_PROMPTS.keys()),
                       help="百科咨询场景类型")
    parser.add_argument("--max-tokens", "-m", type=int, 
                       default=512, 
                       help="最大生成token数")
    parser.add_argument("--batch", "-b", type=str, 
                       help="批量处理问题文件（JSON格式）")
    parser.add_argument("--save-history", action="store_true", 
                       help="保存对话历史")
    
    args = parser.parse_args()
    load_category('../dataSets/chinese-category.jsonl')
    # 创建问答小助手实例
    assistant = MedicalAssistant(args.checkpoint)
    
    # 加载模型
    assistant.load_model()
    
    if args.batch:
        # 批量处理模式
        assistant.batch_questions(args.batch)
    elif args.question:
        # 单次问答模式
        print(f"🤖 问答小助手回答:")
        print("=" * 50)
        response = assistant.ask_question(args.question, args.scenario, args.max_tokens)
        print(response)
        print("=" * 50)
    else:
        # 交互模式
        assistant.interactive_mode()
    
    # 保存对话历史
    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()
