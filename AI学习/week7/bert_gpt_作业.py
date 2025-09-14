import torch
from sympy.physics.units import temperature
from torch.optim import AdamW
from torchgen.api.cpp import return_type
from transformers import AutoTokenizer,BertModel,GPT2Model
from transformers import BertForSequenceClassification , GPT2LMHeadModel
import warnings
warnings.filterwarnings("ignore")

class BERTGPTDEMO:
    def __init__(self):
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        print(f"使用设备:{self.device}")

    def load_models(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert_model = BertModel.from_pretrained("bert-base-chinese").to(self.device)
        self.bert_classifier = BertForSequenceClassification.from_pretrained("bert-base-chinese"
                                                                             ,num_labels = 2).to(self.device)
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt_base = GPT2Model.from_pretrained("gpt2").to(self.device)
        self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token

        print("模型加载完成")


    def compare_architectures(self):
        """1. 架构对比分析"""
        print("\n=== 1. 架构对比分析 ===")

        architecture_comparison = {
            "核心架构": {
                "BERT": "双向Transformer编码器",
                "GPT": "单向Transformer解码器"
            },
            "注意力机制": {
                "BERT": "全词注意力（无掩码）",
                "GPT": "因果掩码自注意力"
            },
            "输入处理": {
                "BERT": "[CLS] + 句子对",
                "GPT": "序列自回归"
            },
            "输出方式": {
                "BERT": "上下文表征向量",
                "GPT": "下一个token预测"
            }
        }

        print("架构对比:")
        for aspect, models in architecture_comparison.items():
            print(f"\n{aspect}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")

    def compare_pretraining_tasks(self):
        """2. 预训练任务对比"""
        print("\n=== 2. 预训练任务对比 ===")

        task_comparison = {
            "主要任务": {
                "BERT": "MLM（掩码语言建模）+ NSP（下一句预测）",
                "GPT": "自回归语言建模（预测下一词）"
            },
            "训练目标": {
                "BERT": "学习双向上下文表征",
                "GPT": "学习生成概率分布"
            },
            "数据利用": {
                "BERT": "利用左右上下文",
                "GPT": "利用前序上下文"
            },
            "泛化能力": {
                "BERT": "强理解能力",
                "GPT": "强生成能力"
            }
        }

        print("预训练任务对比:")
        for aspect, models in task_comparison.items():
            print(f"\n{aspect}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")

    def bert_bidirectional(self):
        print("\n=== 3. BERT分词演示 ===")
        text = "中国的首都北京是一个现代化大都市"

        tokens = self.bert_tokenizer.tokenize(text)
        print(f"分词结果: {tokens}")

        inputIds = self.bert_tokenizer.encode(text,add_special_tokens=True)
        print(f"输入ID: {inputIds}")
        print(f"特殊[CLS]:{self.bert_tokenizer.cls_token_id},特殊[SEP]:{self.bert_tokenizer.sep_token_id}")

    def bert_demonstrate_tokenization(self):
        print("\n=== 4. BERT分词演示 ===")

        text = "中国的首都北京是一个现代化大都市"
        mask_text= "中国的首都[MASK]是一个现代化大都市"

        print(f"原始文本: {text}")
        print(f"掩码文本: {mask_text}")

        # 转换为ID
        input_ids = self.bert_tokenizer(mask_text,return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.bert_classifier(**input_ids)
            logits = outputs.logits
            print(f"outputs: {outputs}")
            print(f"logits: {logits}")


    def bert_attention_mask(self):
        print("\n=== 5. BERT的注意力掩码 ===")
        text1 = "首都北京是一个现代化大都市"
        text2 = "首都北京特别美"
        texts = [text1,text2]
        encode = self.bert_tokenizer(texts,padding=True,truncation=True,return_tensors="pt")
        print(f"输入Id: {encode['input_ids'].shape}")
        print(f"注意力掩码: {encode['attention_mask']}")

    def bert_forward(self):
        print("\n=== 6. BERT前向传播演示 ===")
        text = "中国的首都北京是一个现代化大都市"
        encode = self.bert_tokenizer(text,return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**encode)
        print(f"last_hidden_state: {outputs.last_hidden_state.shape}")
        print(f"pooler_output: {outputs.pooler_output.shape}")

        cls_embedding = outputs.last_hidden_state[0,0,:]
        print(f"cls_embedding: {cls_embedding.shape}")

    def gpt_unidirectional(self):
        print("\n=== 7. GPT自回归语言建模演示 ===")
        text = "This evening eat"
        print(f"输入文本: {text}")

        inputs = self.gpt_tokenizer(text,return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gpt_model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            print(f"next_token_logits: {next_token_logits.shape}")

        top_k = 5
        top_tokens = torch.topk(next_token_logits,top_k,dim=-1)

        for i in range(top_k):
            token_id = top_tokens.indices[0,i].item()
            token = self.gpt_tokenizer.decode(token_id)
            prob = torch.softmax(next_token_logits,dim=-1)[0,token_id].item()
            print(f"预测下一个token: {token} (概率: {prob:.4f})")

    def gpt_fine_tune_generator(self):
        print("\n=== 8. GPT微调演示 ===")
        prompts = self.create_generation_dataset()

        optimizer = AdamW(self.gpt_model.parameters(),lr=5e-5)
        num_epochs = 5

        for epoch in range(num_epochs):
            total_loss = 0
            for prompt in prompts:
                inputs = self.gpt_tokenizer(prompt,padding=True,truncation=True,return_tensors="pt").to(self.device)

                labels = inputs['input_ids'].clone()
                outputs = self.gpt_model(**inputs,labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss +=loss.item()

            avg_loss = total_loss / len(prompts)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("微调完成")

        test_prompt = "Today is the weekend. What should I do, "
        test_inputs = self.gpt_tokenizer(test_prompt,return_tensors="pt").to(self.device)

        with torch.no_grad():
            test_outputs = self.gpt_model.generate(
                test_inputs['input_ids'],
                max_length=50,
                temperature = 0.8,
                do_sample=True,
                pad_token_id=self.gpt_tokenizer.eos_token_id
            )

        generated_text = self.gpt_tokenizer.decode(test_outputs[0],skip_special_tokens=True)
        print(f"生成文本: {generated_text}")


    def create_generation_dataset(self):
        """创建文本生成数据集"""
        # 模拟故事续写数据集
        prompts = [
            "Once upon a time, in a magical forest,",
            "The scientist discovered a new element that",
            "In the year 2050, artificial intelligence",
            "The young adventurer found an ancient map leading to"
        ]
        return prompts

    def run_all_demos(self):
        print("🚀 开始作业演示\n")

        self.load_models()
        self.compare_architectures()
        self.compare_pretraining_tasks()
        self.bert_bidirectional()
        self.bert_demonstrate_tokenization()
        self.bert_attention_mask()
        self.bert_forward()
        self.gpt_unidirectional()
        self.gpt_fine_tune_generator()

        print("\n🎉 作业完成！")

# 主程序
if __name__ == "__main__":
    demo = BERTGPTDEMO()
    demo.run_all_demos()
