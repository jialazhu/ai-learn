import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class BERTTeachingDemo:
    def __init__(self):
        """初始化BERT教学演示类"""
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def load_bert_model(self):
        """1. 加载预训练BERT模型"""
        print("=== 1. 加载BERT模型 ===")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

            # 加载BERT模型
            self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)

            # 加载MLM任务模型
            self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-chinese').to(self.device)

            print("✓ BERT模型加载完成")
            print(f"模型参数量: {sum(p.numel() for p in self.bert_model.parameters()):,}")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("💡 请检查网络连接或下载模型文件到本地")
            print("   离线模式: 设置环境变量 HF_HUB_OFFLINE=1")
            print("   或手动下载模型到: ~/.cache/huggingface/hub/")
            return False

    def demonstrate_tokenization(self):
        """2. 演示BERT分词"""
        print("\n=== 2. BERT分词演示 ===")

        text = "自然语言处理技术正在快速发展"

        # 基本分词
        tokens = self.tokenizer.tokenize(text)
        print(f"原始文本: {text}")
        print(f"分词结果: {tokens}")

        # 转换为ID
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        print(f"输入ID: {input_ids}")
        print(f"特殊token: [CLS]={self.tokenizer.cls_token_id}, [SEP]={self.tokenizer.sep_token_id}")

    def demonstrate_attention_mask(self):
        """3. 演示注意力掩码"""
        print("\n=== 3. 注意力掩码演示 ===")

        texts = ["今天天气很好", "自然语言处理很有趣"]
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        print("批量编码结果:")
        print(f"输入ID形状: {encoded['input_ids'].shape}")
        print(f"注意力掩码: {encoded['attention_mask']}")

    def demonstrate_bert_forward(self):
        """4. 演示BERT前向传播"""
        print("\n=== 4. BERT前向传播演示 ===")

        text = "BERT模型能够学习丰富的语言表征"
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        print(f"输入文本: {text}")
        print(f"last_hidden_state形状: {outputs.last_hidden_state.shape}")
        print(f"pooler_output形状: {outputs.pooler_output.shape}")

        # [CLS] token的表征
        cls_embedding = outputs.last_hidden_state[0, 0, :]
        print(f"[CLS] token表征维度: {cls_embedding.shape}")

    def demonstrate_mlm_task(self):
        """5. 演示掩码语言模型任务"""
        print("\n=== 5. 掩码语言模型(MLM)演示 ===")

        text = "自然语言处理是人工智能的重要研究方向"
        print(f"原始文本: {text}")

        # 使用tokenizer进行分词，然后创建掩码
        tokens = self.tokenizer.tokenize(text)
        print(f"分词结果: {tokens}")
        
        # 创建掩码输入
        masked_tokens = tokens.copy()
        mask_positions = []

        # 随机掩码15%的token（跳过特殊token）
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]'] and np.random.random() < 0.15:
                masked_tokens[i] = "[MASK]"
                mask_positions.append(i)

        # 重新编码为完整句子
        masked_sentence = self.tokenizer.convert_tokens_to_string(masked_tokens)
        print(f"掩码后: {masked_sentence}")
        print(f"掩码位置: {mask_positions}")

        # 模型预测
        inputs = self.tokenizer(masked_sentence, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.mlm_model(**inputs)
            predictions = outputs.logits

        # 获取预测结果
        mask_token_index = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]

        if len(mask_token_index) > 0:
            for i, pos in enumerate(mask_token_index):
                predicted_token_id = predictions[0, pos].argmax().item()
                predicted_token = self.tokenizer.decode(predicted_token_id)
                print(f"位置{pos.item()}: [MASK] -> {predicted_token}")
        else:
            print("本次运行没有生成掩码token，这是正常的随机行为")

    def create_classification_dataset(self):
        """创建文本分类数据集"""
        # 模拟新闻分类数据集
        texts = [
            "科技公司发布新款智能手机",  # 科技
            "政府出台新的经济政策",      # 政治
            "足球比赛精彩纷呈",          # 体育
            "电影票房创下新高",          # 娱乐
            "股市出现大幅波动",          # 财经
        ]
        labels = [0, 1, 2, 3, 4]  # 对应的标签

        return texts, labels

    def fine_tune_bert_classifier(self):
        """6. BERT微调实战：文本分类（含验证环节）"""
        print("\n=== 6. BERT微调实战：新闻分类 ===")

        # 准备训练和验证数据
        train_texts, train_labels = self.create_classification_dataset()
        val_texts, val_labels = self.create_validation_dataset()  # 新增验证集

        train_labels = torch.tensor(train_labels).to(self.device)
        val_labels = torch.tensor(val_labels).to(self.device)

        # 加载分类模型
        classifier = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=5
        ).to(self.device)

        # 编码数据
        train_inputs = self.tokenizer(train_texts, padding=True, truncation=True,
                                      return_tensors='pt').to(self.device)
        val_inputs = self.tokenizer(val_texts, padding=True, truncation=True,
                                    return_tensors='pt').to(self.device)  # 编码验证数据

        # 训练参数
        optimizer = AdamW(classifier.parameters(), lr=2e-5)
        num_epochs = 5
        best_val_acc = 0.0  # 记录最佳验证准确率

        print("开始微调训练...")

        for epoch in range(num_epochs):
            # 训练阶段
            classifier.train()
            optimizer.zero_grad()

            # 前向传播
            train_outputs = classifier(**train_inputs, labels=train_labels)
            train_loss = train_outputs.loss
            train_logits = train_outputs.logits

            # 反向传播
            train_loss.backward()
            optimizer.step()

            # 计算训练集准确率
            train_predictions = torch.argmax(train_logits, dim=1)
            train_accuracy = (train_predictions == train_labels).float().mean()

            # 验证阶段
            classifier.eval()  # 切换到评估模式
            with torch.no_grad():  # 关闭梯度计算
                val_outputs = classifier(**val_inputs, labels=val_labels)
                val_loss = val_outputs.loss
                val_logits = val_outputs.logits

                # 计算验证集指标
                val_predictions = torch.argmax(val_logits, dim=1)
                val_accuracy = (val_predictions == val_labels).float().mean()

                # 计算F1分数（需要转换为numpy数组）
                val_f1 = f1_score(
                    val_labels.cpu().numpy(),
                    val_predictions.cpu().numpy(),
                    average='weighted'
                )

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  训练: Loss: {train_loss.item():.3f}, Accuracy: {train_accuracy.item():.3f}")
            print(f"  验证: Loss: {val_loss.item():.3f}, Accuracy: {val_accuracy.item():.3f}, F1: {val_f1:.3f}")

            # 保存最佳模型
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(classifier.state_dict(), "best_bert_classifier.pt")
                print(f"  保存最佳模型 (验证准确率: {best_val_acc.item():.3f})")

        print("✓ 微调完成")

        #将微调完成后的模型.进行验证
        # 最终评估展示
        self.evaluate_final_model(classifier, val_inputs, val_labels, val_texts)

    def create_validation_dataset(self):
        """创建验证数据集"""
        # 与训练集不同的验证样本
        val_texts = [
            "人工智能技术取得重大突破",  # 科技
            "议会通过新的环保法案",  # 政治
            "篮球锦标赛冠军诞生",  # 体育
            "新上映电影获得观众好评",  # 娱乐
            "央行调整金融政策应对通胀",  # 财经
        ]
        val_labels = [0, 1, 2, 3, 4]  # 对应的标签
        return val_texts, val_labels

    def evaluate_final_model(self, model, val_inputs, val_labels, val_texts):
        """最终模型评估与预测演示"""
        print("\n=== 最终模型评估 ===")

        # 加载最佳模型
        model.load_state_dict(torch.load("best_bert_classifier.pt"))
        model.eval()

        # 在验证集上进行预测
        with torch.no_grad():
            outputs = model(**val_inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

        # 标签映射
        label_map = {0: "科技", 1: "政治", 2: "体育", 3: "娱乐", 4: "财经"}

        # 展示预测结果
        print("\n预测结果示例:")
        for text, true_label, pred_label in zip(val_texts, val_labels, predictions):
            true_label_name = label_map[true_label.item()]
            pred_label_name = label_map[pred_label.item()]
            status = "✓" if true_label == pred_label else "✗"
            print(f"{status} 文本: {text}")
            print(f"   真实标签: {true_label_name}, 预测标签: {pred_label_name}\n")

        # 计算总体指标
        accuracy = accuracy_score(val_labels.cpu().numpy(), predictions.cpu().numpy())
        f1 = f1_score(val_labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
        print(f"最终验证集性能:")
        print(f"准确率: {accuracy:.3f}")
        print(f"F1分数: {f1:.3f}")

    def compare_bert_vs_gpt(self):
        """7. BERT与GPT对比分析"""
        print("\n=== 7. BERT vs GPT 对比分析 ===")

        comparison = {
            "架构": {
                "BERT": "双向Transformer编码器",
                "GPT": "单向Transformer解码器"
            },
            "注意力机制": {
                "BERT": "全词关注（无掩码）",
                "GPT": "掩码自注意力"
            },
            "优势场景": {
                "BERT": "文本理解任务",
                "GPT": "文本生成任务"
            },
            "预训练任务": {
                "BERT": "MLM + NSP",
                "GPT": "自回归语言建模"
            },
            "参数量级": {
                "BERT": "Base版1.1亿参数",
                "GPT": "GPT-3达1750亿参数"
            }
        }

        print("BERT与GPT核心对比:")
        for aspect, models in comparison.items():
            print(f"{aspect}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")
            print()

    def run_all_demos(self):
        """运行所有BERT教学演示"""
        print("🚀 开始BERT教学演示\n")

        if not self.load_bert_model():
            print("\n⚠️  由于模型加载失败，跳过需要模型的演示")
            print("📖 您可以查看代码注释了解BERT的工作原理")
            return

        self.demonstrate_tokenization()
        self.demonstrate_attention_mask()
        self.demonstrate_bert_forward()
        self.demonstrate_mlm_task()
        self.fine_tune_bert_classifier()
        self.compare_bert_vs_gpt()

        print("\n🎉 BERT教学演示完成！")

# 主程序
if __name__ == "__main__":
    demo = BERTTeachingDemo()
    demo.run_all_demos()
