import torch
import torch.nn as nn
import torch.optim as optim  # 优化器 对超参数进行自动优化
import numpy as np  # 修正拼写错误
import random

# 数据准备
corpus = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""".lower()

# 对语料库处理
chars = sorted(list(set(corpus)))
char_to_index = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)  # 修正拼写错误

seq_length = 40  # 修正拼写错误

input_seqs = []
target_seqs = []

for i in range(len(corpus) - seq_length):
    input_seqs.append([char_to_index[c] for c in corpus[i: i + seq_length]])
    target_seqs.append([char_to_index[c] for c in corpus[i + 1: i + seq_length + 1]])


# 定义模型
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):  # 修正拼写错误
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 表示内部记忆单元.逻辑按照批次大小.序列长度,特征维度
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size=1):  # 在初始化的时候没有资格赋值.所以把初始值初始化为0
        return torch.zeros(1, batch_size, self.hidden_size)


# 训练模型
embedding_dim = 16  # 修正拼写错误
hidden_size = 64
learning_rate = 0.005
epochs = 500

model = CharRNN(vocab_size, embedding_dim, hidden_size)  # 修正拼写错误
# 定义损失函数 交叉熵误差
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    seq_idx = random.randint(0, len(input_seqs) - 1)
    input_tensor = torch.tensor(input_seqs[seq_idx]).unsqueeze(0)
    target_tensor = torch.tensor(target_seqs[seq_idx])

    hidden = model.init_hidden()
    optimizer.zero_grad()  # 计算梯度时.清楚上一轮旧梯度
    output, hidden = model(input_tensor, hidden)
    loss = criterion(output.squeeze(0), target_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# 测试.输入结果 生成文本(温度采样)
def generate_text(model, start_char, length, temperature=0.8):  # 修正拼写错误
    # 确保起始字符在词汇表中
    if start_char not in char_to_index:
        raise ValueError(f"起始字符 '{start_char}' 不在词汇表中")

    # 将模型切换到评估模式
    model.eval()
    with torch.no_grad():  # 梯度在函数当中计算.不需要用pytorch再计算
        result = start_char
        input_char = torch.tensor([char_to_index[start_char]]).unsqueeze(0)
        hidden = model.init_hidden()

        for _ in range(length):
            output, hidden = model(input_char, hidden)  # hidden 值更新
            output_dist = output.squeeze().div(temperature).exp()  # 修正拼写错误
            top_i = torch.multinomial(output_dist, 1)[0]  # 修正拼写错误

            predicted_char = idx_to_char[top_i.item()]  # 修正拼写错误
            result += predicted_char

            input_char = torch.tensor([top_i.item()]).unsqueeze(0)

        return result


# 评估和展示
# 尝试不同的温度来观察生成效果
print("\n--- 生成文本 (温度: 0.5 - 比较保守) ---")
print(generate_text(model, 't', 200, temperature=0.5))

print("\n--- 生成文本 (温度: 1.0 - 更有创意) ---")
print(generate_text(model, 't', 200, temperature=1.0))

print("\n--- 生成文本 (温度: 1.5 - 可能开始胡言乱语) ---")
print(generate_text(model, 't', 200, temperature=1.5))
