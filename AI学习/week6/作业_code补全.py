import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):  # 继承自nn.Module
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义四个线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 判断完掩码之后，分配注意力权重，返回注意力和注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, x):  # 形状(batch_size, seq_len, d_model)
        # 定义分割头的方法，然后将输入分数分割成多个头
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 合并头
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)


    def forward(self, Q, K, V, mask=None):
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(output)
        return self.W_o(output)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    # 进行初始化，实现feed Forwardde 部分
    #todo
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数索引使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数索引使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加批次维度
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x 的形状: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 多头注意力模块
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈网络模块

        # 两个子层的层归一化 (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 两个子层的 Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 1. 多头自注意力子层
        #    Q, K, V 都来自 src
        attn_output = self.self_attn(src, src, src, src_mask)

        # 2. 残差连接 和 层归一化 (Add & Norm)
        #    src + dropout(attention_output)
        src = self.norm1(src + self.dropout1(attn_output))

        # 3. 前馈网络子层
        ff_output = self.feed_forward(src)

        # 4. 第二个残差连接 和 层归一化 (Add & Norm)
        src = self.norm2(src + self.dropout2(ff_output))

        return src


if __name__ == '__main__':
    # 定义模型超参数
    d_model = 512  # 模型的总维度
    num_heads = 8  # 多头注意力的头数
    d_ff = 2048  # 前馈网络中间层的维度
    dropout = 0.1

    # 准备输入数据
    sentence = "A transformer is a deep learning model used primarily in NLP"
    tokens = sentence.split()
    vocab = {word: i for i, word in enumerate(tokens, 1)}  # +1 for padding
    vocab['<pad>'] = 0

    seq_len = len(tokens)
    batch_size = 1

    # 将 token 转换为索引
    token_ids = torch.tensor([vocab[word] for word in tokens]).unsqueeze(0)  # (1, seq_len)

    print("--- 数据流演示 ---")
    print(f"1. 原始输入句子: '{sentence}'")
    print(f"   Token ID 形状: {token_ids.shape}")

    # a. 词嵌入
    embedding = nn.Embedding(len(vocab), d_model)
    embedded_input = embedding(token_ids)
    print(f"2. 经过词嵌入层后，形状变为: {embedded_input.shape} (Batch, SeqLen, d_model)")

    # b. 添加位置编码
    # PyTorch 的 Transformer 模块通常期望 (SeqLen, Batch, Dim) 的形状
    embedded_input = embedded_input.transpose(0, 1)
    pos_encoder = PositionalEncoding(d_model, dropout)
    final_input = pos_encoder(embedded_input)
    print(f"3. 添加位置编码后 (并转置)，形状变为: {final_input.shape} (SeqLen, Batch, d_model)")

    # 初始化 Transformer Encoder Layer
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)

    # c. 通过 Encoder Layer
    output = encoder_layer(final_input)

    print(f"4. 经过一个 Encoder Layer 后，输出形状: {output.shape} (保持不变)")
    print("-" * 20)

    print("\n--- 结论 ---")
    print("输入张量成功流经了一个完整的 Transformer Encoder 层。")
    print(" - 它首先通过多头自注意力机制，让每个词的表示都融合了句子中所有其他词的信息。")
    print(" - 然后通过前馈网络进行非线性变换，进一步增强了模型的表达能力。")
    print(" - 每个子层都使用了残差连接和层归一化，以保证训练的稳定性和效率。")
    print("输出的张量是输入句子新的、富含上下文信息的表示。在实际模型中，这样的层会堆叠N次。")
