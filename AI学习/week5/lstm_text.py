import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

train_data = [
    ("this movie is great", 1),
    ("i love this film", 1),
    ("what a fantastic show", 1),
    ("the plot is boring", 0),
    ("i did not like the acting", 0),
    ("it was a waste of time", 0),
]

word_to_idx = {"PAD" : 0}
for sentence, _ in train_data:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
vocab_size = len(word_to_idx)
idx_to_word = { i : x for x ,i in word_to_idx.items()}
#将句子转换为索引序列
sequences = [ torch.tensor([word_to_idx[w] for w in s.split()]) for s, _ in train_data ]
labels = torch.tensor([label for _,label in train_data],dtype=torch.float32)

#填充序列 是他们长度一致
#传统的神经网络要求大小固定 因此我们需要将不同长度的句子填充到相同长度
#这对应了ppt中提到的传统模型处理序列数据的挑战之一
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=word_to_idx["PAD"])

#定义LSTM
#相比基础RNN LSTM通过门控机制. 解决梯度消失问题
#从而能更好的捕捉句子中的长距离依赖关系

class LSTMSentimentCLassifier(nn.Module):

    def __init__(self,vocab_size,embedding_dim,hidden_dim,outpot_dim):
        super(LSTMSentimentCLassifier,self).__init__()

        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        #LSTM
        # 接受词向量序列作为输入 并输出隐藏状态
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,batch_first=True)

        self.fc = nn.Linear(hidden_dim,outpot_dim)

    def forward(self,text):
        embedded = self.embedding(text)
        #LSTM输出包括所有时间步的输出和最后一个时间步的隐藏状态(h_n)和细胞状态(c_n)
        #这里只需要最后一个隐藏状态 h_n 来代表整个句子
        lstm_out, (hidden,cell) = self.lstm(embedded)

        final_hidden_state = hidden.squeeze(0)

        output = self.fc(final_hidden_state)

        return torch.sigmoid(output)

embedding_dim = 10
hidden_dim = 32
output_dim = 1
learing_rate = 0.1
epochs = 200

model = LSTMSentimentCLassifier(vocab_size,embedding_dim,hidden_dim,output_dim)
optimizer = optim.Adam(model.parameters(),lr=learing_rate)
criterion = nn.BCELoss() #二元交叉熵损失 适用于二分类问题

print("开始训练LSTM情感分类模型")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    #前向传播
    predictions = model(padded_sequences).squeeze(1)

    #计算损失
    loss = criterion(predictions,labels)

    loss.backward()
    optimizer.step()

    if(epoch +1 ) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("训练完成")

def predict_sentiment(model,sentence):
    model.eval()
    with torch.no_grad():
        #将句子转换为索引序列
        words = sentence.split()
        indexed = [ word_to_idx.get(w,0) for w in words ]
        tensor = torch.LongTensor(indexed).unsqueeze(0)
        #预测
        prediction = model(tensor)

        return "正面" if prediction.item() > 0.5 else "负面"


test_sentence = "movie"

print(f"测试句子: {test_sentence}")
print(f"情感预测: {predict_sentiment(model,test_sentence)}")

test_sentence_2 = "the acting was terrible"
print(f"'{test_sentence_2}' 的情感是: {predict_sentiment(model, test_sentence_2)}")
