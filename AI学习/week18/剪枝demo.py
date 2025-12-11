import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import time
import os
import numpy as np
from collections import OrderedDict

from fontTools.misc.plistlib import end_data

from AI学习.week5.rnn_text import criterion


def get_device():
    if torch.mps.is_available():
        device = torch.device("mps")
        device_info = f"mps: {torch.mps.get_rng_state()}"
        print(f"使用设备: {device_info}")
        return device
    else:
        return torch.device("cpu")

#所有的pytorch神经网络模型必须继承nn.Module

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层
        self.conv1 = nn.Conv2d(1, 16, 3,padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=1)
        #全联接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #卷积 + ReLU + 池化

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        #展平
        x = x.view(x.size(0), -1)
        #全联接 + ReLU
        x = F.relu(self.fc1(x))
        #输出层
        x = self.fc2(x)
        return x


class simleNet(nn.Module):
    def __int__(self):
        super().__int__()

        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#工具函数
def count_parameters(model): #用于计算模型参数总数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_nonzero_parameters(model): #用于计算模型中非零参数总数
    total_nonzero = 0
    for module in model.modules():
        if isinstance(module,(nn.Linear,nn.Conv2d)):
            if hasattr(module,'weight_mask'):
                total_nonzero += (module.weight_mask != 0).sum().item()
            else:
                total_nonzero += (module.weight != 0).sum().item()
    return total_nonzero


def get_model_size(model,filepath='temp_model.pth',use_compression = False):
    if use_compression:
        import gzip
        import pickle
        with gzip.open(filepath +'.gz','wb') as f:
            pickle.dump(model.state_dict(),f)
        size = os.path.getsize(filepath +'.gz') / (1024 ** 2)
        os.remove(filepath +'.gz')
    else:
        torch.save(model.state_dict(),filepath)
        size = os.path.getsize(filepath) / (1024 ** 2)
        os.remove(filepath+'.gz')
    return size

def get_theroetical_compressed_size(model):
    total_size = 0
    for param in model.parameters():
        nonzero_count = (param != 0).sum().item()
        total_size += nonzero_count * param.numel() * 4 / (1024 ** 2)
    return total_size

def evaluate_model(model,test_loader,device):
    model.eval() #评估模式 自动禁用dropout
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 评估模型推理速度
def benchmark_inference_speed(model,test_loader,device,num_rens=100):
    model.eval()
    data_iter = iter(test_loader)
    images,_ = next(data_iter)
    images = images.to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(images)

    torch.mps.synchronize() if device.type == 'mps' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(10):
            _ = model(images)

    torch.mps.synchronize() if device.type == 'mps' else None
    end_time = time.time()

    avg_time = (end_time - start_time) / num_rens * 1000

    print(f"推理速度: {avg_time:.4f} ms")
    return avg_time

def train_model(model,train_loader,device,epochs = 3):
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数 适合多分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    print(f"开始训练模型, 设备: {device}, 轮数: {epochs}")
    for epoch in range(epochs):
        running_loss = 0.0
        for i , (images,labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"轮次 [{epoch+1}/{epochs}], 批次 [{i+1}/{len(train_loader)}], 损失: {running_loss/100:.4f}")
                running_loss = 0.0
    print("训练完成")



if __name__ == '__main__':
    device = get_device()
