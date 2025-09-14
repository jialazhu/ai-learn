import torch
import time

print("PyTorch 版本:", torch.__version__)
print("是否有 CUDA:", torch.cuda.is_available())   # macOS 永远 False
print("是否有 MPS:", torch.backends.mps.is_available())

device = torch.device('cpu')
x = torch.rand((10000, 10000), dtype=torch.float32)
y = torch.rand((10000, 10000), dtype=torch.float32)
x = x.to(device)
y = y.to(device)

start = time.time()
z = x @ y
end = time.time()
print("CPU 时间:", end - start)
print("CPU:", z)
device = torch.device('mps')
x = torch.rand((10000, 10000), dtype=torch.float32)
y = torch.rand((10000, 10000), dtype=torch.float32)
x = x.to(device)
y = y.to(device)
start = time.time()
z = x @ y
end = time.time()
print("mps 时间:", end - start)
print("mps:", z)
