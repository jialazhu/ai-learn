
import torch
import torch.nn as nn
import time
import os
import numpy as np
from typing import Tuple, Dict, Any

# 尝试导入ONNX相关库
ONNX_AVAILABLE = False
import_error_msg = ""

try:
    import onnxruntime as ort
    import onnx
    ONNX_AVAILABLE = True
    print("ONNX和ONNX Runtime导入成功！")
except ImportError as e:
    import_error_msg = str(e)
    ONNX_AVAILABLE = False
    print(f"ONNX导入失败: {e}")
    print("警告: ONNX或ONNX Runtime未安装，部分功能将不可用")
    print("\n推荐的解决方案:")
    print("1. pip install onnx==1.15.0 onnxruntime==1.16.3")
    print("2. 或者: pip install onnxruntime-cpp")
    print("3. 安装Visual C++ Redistributable后重启")
except Exception as e:
    import_error_msg = str(e)
    ONNX_AVAILABLE = False
    print(f"ONNX Runtime DLL加载失败: {e}")
    print("\n这通常是Windows系统上的DLL加载问题，请尝试以下解决方案:")
    print("1. 重新安装: pip uninstall onnxruntime -y && pip install onnxruntime==1.16.3")
    print("2. 使用CPU版本: pip install onnxruntime-cpp")
    print("3. 安装Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("4. 重启计算机后再试")


class InferenceCNN(nn.Module):
    """
    CNN模型类用于推理任务
    输入：3通道图像，尺寸为32x32
    输出：10个类别的分类结果
    """
    def __init__(self):
        super(InferenceCNN, self).__init__()
        # 第一个卷积层：3->16通道，3x3卷积核，padding=1保持尺寸
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # 第二个卷积层：16->32通道，3x3卷积核，padding=1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # 自适应平均池化，将特征图调整为4x4
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # 全连接层：32*4*4 -> 10
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # 前向传播
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


def get_model_size(model: torch.nn.Module, model_path: str = "temp_model.pth") -> float:
    """
    计算模型大小（MB）

    Args:
        model: PyTorch模型
        model_path: 临时保存路径

    Returns:
        模型大小（MB）
    """
    torch.save(model.state_dict(), model_path)
    size = os.path.getsize(model_path) / (1024 * 1024)
    os.remove(model_path)
    return size


def export_model_to_onnx(model: torch.nn.Module, onnx_path: str = "inference_model.onnx") -> bool:
    """
    将PyTorch模型导出为ONNX格式

    Args:
        model: PyTorch模型
        onnx_path: ONNX模型保存路径

    Returns:
        是否导出成功
    """
    try:
        model.eval()

        # 创建示例输入（batch_size=1）
        dummy_input = torch.randn(1, 3, 32, 32)

        # 导出ONNX模型
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # 使用较新的opset版本
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},  # 支持动态batch_size
                'output': {0: 'batch_size'}
            }
        )

        print(f"模型已成功导出到: {onnx_path}")

        # 验证ONNX模型
        if ONNX_AVAILABLE:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证通过！")

            # 打印模型信息
            print(f"ONNX模型大小: {os.path.getsize(onnx_path) / (1024 * 1024):.4f} MB")

        return True

    except Exception as e:
        print(f"导出ONNX模型失败: {str(e)}")
        return False


def pytorch_inference(model: torch.nn.Module, input_data: torch.Tensor, num_runs: int = 100) -> Tuple[torch.Tensor, float]:
    """
    PyTorch原生推理

    Args:
        model: PyTorch模型
        input_data: 输入数据
        num_runs: 运行次数

    Returns:
        (模型输出, 平均推理时间ms)
    """
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    # 正式推理测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_data)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒

    return output, avg_time


def onnx_inference(onnx_path: str, input_data: torch.Tensor, num_runs: int = 100) -> Tuple[np.ndarray, float]:
    """
    ONNX Runtime推理

    Args:
        onnx_path: ONNX模型路径
        input_data: 输入数据
        num_runs: 运行次数

    Returns:
        (模型输出, 平均推理时间ms)
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime未安装，请运行: pip install onnxruntime")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")

    # 创建推理会话
    session = ort.InferenceSession(onnx_path)

    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 转换输入数据为numpy格式
    input_np = input_data.numpy().astype(np.float32)

    # 预热
    for _ in range(10):
        _ = session.run([output_name], {input_name: input_np})

    # 正式推理测试
    start_time = time.time()
    for _ in range(num_runs):
        outputs = session.run([output_name], {input_name: input_np})
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒

    return outputs[0], avg_time


def compare_inference_performance(model: torch.nn.Module, onnx_path: str, test_input: torch.Tensor):
    """
    对比PyTorch和ONNX Runtime的推理性能

    Args:
        model: PyTorch模型
        onnx_path: ONNX模型路径
        test_input: 测试输入
    """
    print("=" * 60)
    print("推理性能对比报告")
    print("=" * 60)

    # PyTorch推理
    pytorch_output, pytorch_time = pytorch_inference(model, test_input)

    # ONNX Runtime推理
    try:
        onnx_output, onnx_time = onnx_inference(onnx_path, test_input)

        # 模型大小对比
        pytorch_size = get_model_size(model)
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)

        # 输出一致性验证
        pytorch_np = pytorch_output.numpy()
        mse = np.mean((pytorch_np - onnx_output) ** 2)
        max_abs_error = np.max(np.abs(pytorch_np - onnx_output))

        # 打印对比结果
        print(f"\n1. 推理时间对比:")
        print(f"   PyTorch推理时间:     {pytorch_time:.4f} ms")
        print(f"   ONNX Runtime推理时间: {onnx_time:.4f} ms")
        if onnx_time > 0:
            speedup = pytorch_time / onnx_time
            print(f"   加速比:              {speedup:.2f}x")

        print(f"\n2. 模型大小对比:")
        print(f"   PyTorch模型大小:     {pytorch_size:.4f} MB")
        print(f"   ONNX模型大小:        {onnx_size:.4f} MB")
        if onnx_size > 0:
            size_ratio = pytorch_size / onnx_size
            print(f"   大小比例:            {size_ratio:.2f}x")

        print(f"\n3. 输出一致性验证:")
        print(f"   MSE误差:             {mse:.8f}")
        print(f"   最大绝对误差:         {max_abs_error:.8f}")

        # 判断一致性
        if mse < 1e-5:
            print("   [OK] 输出结果高度一致")
        elif mse < 1e-3:
            print("   [WARN] 输出结果基本一致")
        else:
            print("   [ERROR] 输出结果差异较大")

    except Exception as e:
        print(f"ONNX推理失败: {str(e)}")
        print(f"仅PyTorch推理时间: {pytorch_time:.4f} ms")


def compare_batch_inference(model: torch.nn.Module, onnx_path: str):
    """
    对比不同batch size下的推理性能

    Args:
        model: PyTorch模型
        onnx_path: ONNX模型路径
    """
    print("\n" + "=" * 60)
    print("批量推理性能对比")
    print("=" * 60)

    batch_sizes = [1, 4, 8, 16]
    results = []

    for batch_size in batch_sizes:
        print(f"\n测试 Batch Size: {batch_size}")
        print("-" * 40)

        # 创建对应batch size的输入
        test_input = torch.randn(batch_size, 3, 32, 32)

        # PyTorch推理
        try:
            pytorch_output, pytorch_time = pytorch_inference(model, test_input, num_runs=50)
            pytorch_throughput = batch_size / pytorch_time * 1000  # samples/sec
        except Exception as e:
            print(f"PyTorch推理失败: {str(e)}")
            pytorch_time = float('inf')
            pytorch_throughput = 0

        # ONNX Runtime推理
        try:
            onnx_output, onnx_time = onnx_inference(onnx_path, test_input, num_runs=50)
            onnx_throughput = batch_size / onnx_time * 1000  # samples/sec

            # 计算一致性
            mse = np.mean((pytorch_output.numpy() - onnx_output) ** 2)
            consistency = "[OK]" if mse < 1e-5 else "[WARN]" if mse < 1e-3 else "[ERROR]"

        except Exception as e:
            print(f"ONNX推理失败: {str(e)}")
            onnx_time = float('inf')
            onnx_throughput = 0
            consistency = "[ERROR]"

        # 记录结果
        result = {
            'batch_size': batch_size,
            'pytorch_time': pytorch_time,
            'onnx_time': onnx_time,
            'pytorch_throughput': pytorch_throughput,
            'onnx_throughput': onnx_throughput,
            'consistency': consistency
        }
        results.append(result)

        # 打印结果
        print(f"PyTorch:   {pytorch_time:.4f}ms ({pytorch_throughput:.1f} samples/sec)")
        print(f"ONNX:      {onnx_time:.4f}ms ({onnx_throughput:.1f} samples/sec)")
        print(f"一致性:     {consistency}")

    # 打印汇总表格
    print(f"\n{'Batch Size':<10} {'PyTorch(ms)':<12} {'ONNX(ms)':<10} {'PyTorch(样本/s)':<15} {'ONNX(样本/s)':<13} {'一致性':<6}")
    print("-" * 70)
    for result in results:
        print(f"{result['batch_size']:<10} {result['pytorch_time']:<12.4f} {result['onnx_time']:<10.4f} "
              f"{result['pytorch_throughput']:<15.1f} {result['onnx_throughput']:<13.1f} {result['consistency']:<6}")

    # 分析性能趋势
    print(f"\n性能分析:")
    if results:
        best_pytorch_batch = min(results, key=lambda x: x['pytorch_time'] if x['pytorch_time'] != float('inf') else float('inf'))
        best_onnx_batch = min(results, key=lambda x: x['onnx_time'] if x['onnx_time'] != float('inf') else float('inf'))

        if best_pytorch_batch['pytorch_time'] != float('inf'):
            print(f"PyTorch最佳batch size: {best_pytorch_batch['batch_size']} "
                  f"({best_pytorch_batch['pytorch_time']:.4f}ms)")
        if best_onnx_batch['onnx_time'] != float('inf'):
            print(f"ONNX Runtime最佳batch size: {best_onnx_batch['batch_size']} "
                  f"({best_onnx_batch['onnx_time']:.4f}ms)")


def main():
    """主函数：执行所有任务"""
    print("=" * 70)
    print("AI模型优化与部署 - Week18作业")
    print("CNN模型PyTorch与ONNX Runtime推理性能对比")
    print("=" * 70)

    # 任务1：创建CNN模型
    print("\n任务1: 创建CNN模型")
    model = InferenceCNN()
    print(f"模型结构: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    # 任务2：导出ONNX模型
    print("\n任务2: 导出ONNX模型")
    onnx_path = "inference_cnn_model.onnx"
    export_success = export_model_to_onnx(model, onnx_path)

    if not export_success:
        print("ONNX导出失败，跳过相关对比测试")
        return

    # 准备测试数据
    test_input = torch.randn(1, 3, 32, 32)

    # 任务3-5：推理性能对比
    print("\n任务3-5: 推理性能对比")
    compare_inference_performance(model, onnx_path, test_input)

    # 任务6：批量推理对比
    print("\n任务6: 批量推理对比")
    compare_batch_inference(model, onnx_path)

    print("\n" + "=" * 70)
    print("作业完成！")
    print("=" * 70)

    # 清理临时文件
    try:
        if os.path.exists(onnx_path):
            print(f"\n是否删除ONNX模型文件 {onnx_path}? (y/n)")
            # 注释掉交互式删除，自动清理
            choice = input().strip().lower()
            if choice == 'y' or choice == '是' or choice == 'yes':
                os.remove(onnx_path)
                print("ONNX模型文件已删除")
    except:
        pass


if __name__ == "__main__":
    main()
