"""
多模态风控系统测试脚本
用于验证各项功能是否正常工作
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_prediction():
    """测试基础预测功能"""
    logger.info("测试基础预测功能...")

    try:
        from explain import load_model, explain_prediction, generate_explanation_text

        # 加载模型
        model, scaler, feature_names = load_model()
        logger.info("✓ 模型加载成功")

        # 示例特征数据
        test_features = {
            "Time": 0.0,
            "V1": -1.3598,
            "V2": -0.0727,
            "V3": 2.5363,
            "V4": 1.3782,
            "V5": -0.3383,
            "V6": 0.4624,
            "V7": 0.2396,
            "V8": 0.0987,
            "V9": 0.3637,
            "V10": 0.0908,
            "V11": -0.5516,
            "V12": -0.6178,
            "V13": -0.9914,
            "V14": -0.3112,
            "V15": 1.4682,
            "V16": -0.4704,
            "V17": 0.2079,
            "V18": 0.0258,
            "V19": 0.4039,
            "V20": 0.2514,
            "V21": -0.0183,
            "V22": 0.2778,
            "V23": -0.1105,
            "V24": 0.0669,
            "V25": 0.1285,
            "V26": -0.1891,
            "V27": 0.1336,
            "V28": -0.0211,
            "Amount": 149.62
        }

        # 进行预测
        explanation_data = explain_prediction(model, scaler, feature_names, test_features)
        logger.info(f"✓ 预测完成 - 风险分数: {explanation_data['prediction_score']:.4f}")
        logger.info(f"✓ 预测标签: {explanation_data['prediction_label']}")

        # 生成解释文本
        explanation_text = generate_explanation_text(explanation_data)
        logger.info("✓ SHAP解释生成成功")
        logger.info(f"解释摘要: {explanation_text[:100]}...")

        return True

    except Exception as e:
        logger.error(f"基础预测测试失败: {e}")
        return False


async def test_ollama_client():
    """测试Ollama客户端"""
    logger.info("测试Ollama客户端...")

    try:
        from ollama_client import OllamaClient

        client = OllamaClient()

        # 健康检查
        if client.health_check():
            logger.info("✓ Ollama服务连接正常")
        else:
            logger.error("✗ Ollama服务不可用")
            return False

        # 获取模型列表
        models_info = client.list_available_models()
        all_models = models_info.get("all_models", [])
        multimodal_models = models_info.get("multimodal_models", [])

        logger.info(f"✓ 发现模型: {len(all_models)}个")
        logger.info(f"✓ 多模态模型: {len(multimodal_models)}个")

        if not all_models:
            logger.warning("⚠ 未发现任何模型，建议安装: ollama pull qwen3:4b")
        else:
            logger.info(f"可用模型: {all_models}")

        if not multimodal_models:
            logger.warning("⚠ 未发现多模态模型，图像分析功能将受限")
            logger.info("建议安装: ollama pull qwen3-vl:4b")
        else:
            logger.info(f"多模态模型: {multimodal_models}")

        # 测试文本对话
        if all_models:
            try:
                response = client.chat(
                    model=all_models[0],
                    messages=[{"role": "user", "content": "你好"}]
                )
                logger.info(f"✓ 文本对话测试成功: {response[:50]}...")
            except Exception as e:
                logger.warning(f"⚠ 文本对话测试失败: {e}")

        return True

    except Exception as e:
        logger.error(f"Ollama客户端测试失败: {e}")
        return False


async def test_multimodal_processor():
    """测试多模态处理器"""
    logger.info("测试多模态处理器...")

    try:
        from multimodal_processor import MultimodalProcessor

        processor = MultimodalProcessor()

        # 测试图像格式验证
        valid_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        for fmt in valid_formats:
            if processor.validate_image_format(f"test{fmt}"):
                logger.info(f"✓ 图像格式支持: {fmt}")

        # 测试图像质量分析（使用示例图像）
        test_image_path = "test_sample.jpg"
        if os.path.exists(test_image_path):
            try:
                # 加载并分析图像
                image = processor.load_image_from_path(test_image_path)
                logger.info(f"✓ 图像加载成功: {image.size}")

                # 质量分析
                quality_info = processor.analyze_image_quality(image)
                logger.info(f"✓ 图像质量分析完成: {quality_info.get('quality_grade', '未知')}")

                # 人脸检测
                face_info = processor.detect_faces(image)
                logger.info(f"✓ 人脸检测完成: 发现{face_info.get('faces_detected', 0)}张人脸")

            except Exception as e:
                logger.warning(f"⚠ 图像处理测试失败（可能没有测试图像）: {e}")
        else:
            logger.info("⚠ 跳过图像处理测试（需要test_sample.jpg文件）")

        return True

    except Exception as e:
        logger.error(f"多模态处理器测试失败: {e}")
        return False


async def test_fraud_analyzer():
    """测试风控分析器"""
    logger.info("测试风控分析器...")

    try:
        from ollama_client import OllamaClient
        from fraud_multimodal_analyzer import FraudMultimodalAnalyzer
        from explain import load_model

        # 初始化
        client = OllamaClient()
        if not client.health_check():
            logger.warning("⚠ Ollama服务不可用，跳过分析器测试")
            return True

        model, scaler, feature_names = load_model()
        analyzer = FraudMultimodalAnalyzer(client, model, scaler, feature_names)

        # 测试特征数据
        test_features = {
            "Time": 0.0,
            "Amount": 1000.0,
            "V1": -1.2,
            "V2": 0.5,
            "V3": 1.0,
            "V4": -0.8,
            "V5": 0.3,
            "V6": -0.1,
            "V7": 0.7,
            "V8": -0.4,
            "V9": 0.2,
            "V10": -0.6,
            "V11": 0.9,
            "V12": -0.3,
            "V13": 0.1,
            "V14": -0.7,
            "V15": 0.4,
            "V16": -0.2,
            "V17": 0.6,
            "V18": -0.5,
            "V19": 0.8,
            "V20": -0.1,
            "V21": 0.3,
            "V22": -0.4,
            "V23": 0.2,
            "V24": -0.6,
            "V25": 0.5,
            "V26": -0.3,
            "V27": 0.1,
            "V28": -0.2
        }

        # 测试综合分析（无图像）
        result = await analyzer.comprehensive_analysis(
            transaction_features=test_features,
            images=None
        )
        logger.info("✓ 综合分析测试成功（仅特征）")
        logger.info(f"风险等级: {result.get('overall_risk', {}).get('level', '未知')}")

        return True

    except Exception as e:
        logger.error(f"风控分析器测试失败: {e}")
        return False


async def test_api_endpoints():
    """测试API接口"""
    logger.info("测试API接口...")

    try:
        import requests
        import time

        base_url = "http://localhost:8001"

        # 健康检查
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ 健康检查接口正常")
                health_data = response.json()
                logger.info(f"服务状态: {health_data.get('status', '未知')}")
            else:
                logger.warning(f"⚠ 健康检查接口异常: {response.status_code}")
        except requests.exceptions.RequestException:
            logger.warning("⚠ API服务未启动，跳过接口测试")
            return True

        # 测试基础预测接口
        test_features = {
            "Time": 0.0,
            "V1": -1.3598,
            "V2": -0.0727,
            "Amount": 149.62
        }

        try:
            response = requests.post(
                f"{base_url}/predict",
                json={
                    "features": test_features,
                    "explain": True,
                    "use_llm": False  # 避免依赖LLM
                },
                timeout=10
            )
            if response.status_code == 200:
                logger.info("✓ 基础预测接口正常")
                prediction_data = response.json()
                logger.info(f"预测结果: 风险分数={prediction_data.get('score', 0):.4f}")
            else:
                logger.warning(f"⚠ 基础预测接口异常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠ 基础预测接口请求失败: {e}")

        # 测试模型列表接口
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                logger.info("✓ 模型列表接口正常")
                models_data = response.json()
                logger.info(f"模型信息: {list(models_data.keys())}")
            else:
                logger.warning(f"⚠ 模型列表接口异常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠ 模型列表接口请求失败: {e}")

        return True

    except Exception as e:
        logger.error(f"API接口测试失败: {e}")
        return False


async def create_test_sample():
    """创建测试样本图像"""
    logger.info("创建测试样本图像...")

    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        # 创建身份证示例图像
        id_card = Image.new('RGB', (400, 250), color='white')
        draw = ImageDraw.Draw(id_card)

        # 绘制基本信息
        draw.rectangle([10, 10, 390, 240], outline='black', width=2)

        # 模拟身份证信息
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((20, 30), "姓名：张三", fill='black', font=font)
        draw.text((20, 60), "性别：男", fill='black', font=font)
        draw.text((20, 90), "民族：汉", fill='black', font=font)
        draw.text((20, 120), "出生：1990年01月01日", fill='black', font=font)
        draw.text((20, 150), "住址：北京市朝阳区", fill='black', font=font)
        draw.text((20, 180), "公民身份号码：110101199001011234", fill='black', font=font)

        # 保存测试图像
        id_card.save("test_id_card.jpg")
        logger.info("✓ 创建身份证测试图像: test_id_card.jpg")

        # 创建票据示例图像
        receipt = Image.new('RGB', (500, 300), color='white')
        draw = ImageDraw.Draw(receipt)

        draw.rectangle([10, 10, 490, 290], outline='black', width=1)

        draw.text((20, 30), "发票", fill='black', font=font)
        draw.text((20, 60), "金额：￥1,000.00", fill='black', font=font)
        draw.text((20, 90), "日期：2024年12月18日", fill='black', font=font)
        draw.text((20, 120), "收款方：测试公司", fill='black', font=font)
        draw.text((20, 150), "付款方：张三", fill='black', font=font)

        receipt.save("test_receipt.jpg")
        logger.info("✓ 创建票据测试图像: test_receipt.jpg")

        return True

    except Exception as e:
        logger.warning(f"创建测试图像失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("=" * 60)
    print("多模态风控系统功能测试")
    print("=" * 60)

    tests = [
        ("基础预测功能", test_basic_prediction),
        ("Ollama客户端", test_ollama_client),
        ("多模态处理器", test_multimodal_processor),
        ("风控分析器", test_fraud_analyzer),
        ("API接口", test_api_endpoints),
        ("创建测试样本", create_test_sample)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n开始测试: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"测试结果: {status}")
        except Exception as e:
            logger.error(f"测试异常: {e}")
            results.append((test_name, False))

    # 测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    print(f"通过率: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常。")
        print("\n下一步：")
        print("1. 启动API服务: python start_multimodal_api.py")
        print("2. 访问API文档: http://localhost:8001/docs")
        print("3. 开始使用多模态功能")
    else:
        print("\n⚠️  部分测试失败，请检查相关配置。")
        print("详细错误信息请查看上面的日志。")


if __name__ == "__main__":
    asyncio.run(main())