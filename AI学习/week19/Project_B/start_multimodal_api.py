"""
多模态API启动脚本
支持智能风控预测与图像分析的完整服务
"""
import os
import sys
import pathlib
import subprocess
import logging
import time

# 添加项目根目录到路径
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查必要的依赖"""
    logger.info("检查依赖包...")

    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'lightgbm',
        'requests', 'shap', 'Pillow', 'opencv-python', 'aiohttp'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"缺少依赖包: {missing_packages}")
        logger.info("请运行: pip install -r requirements.txt")
        return False

    logger.info("✓ 依赖包检查通过")
    return True


def check_project_a():
    """检查Project_A的模型文件"""
    logger.info("检查Project_A模型文件...")

    project_a_root = PROJECT_ROOT.parent / "Project_A"

    # 检查必要文件
    required_files = [
        "models/lgbm_model.pkl",
        "data/processed/scaler.joblib",
        "data/processed/columns.json"
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_a_root / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))

    if missing_files:
        logger.error(f"缺少Project_A文件: {missing_files}")
        logger.info("请确保Project_A已完成模型训练")
        return False

    logger.info("✓ Project_A模型文件检查通过")
    return True


def check_ollama():
    """检查Ollama服务"""
    logger.info("检查Ollama服务...")

    try:
        # 检查Ollama是否安装
        result = subprocess.run(['ollama', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.error("Ollama未安装或不在PATH中")
            logger.info("请访问 https://ollama.com 下载安装")
            return False

        logger.info(f"✓ Ollama版本: {result.stdout.strip()}")

        # 检查Ollama服务是否运行
        from ollama_client import OllamaClient
        client = OllamaClient()

        if client.health_check():
            logger.info("✓ Ollama服务运行正常")

            # 检查模型
            models_info = client.list_available_models()
            all_models = models_info.get("all_models", [])

            if not all_models:
                logger.warning("未发现任何模型")
                logger.info("建议安装基础模型: ollama pull qwen3:4b")
            else:
                logger.info(f"✓ 发现模型: {all_models}")

            # 检查多模态模型
            multimodal_models = models_info.get("multimodal_models", [])
            if not multimodal_models:
                logger.warning("未发现多模态模型")
                logger.info("建议安装多模态模型: ollama pull qwen3-vl:4b")
                logger.info("安装后可使用图像分析功能")
            else:
                logger.info(f"✓ 发现多模态模型: {multimodal_models}")

            return True
        else:
            logger.error("Ollama服务未运行")
            logger.info("请启动Ollama服务: ollama serve")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Ollama命令超时")
        return False
    except Exception as e:
        logger.error(f"检查Ollama时出错: {e}")
        return False


def check_tesseract():
    """检查Tesseract OCR（可选）"""
    logger.info("检查Tesseract OCR...")

    try:
        result = subprocess.run(['tesseract', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✓ Tesseract OCR可用")
            return True
        else:
            logger.warning("Tesseract OCR不可用，图像文字提取功能将受限")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Tesseract OCR未安装，图像文字提取功能将受限")
        logger.info("可选安装：")
        logger.info("  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim")
        logger.info("  macOS: brew install tesseract tesseract-lang")
        logger.info("  Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装")
        return False


def setup_environment():
    """设置环境变量"""
    logger.info("设置环境变量...")

    # 设置Python路径
    os.environ['PYTHONPATH'] = str(SRC_DIR)

    # 设置临时文件目录
    import tempfile
    temp_dir = pathlib.Path(tempfile.gettempdir()) / "multimodal_fraud_analysis"
    temp_dir.mkdir(exist_ok=True)
    os.environ['MULTIMODAL_TEMP_DIR'] = str(temp_dir)

    logger.info("✓ 环境变量设置完成")


def create_config_file():
    """创建配置文件"""
    config_file = PROJECT_ROOT / ".env"

    if not config_file.exists():
        config_content = """# 多模态风控系统配置

# Ollama服务配置
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_TEXT_MODEL=qwen3:4b
DEFAULT_MULTIMODAL_MODEL=qwen3-vl:4b

# API服务配置
HOST=0.0.0.0
PORT=8001
DEBUG=false

# 处理配置
MAX_BATCH_SIZE=100
REQUEST_TIMEOUT=300
CACHE_TTL=3600
MAX_CONCURRENT_LLM=3

# 图像处理配置
MAX_IMAGE_SIZE=10485760  # 10MB
SUPPORTED_IMAGE_FORMATS=jpg,jpeg,png,bmp,tiff,webp

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# 项目路径
PROJECT_A_ROOT=../Project_A
"""

        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)

        logger.info(f"✓ 创建配置文件: {config_file}")
    else:
        logger.info("✓ 配置文件已存在")


def run_api_server():
    """启动API服务器"""
    logger.info("启动多模态API服务器...")

    try:
        # 确保日志目录存在
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)

        # 启动服务器
        import uvicorn
        from api.app_multimodal import app

        # 从环境变量读取配置
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 8001))
        debug = os.getenv('DEBUG', 'false').lower() == 'true'

        logger.info(f"API服务启动地址: http://{host}:{port}")
        logger.info("API文档地址: http://localhost:8001/docs")
        logger.info("按 Ctrl+C 停止服务")

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=debug,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("服务已停止")
    except Exception as e:
        logger.error(f"启动服务失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    print("=" * 60)
    print("智能多模态风控预测与解释系统")
    print("Project_B - Multimodal Fraud Detection System")
    print("=" * 60)

    # 检查系统要求
    checks = [
        ("Python包依赖", check_dependencies),
        ("Project_A模型", check_project_a),
        ("Ollama服务", check_ollama),
        ("Tesseract OCR", check_tesseract)  # 可选
    ]

    failed_checks = []
    for check_name, check_func in checks:
        if not check_func():
            if check_name == "Tesseract OCR":
                continue  # OCR是可选的
            failed_checks.append(check_name)

    if failed_checks:
        logger.error(f"系统检查失败: {failed_checks}")
        logger.info("请解决上述问题后重新运行")
        sys.exit(1)

    logger.info("✓ 系统检查通过")

    # 设置环境
    setup_environment()
    create_config_file()

    # 启动服务
    run_api_server()


if __name__ == "__main__":
    main()