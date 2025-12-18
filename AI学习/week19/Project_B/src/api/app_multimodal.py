"""
Project_B 多模态API服务：整合LightGBM预测、图像分析和Ollama解释
支持文件上传、多模态分析和综合风险评估
"""
import os
import pathlib
import sys
import asyncio
import logging
import time
import tempfile
import uuid
from typing import Dict, List, Optional, Union

# 添加项目根目录到路径
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import json

# 导入自定义模块
from explain import load_model, explain_prediction, generate_explanation_text
from ollama_client import OllamaClient
from fraud_multimodal_analyzer import FraudMultimodalAnalyzer
from multimodal_processor import multimodal_processor
from models import (
    PredictionRequest, PredictionResponse,
    MultimodalPredictionRequest, MultimodalPredictionResponse,
    ImageAnalysisResponse, BatchPredictionRequest, BatchPredictionResponse,
    HealthCheckResponse, ErrorResponse, ComponentStatus,
    ImageType, AnalysisType, RiskLevel, Features
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 路径配置（使用Project_A的模型）
PROJECT_A_ROOT = PROJECT_ROOT.parent / "Project_A"
MODEL_DIR = PROJECT_A_ROOT / "models"
DATA_DIR = PROJECT_A_ROOT / "data" / "processed"

# 全局变量
model = None
scaler = None
feature_names = None
ollama_client = None
multimodal_analyzer = None

app = FastAPI(
    title="智能多模态风控预测与解释API",
    version="1.0.0",
    description="""
    ## 智能多模态风控预测与解释系统 API

    本API提供以下功能：

    * **欺诈预测**：使用LightGBM模型预测信用卡交易是否为欺诈
    * **SHAP解释**：分析哪些特征对预测结果影响最大
    * **多模态图像分析**：支持身份证、票据、自拍等图像分析
    * **LLM智能解释**：使用qwen3-vl:4b生成图文并茂的中文解释报告
    * **综合风险评估**：结合交易特征和图像分析的智能风控
    * **批量处理**：支持批量交易分析
    * **实时监控**：提供系统健康检查和性能监控

    ### 多模态支持

    - **身份证分析**：检测身份证真实性、人脸比对
    - **票据验证**：分析票据真伪、金额匹配验证
    - **综合分析**：多图像联合分析，提供全面风险评估

    ### 使用说明

    1. 确保Ollama服务正在运行（默认端口11434）
    2. 安装多模态模型：`ollama pull qwen3-vl:4b`
    3. 确保Project_A的模型文件已准备好
    4. 使用 `/predict/multimodal` 接口进行多模态分析
    5. 查看 `/docs` 获取完整的API文档
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "预测",
            "description": "欺诈预测相关接口"
        },
        {
            "name": "多模态分析",
            "description": "图像和多媒体分析接口"
        },
        {
            "name": "批量处理",
            "description": "批量预测和处理接口"
        },
        {
            "name": "健康检查",
            "description": "服务健康检查接口"
        }
    ]
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件目录
TEMP_DIR = pathlib.Path(tempfile.gettempdir()) / "multimodal_fraud_analysis"
TEMP_DIR.mkdir(exist_ok=True)


async def get_multimodal_analyzer():
    """获取多模态分析器实例"""
    global multimodal_analyzer
    if multimodal_analyzer is None:
        multimodal_analyzer = FraudMultimodalAnalyzer(
            ollama_client=ollama_client,
            model=model,
            scaler=scaler,
            feature_names=feature_names
        )
    return multimodal_analyzer


def cleanup_temp_files():
    """清理临时文件"""
    try:
        import glob
        temp_files = glob.glob(str(TEMP_DIR / "*"))
        for file_path in temp_files:
            try:
                os.remove(file_path)
            except:
                pass
    except Exception as e:
        logger.warning(f"清理临时文件失败: {e}")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global model, scaler, feature_names, ollama_client, multimodal_analyzer

    try:
        # 加载预测模型
        model, scaler, feature_names = load_model()
        logger.info("✓ 预测模型加载成功")

        # 初始化Ollama客户端
        ollama_client = OllamaClient()
        if ollama_client.health_check():
            logger.info("✓ Ollama服务连接成功")

            # 检查多模态模型
            models_info = ollama_client.list_available_models()
            multimodal_models = models_info.get("multimodal_models", [])
            if multimodal_models:
                logger.info(f"✓ 发现多模态模型: {multimodal_models}")
            else:
                logger.warning("⚠ 未发现多模态模型，建议安装: ollama pull qwen3-vl:4b")
        else:
            logger.error("✗ Ollama服务不可用")
            raise Exception("Ollama服务连接失败")

        # 初始化多模态分析器
        multimodal_analyzer = FraudMultimodalAnalyzer(
            ollama_client=ollama_client,
            model=model,
            scaler=scaler,
            feature_names=feature_names
        )
        logger.info("✓ 多模态分析器初始化完成")

        # 清理旧的临时文件
        cleanup_temp_files()

    except Exception as e:
        logger.error(f"✗ 启动失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    try:
        cleanup_temp_files()
        logger.info("✓ 资源清理完成")
    except Exception as e:
        logger.warning(f"⚠ 清理资源时出错: {e}")


# 错误处理
@app.exception_handler(ValueError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "error_message": str(exc),
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "error_message": "服务器内部错误",
            "timestamp": time.time()
        }
    )


# 基础接口（保持向后兼容）
@app.get("/", tags=["健康检查"])
async def root():
    """根路径 - 获取API基本信息"""
    return {
        "message": "智能多模态风控预测与解释API",
        "version": "1.0.0",
        "description": "整合LightGBM预测、图像分析和Ollama LLM解释的智能风控系统",
        "features": [
            "欺诈预测",
            "多模态图像分析",
            "SHAP特征解释",
            "LLM智能解释",
            "综合风险评估",
            "批量处理"
        ],
        "endpoints": {
            "health": "/health - 服务健康检查",
            "predict": "/predict - 基础预测接口",
            "predict_multimodal": "/predict/multimodal - 多模态预测接口",
            "analyze_image": "/analyze/image - 图像分析接口",
            "batch_predict": "/predict/batch - 批量预测接口",
            "docs": "/docs - API文档"
        },
        "models": {
            "prediction": "LightGBM",
            "llm_text": "qwen3:4b",
            "llm_multimodal": "qwen3-vl:4b"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["健康检查"])
async def health_check():
    """详细健康检查"""
    components = {}
    current_time = time.time()

    # 检查预测模型
    components["prediction_model"] = {
        "status": "ok" if model is not None else "error",
        "last_check": current_time,
        "details": {
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "features_loaded": feature_names is not None
        }
    }

    # 检查Ollama服务
    ollama_status = ollama_client.health_check() if ollama_client else False
    components["ollama_service"] = {
        "status": "ok" if ollama_status else "error",
        "last_check": current_time,
        "response_time": 0.1,
        "details": {
            "service_available": ollama_status,
            "base_url": ollama_client.base_url if ollama_client else None
        }
    }

    # 检查多模态模型
    if ollama_status:
        try:
            models_info = ollama_client.list_available_models()
            multimodal_models = models_info.get("multimodal_models", [])
            components["multimodal_models"] = {
                "status": "ok" if multimodal_models else "warning",
                "last_check": current_time,
                "details": {
                    "available_models": multimodal_models,
                    "recommended_model": "qwen3-vl:4b"
                }
            }
        except Exception as e:
            components["multimodal_models"] = {
                "status": "error",
                "last_check": current_time,
                "error_message": str(e)
            }

    # 检查临时目录
    try:
        temp_accessible = os.access(TEMP_DIR, os.W_OK)
        components["temp_directory"] = {
            "status": "ok" if temp_accessible else "error",
            "last_check": current_time,
            "details": {
                "path": str(TEMP_DIR),
                "writable": temp_accessible
            }
        }
    except Exception as e:
        components["temp_directory"] = {
            "status": "error",
            "last_check": current_time,
            "error_message": str(e)
        }

    # 整体状态
    all_ok = all(comp["status"] == "ok" for comp in components.values())

    return HealthCheckResponse(
        status="ok" if all_ok else "degraded",
        timestamp=current_time,
        components=components,
        uptime=current_time  # 这里应该是实际运行时间
    )


@app.post("/predict", response_model=PredictionResponse, tags=["预测"])
async def predict(request: PredictionRequest):
    """
    基础预测接口（向后兼容）

    支持传统的文本特征预测和解释
    """
    start_time = time.time()

    try:
        # 验证特征
        features = Features(data=request.features)

        # 预测
        explanation_data = explain_prediction(
            model, scaler, feature_names, request.features
        )

        score = explanation_data["prediction_score"]
        label = explanation_data["prediction_label"]

        response = PredictionResponse(
            score=score,
            label=label
        )

        # 生成解释
        if request.explain:
            response.explanation = {
                "top_features": explanation_data["top_features"],
                "explanation_text": generate_explanation_text(explanation_data)
            }

            # 使用LLM生成解释
            if request.use_llm and ollama_client:
                try:
                    explanation_text = response.explanation["explanation_text"]
                    llm_explanation = ollama_client.explain_prediction(
                        model=request.model_name,
                        explanation_text=explanation_text
                    )
                    response.llm_explanation = llm_explanation

                    # 生成策略建议
                    if label == 1:
                        strategy = ollama_client.generate_strategy(
                            model=request.model_name,
                            risk_summary=explanation_text
                        )
                        response.strategy_suggestion = strategy
                except Exception as e:
                    logger.warning(f"LLM解释生成失败: {e}")

        response.processing_time = time.time() - start_time
        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/multimodal", response_model=MultimodalPredictionResponse, tags=["多模态分析"])
async def predict_multimodal(
    background_tasks: BackgroundTasks,
    features: str = Form(..., description="JSON格式的特征数据"),
    images: List[UploadFile] = File(default=[], description="上传的图像文件"),
    image_types: List[str] = Form(default=[], description="图像类型列表"),
    explain: bool = Form(True, description="是否生成解释"),
    use_llm: bool = Form(True, description="是否使用LLM解释"),
    model_name: str = Form("qwen3-vl:4b", description="LLM模型名称"),
    fallback_model: str = Form("qwen3:4b", description="回退模型名称")
):
    """
    多模态预测接口

    支持文本特征和图像的联合分析

    **使用方式**：
    - features: JSON格式的30个特征值
    - images: 上传的图像文件（身份证、票据、自拍等）
    - image_types: 对应图像的类型（id_card, receipt, selfie等）

    **示例**：
    ```
    curl -X POST "http://localhost:8001/predict/multimodal" \
      -F "features='{\"Time\": 0.0, \"Amount\": 100.0, ...}'" \
      -F "images=@id_card.jpg" \
      -F "image_types=id_card"
    ```
    """
    start_time = time.time()

    try:
        # 解析特征数据
        try:
            transaction_features = json.loads(features)
        except json.JSONDecodeError:
            raise ValueError("特征数据格式错误，需要有效的JSON")

        # 验证特征
        Features(data=transaction_features)

        # 处理上传的图像
        uploaded_images = {}
        temp_files = []

        try:
            for i, (image_file, img_type) in enumerate(zip(images, image_types)):
                if not image_file.filename:
                    continue

                # 验证图像格式
                if not multimodal_processor.validate_image_format(image_file.filename):
                    raise ValueError(f"不支持的图像格式: {image_file.filename}")

                # 保存到临时文件
                file_extension = pathlib.Path(image_file.filename).suffix
                temp_filename = f"{uuid.uuid4()}{file_extension}"
                temp_path = TEMP_DIR / temp_filename

                with open(temp_path, "wb") as temp_file:
                    content = await image_file.read()
                    temp_file.write(content)

                temp_files.append(temp_path)
                uploaded_images[img_type] = str(temp_path)

                logger.info(f"图像已保存: {img_type} -> {temp_path}")

            # 获取多模态分析器
            analyzer = await get_multimodal_analyzer()

            # 执行多模态分析
            if uploaded_images:
                result = await analyzer.comprehensive_analysis(
                    transaction_features=transaction_features,
                    images=uploaded_images,
                    model_name=model_name
                )
            else:
                # 如果没有图像，回退到基础预测
                logger.warning("未上传图像，执行基础预测")
                explanation_data = explain_prediction(
                    model, scaler, feature_names, transaction_features
                )

                base_explanation = generate_explanation_text(explanation_data)

                if use_llm and ollama_client:
                    llm_explanation = ollama_client.explain_prediction(
                        model=fallback_model,
                        explanation_text=base_explanation
                    )
                else:
                    llm_explanation = None

                result = {
                    "transaction_prediction": {
                        "risk_score": explanation_data["prediction_score"],
                        "risk_label": explanation_data["prediction_label"],
                        "base_explanation": base_explanation
                    },
                    "image_analysis": {},
                    "overall_risk": {
                        "level": "MEDIUM" if explanation_data["prediction_label"] else "LOW",
                        "score": explanation_data["prediction_score"]
                    },
                    "comprehensive_explanation": llm_explanation
                }

            # 构建响应
            response = MultimodalPredictionResponse(
                transaction_prediction=result.get("transaction_prediction"),
                image_analysis=result.get("image_analysis"),
                overall_risk=result.get("overall_risk"),
                comprehensive_explanation=result.get("comprehensive_explanation"),
                model_used=model_name,
                analysis_timestamp=result.get("analysis_timestamp"),
                processing_time=time.time() - start_time
            )

            # 添加清理任务
            background_tasks.add_task(cleanup_temp_files)

            return response

        except Exception as e:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
            raise e

    except Exception as e:
        logger.error(f"多模态预测失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze/image", response_model=ImageAnalysisResponse, tags=["多模态分析"])
async def analyze_image_only(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="要分析的图像文件"),
    image_type: str = Form(..., description="图像类型 (id_card, receipt, selfie, document)"),
    features: Optional[str] = Form(None, description="关联的交易特征JSON（可选）"),
    model_name: str = Form("qwen3-vl:4b", description="LLM模型名称")
):
    """
    纯图像分析接口

    对上传的图像进行专业分析，不依赖预测结果

    **支持的图像类型**：
    - id_card: 身份证分析
    - receipt: 票据分析
    - selfie: 自拍分析
    - document: 文档分析
    """
    start_time = time.time()
    temp_path = None

    try:
        # 验证图像类型
        try:
            img_type = ImageType(image_type)
        except ValueError:
            raise ValueError(f"不支持的图像类型: {image_type}. 支持的类型: {[t.value for t in ImageType]}")

        # 验证图像格式
        if not multimodal_processor.validate_image_format(image.filename):
            raise ValueError(f"不支持的图像格式: {image.filename}")

        # 保存临时文件
        file_extension = pathlib.Path(image.filename).suffix
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = TEMP_DIR / temp_filename

        with open(temp_path, "wb") as temp_file:
            content = await image.read()
            temp_file.write(content)

        # 解析特征数据（如果有）
        transaction_features = {}
        if features:
            try:
                transaction_features = json.loads(features)
            except json.JSONDecodeError:
                logger.warning("特征数据格式错误，将使用空特征")

        # 获取分析器
        analyzer = await get_multimodal_analyzer()

        # 执行图像分析
        if img_type == ImageType.ID_CARD:
            result = await analyzer.analyze_id_card(
                transaction_features=transaction_features,
                id_card_image=str(temp_path),
                model_name=model_name
            )
        elif img_type == ImageType.RECEIPT:
            result = await analyzer.analyze_receipt(
                transaction_features=transaction_features,
                receipt_image=str(temp_path),
                model_name=model_name
            )
        else:
            # 通用图像分析
            image_obj = multimodal_processor.load_image_from_path(str(temp_path))
            image_quality = multimodal_processor.analyze_image_quality(image_obj)
            extracted_text = multimodal_processor.extract_text_from_image(image_obj)

            result = {
                "analysis_type": image_type,
                "risk_level": RiskLevel.UNKNOWN,
                "llm_analysis": "通用图像分析完成",
                "image_quality": image_quality,
                "extracted_text": extracted_text,
                "model_used": model_name
            }

        # 构建响应
        response = ImageAnalysisResponse(
            analysis_type=result["analysis_type"],
            risk_level=RiskLevel(result.get("risk_level", "UNKNOWN")),
            llm_analysis=result.get("llm_analysis", ""),
            image_quality=result.get("image_quality", {}),
            extracted_text=result.get("extracted_text"),
            additional_info={
                "face_detection": result.get("face_detection"),
                "amount_match": result.get("amount_match"),
                "has_selfie": result.get("has_selfie", False)
            },
            model_used=result.get("model_used", model_name),
            processing_time=time.time() - start_time
        )

        # 添加清理任务
        background_tasks.add_task(cleanup_temp_files)

        return response

    except Exception as e:
        logger.error(f"图像分析失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["批量处理"])
async def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """
    批量预测接口

    支持大量样本的批量处理，适合离线分析场景
    """
    start_time = time.time()

    try:
        # 验证请求
        if not request.samples:
            raise ValueError("样本列表不能为空")

        # 批量处理
        results = []
        successful_count = 0
        failed_count = 0

        for i, sample_features in enumerate(request.samples):
            try:
                # 验证特征
                Features(data=sample_features)

                # 预测
                explanation_data = explain_prediction(
                    model, scaler, feature_names, sample_features
                )

                # 基础响应
                result = PredictionResponse(
                    score=explanation_data["prediction_score"],
                    label=explanation_data["prediction_label"]
                )

                # 添加基础解释
                if request.explain:
                    result.explanation = {
                        "top_features": explanation_data["top_features"][:5],  # 限制返回数量
                        "explanation_text": generate_explanation_text(explanation_data)
                    }

                results.append(result)
                successful_count += 1

            except Exception as e:
                logger.warning(f"样本 {i} 处理失败: {e}")
                failed_count += 1
                # 添加错误结果
                results.append(PredictionResponse(
                    score=0.0,
                    label=0,
                    explanation={"error": str(e)}
                ))

        # 生成摘要统计
        risk_scores = [r.score for r in results if not hasattr(r, 'explanation') or 'error' not in getattr(r.explanation, {})]
        summary = {
            "total_samples": len(request.samples),
            "successful": successful_count,
            "failed": failed_count,
            "success_rate": successful_count / len(request.samples),
            "avg_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
            "high_risk_count": sum(1 for r in results if r.score >= 0.7),
            "medium_risk_count": sum(1 for r in results if 0.3 <= r.score < 0.7),
            "low_risk_count": sum(1 for r in results if r.score < 0.3)
        }

        response = BatchPredictionResponse(
            batch_id=request.batch_id,
            status="completed",
            total_samples=len(request.samples),
            processed_samples=len(results),
            results=results,
            summary=summary,
            processing_time=time.time() - start_time,
            callback_url=request.callback_url
        )

        # 如果有回调URL，添加后台任务
        if request.callback_url:
            background_tasks.add_task(send_callback, request.callback_url, response.dict())

        return response

    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


async def send_callback(callback_url: str, data: dict):
    """发送回调通知"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(callback_url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    logger.info(f"回调发送成功: {callback_url}")
                else:
                    logger.warning(f"回调发送失败，状态码: {resp.status}")
    except Exception as e:
        logger.error(f"回调发送异常: {e}")


@app.get("/models", tags=["健康检查"])
async def list_models():
    """获取可用模型列表"""
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama客户端未初始化")

        models_info = ollama_client.list_available_models()
        return {
            "prediction_model": {
                "type": "LightGBM",
                "status": "loaded" if model is not None else "not_loaded"
            },
            "ollama_models": models_info,
            "recommended": {
                "text": "qwen3:4b",
                "multimodal": "qwen3-vl:4b"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "app_multimodal:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )