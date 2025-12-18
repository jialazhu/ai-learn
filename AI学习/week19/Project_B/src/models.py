"""
多模态风控系统的数据模型定义
包含请求和响应的数据结构
"""
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import json


class RiskLevel(str, Enum):
    """风险等级枚举"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class ImageType(str, Enum):
    """图像类型枚举"""
    ID_CARD = "id_card"
    SELFIE = "selfie"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    DOCUMENT = "document"
    OTHER = "other"


class AnalysisType(str, Enum):
    """分析类型枚举"""
    PREDICTION_ONLY = "prediction_only"
    TEXT_EXPLANATION = "text_explanation"
    MULTIMODAL_ANALYSIS = "multimodal_analysis"
    COMPREHENSIVE = "comprehensive"


# 基础数据模型
class Features(BaseModel):
    """特征数据模型"""
    data: Dict[str, float] = Field(..., description="30个特征值字典")

    @validator('data')
    def validate_features(cls, v):
        if len(v) != 30:
            raise ValueError(f"特征数量必须为30个，当前为{len(v)}个")

        required_features = [
            'Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
        ]

        missing_features = set(required_features) - set(v.keys())
        if missing_features:
            raise ValueError(f"缺失特征: {sorted(missing_features)}")

        return v


class ImageInfo(BaseModel):
    """图像信息模型"""
    image_type: ImageType = Field(..., description="图像类型")
    filename: Optional[str] = Field(None, description="原始文件名")
    size: Optional[int] = Field(None, description="文件大小（字节）")
    format: Optional[str] = Field(None, description="图像格式")
    width: Optional[int] = Field(None, description="图像宽度")
    height: Optional[int] = Field(None, description="图像高度")
    quality_score: Optional[float] = Field(None, description="图像质量分数")


class ImageAnalysisResult(BaseModel):
    """图像分析结果模型"""
    analysis_type: str = Field(..., description="分析类型")
    risk_level: RiskLevel = Field(..., description="风险等级")
    llm_analysis: Optional[str] = Field(None, description="LLM分析结果")
    image_quality: Optional[Dict[str, Any]] = Field(None, description="图像质量分析")
    extracted_text: Optional[str] = Field(None, description="OCR提取的文本")
    face_detection: Optional[Dict[str, Any]] = Field(None, description="人脸检测结果")
    amount_match: Optional[Dict[str, Any]] = Field(None, description="金额匹配验证")
    model_used: Optional[str] = Field(None, description="使用的模型")
    error: Optional[str] = Field(None, description="错误信息")
    has_selfie: Optional[bool] = Field(False, description="是否包含自拍")


class TransactionPrediction(BaseModel):
    """交易预测结果模型"""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="风险分数（0-1）")
    risk_label: int = Field(..., ge=0, le=1, description="风险标签（0=正常，1=欺诈）")
    base_explanation: Optional[str] = Field(None, description="基础模型解释")
    shap_explanation: Optional[Dict[str, Any]] = Field(None, description="SHAP解释结果")


class OverallRisk(BaseModel):
    """综合风险评估模型"""
    level: RiskLevel = Field(..., description="综合风险等级")
    score: float = Field(..., ge=0.0, le=1.0, description="综合风险分数")
    base_risk_score: Optional[float] = Field(None, description="基础特征风险分数")
    image_risk_score: Optional[float] = Field(None, description="图像分析风险分数")
    components: Optional[Dict[str, Any]] = Field(None, description="风险组成部分")
    error: Optional[str] = Field(None, description="错误信息")


# 请求模型
class PredictionRequest(BaseModel):
    """基础预测请求模型"""
    features: Dict[str, float] = Field(..., description="30个特征值")
    explain: bool = Field(True, description="是否生成SHAP解释")
    use_llm: bool = Field(True, description="是否使用LLM生成解释")
    model_name: str = Field("qwen3:4b", description="LLM模型名称")

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "Time": 0.0,
                    "V1": -1.3598,
                    "V2": -0.0727,
                    "Amount": 149.62
                },
                "explain": True,
                "use_llm": True,
                "model_name": "qwen3:4b"
            }
        }


class MultimodalPredictionRequest(BaseModel):
    """多模态预测请求模型"""
    features: Dict[str, float] = Field(..., description="30个特征值")
    explain: bool = Field(True, description="是否生成解释")
    use_llm: bool = Field(True, description="是否使用LLM解释")
    enable_image_analysis: bool = Field(True, description="是否启用图像分析")
    model_name: str = Field("qwen3-vl:4b", description="多模态LLM模型名称")
    fallback_model: str = Field("qwen3:4b", description="回退文本模型名称")
    analysis_types: List[AnalysisType] = Field(
        default=[AnalysisType.COMPREHENSIVE],
        description="分析类型列表"
    )


class BatchPredictionRequest(BaseModel):
    """批量预测请求模型"""
    batch_id: str = Field(..., description="批次ID")
    samples: List[Dict[str, float]] = Field(..., description="样本列表")
    explain: bool = Field(True, description="是否生成解释")
    use_llm: bool = Field(False, description="是否使用LLM解释（批量时建议关闭）")
    callback_url: Optional[str] = Field(None, description="完成后的回调URL")

    @validator('samples')
    def validate_samples(cls, v):
        if not v:
            raise ValueError("样本列表不能为空")
        if len(v) > 1000:  # 限制批量大小
            raise ValueError("批量样本数量不能超过1000")
        return v


# 响应模型
class PredictionResponse(BaseModel):
    """基础预测响应模型"""
    score: float = Field(..., description="风险分数（0-1之间）")
    label: int = Field(..., description="预测标签（0=正常，1=欺诈）")
    explanation: Optional[Dict[str, Any]] = Field(None, description="SHAP特征重要性解释")
    llm_explanation: Optional[str] = Field(None, description="LLM生成的中文解释报告")
    strategy_suggestion: Optional[str] = Field(None, description="风控策略建议")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")


class MultimodalPredictionResponse(BaseModel):
    """多模态预测响应模型"""
    transaction_prediction: Optional[TransactionPrediction] = Field(None, description="交易预测结果")
    image_analysis: Optional[Dict[str, ImageAnalysisResult]] = Field(None, description="图像分析结果")
    overall_risk: Optional[OverallRisk] = Field(None, description="综合风险评估")
    comprehensive_explanation: Optional[str] = Field(None, description="综合解释报告")
    model_used: Optional[str] = Field(None, description="使用的模型")
    analysis_timestamp: Optional[float] = Field(None, description="分析时间戳")
    processing_time: Optional[float] = Field(None, description="总处理时间")
    error: Optional[str] = Field(None, description="错误信息")


class BatchPredictionResponse(BaseModel):
    """批量预测响应模型"""
    batch_id: str = Field(..., description="批次ID")
    status: str = Field(..., description="处理状态")
    total_samples: int = Field(..., description="总样本数")
    processed_samples: int = Field(..., description="已处理样本数")
    results: Optional[List[PredictionResponse]] = Field(None, description="预测结果列表")
    summary: Optional[Dict[str, Any]] = Field(None, description="结果摘要")
    processing_time: Optional[float] = Field(None, description="总处理时间")
    callback_url: Optional[str] = Field(None, description="回调URL")


class ImageAnalysisResponse(BaseModel):
    """纯图像分析响应模型"""
    analysis_type: str = Field(..., description="分析类型")
    risk_level: RiskLevel = Field(..., description="风险等级")
    llm_analysis: str = Field(..., description="LLM分析结果")
    image_quality: Dict[str, Any] = Field(..., description="图像质量分析")
    extracted_text: Optional[str] = Field(None, description="OCR提取文本")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="附加信息")
    model_used: str = Field(..., description="使用的模型")
    processing_time: Optional[float] = Field(None, description="处理时间")


# 配置模型
class ModelConfig(BaseModel):
    """模型配置"""
    project_a_root: str = Field(..., description="Project_A根目录路径")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama服务地址")
    default_text_model: str = Field("qwen3:4b", description="默认文本模型")
    default_multimodal_model: str = Field("qwen3-vl:4b", description="默认多模态模型")
    cache_ttl: int = Field(3600, description="缓存过期时间（秒）")
    max_concurrent_llm: int = Field(3, description="最大并发LLM请求数")
    max_image_size: int = Field(10 * 1024 * 1024, description="最大图像文件大小（字节）")
    supported_image_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        description="支持的图像格式"
    )


class ProcessingConfig(BaseModel):
    """处理配置"""
    enable_async: bool = Field(True, description="是否启用异步处理")
    enable_caching: bool = Field(True, description="是否启用缓存")
    enable_logging: bool = Field(True, description="是否启用详细日志")
    max_batch_size: int = Field(100, description="最大批量处理大小")
    request_timeout: int = Field(300, description="请求超时时间（秒）")
    retry_attempts: int = Field(3, description="重试次数")


# 错误模型
class ErrorResponse(BaseModel):
    """错误响应模型"""
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误信息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: Optional[float] = Field(None, description="错误时间戳")


class ValidationErrorResponse(BaseModel):
    """验证错误响应模型"""
    error_code: str = Field("VALIDATION_ERROR", description="错误代码")
    error_message: str = Field("请求参数验证失败", description="错误信息")
    validation_errors: List[Dict[str, str]] = Field(..., description="具体验证错误")
    timestamp: Optional[float] = Field(None, description="错误时间戳")


# 健康检查模型
class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    timestamp: float = Field(..., description="检查时间")
    components: Dict[str, Dict[str, Any]] = Field(..., description="各组件状态")
    uptime: Optional[float] = Field(None, description="服务运行时间")
    version: Optional[str] = Field(None, description="服务版本")


class ComponentStatus(BaseModel):
    """组件状态模型"""
    name: str = Field(..., description="组件名称")
    status: str = Field(..., description="组件状态")
    last_check: float = Field(..., description="最后检查时间")
    response_time: Optional[float] = Field(None, description="响应时间")
    error_message: Optional[str] = Field(None, description="错误信息")
    details: Optional[Dict[str, Any]] = Field(None, description="附加信息")


# 统计模型
class ProcessingStats(BaseModel):
    """处理统计模型"""
    total_requests: int = Field(0, description="总请求数")
    successful_requests: int = Field(0, description="成功请求数")
    failed_requests: int = Field(0, description="失败请求数")
    avg_response_time: float = Field(0.0, description="平均响应时间")
    requests_per_minute: float = Field(0.0, description="每分钟请求数")
    model_usage: Dict[str, int] = Field(default_factory=dict, description="模型使用统计")
    error_distribution: Dict[str, int] = Field(default_factory=dict, description="错误分布")


class ModelStats(BaseModel):
    """模型统计模型"""
    model_name: str = Field(..., description="模型名称")
    total_calls: int = Field(0, description="总调用次数")
    successful_calls: int = Field(0, description="成功调用次数")
    failed_calls: int = Field(0, description="失败调用次数")
    avg_response_time: float = Field(0.0, description="平均响应时间")
    last_used: Optional[float] = Field(None, description="最后使用时间")
    is_available: bool = Field(True, description="是否可用")


# 缓存模型
class CacheEntry(BaseModel):
    """缓存条目模型"""
    key: str = Field(..., description="缓存键")
    value: Any = Field(..., description="缓存值")
    created_at: float = Field(..., description="创建时间")
    expires_at: float = Field(..., description="过期时间")
    access_count: int = Field(0, description="访问次数")
    last_accessed: float = Field(..., description="最后访问时间")


class CacheStats(BaseModel):
    """缓存统计模型"""
    total_entries: int = Field(0, description="总缓存条目数")
    hit_count: int = Field(0, description="命中次数")
    miss_count: int = Field(0, description="未命中次数")
    hit_rate: float = Field(0.0, description="命中率")
    memory_usage: int = Field(0, description="内存使用量（字节）")
    oldest_entry: Optional[float] = Field(None, description="最旧条目时间")
    newest_entry: Optional[float] = Field(None, description="最新条目时间")