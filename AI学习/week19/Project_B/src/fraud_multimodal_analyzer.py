"""
风控多模态分析器
整合传统风控特征分析和多模态图像分析，提供综合欺诈检测
"""
import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple
import json
import re

from .ollama_client import OllamaClient
from .multimodal_processor import MultimodalProcessor, multimodal_processor
from .explain import explain_prediction, generate_explanation_text

logger = logging.getLogger(__name__)


class FraudMultimodalAnalyzer:
    """多模态欺诈检测分析器"""

    def __init__(self, ollama_client: OllamaClient, model=None, scaler=None, feature_names=None):
        """
        初始化多模态分析器

        Args:
            ollama_client: Ollama客户端实例
            model: 训练好的预测模型
            scaler: 特征标准化器
            feature_names: 特征名称列表
        """
        self.ollama_client = ollama_client
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.processor = multimodal_processor

        # 默认多模态模型
        self.default_multimodal_model = "qwen3-vl:4b"
        self.fallback_text_model = "qwen3:4b"

    async def analyze_id_card(
        self,
        transaction_features: Dict[str, float],
        id_card_image: Union[str, bytes],
        selfie_image: Optional[Union[str, bytes]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, any]:
        """
        身份证图像分析

        Args:
            transaction_features: 交易特征数据
            id_card_image: 身份证图像路径或字节数据
            selfie_image: 自拍图像（可选）
            model_name: 使用的LLM模型名称

        Returns:
            Dict: 身份证分析结果
        """
        try:
            model_name = model_name or self.default_multimodal_model

            # 图像预处理
            if isinstance(id_card_image, str):
                id_image = self.processor.load_image_from_path(id_card_image)
            else:
                id_image = self.processor.load_image_from_bytes(id_card_image)

            id_image = self.processor.resize_image(id_image)
            id_image = self.processor.enhance_image_quality(id_image)
            id_image_b64 = self.processor.encode_pil_image_to_base64(id_image)

            images = [id_image_b64]

            # 如果有自拍图像，也进行处理
            if selfie_image:
                if isinstance(selfie_image, str):
                    selfie = self.processor.load_image_from_path(selfie_image)
                else:
                    selfie = self.processor.load_image_from_bytes(selfie_image)

                selfie = self.processor.resize_image(selfie)
                selfie = self.processor.enhance_image_quality(selffie)
                images.append(self.processor.encode_pil_image_to_base64(selffie))

            # 构建分析提示词
            system_prompt = """你是一个专业的反欺诈分析师和身份验证专家。
请仔细分析提供的身份证图像（如果有自拍还需进行人脸比对），重点关注：

1. **身份证真实性检查**：
   - 身份证格式规范性和完整性
   - 国徽、防伪图案、微缩文字等细节
   - 印章清晰度和规范性
   - 是否有PS、篡改、复制的痕迹

2. **图像质量分析**：
   - 图像清晰度和完整性
   - 光线是否均匀，是否有遮挡
   - 是否有模糊、反光等问题

3. **人脸比对**（如果有自拍）：
   - 人脸特征相似度
   - 面部轮廓匹配度
   - 表情、角度差异是否合理

4. **信息一致性**：
   - 身份证信息是否完整清晰
   - 关键信息是否合理

请用中文提供详细分析结果，并给出真实性评估和风险等级。"""

            # 提取交易特征摘要
            risk_score = transaction_features.get('risk_score', '未知')
            amount = transaction_features.get('Amount', 0)
            time_feature = transaction_features.get('Time', 0)

            user_prompt = f"""请分析以下交易数据和身份证图像：

**交易特征摘要**：
- 风险分数: {risk_score}
- 交易金额: {amount:.2f}
- 时间特征: {time_feature}
- 其他特征: V1-V28等特征值已包含在风险评估中

**分析任务**：
1. 评估身份证的真实性和完整性
2. 检测可能的伪造或篡改痕迹
3. 如有自拍，进行人脸比对分析
4. 结合交易特征评估整体风险
5. 给出具体的欺诈风险等级和建议

请提供详细的分析报告，包括具体的风险点和处理建议。"""

            messages = [{"role": "user", "content": user_prompt}]

            # 调用多模态LLM
            if self.ollama_client.is_multimodal_model(model_name):
                analysis = await self.ollama_client.chat_multimodal_async(
                    model=model_name,
                    messages=messages,
                    images=images,
                    system=system_prompt
                )
            else:
                # 如果不是多模态模型，使用文本模式并添加图像分析结果
                image_quality = self.processor.analyze_image_quality(id_image)
                extracted_text = self.processor.extract_text_from_image(id_image)
                face_detection = self.processor.detect_faces(id_image)

                fallback_prompt = f"""{user_prompt}

**图像分析结果**：
- 图像质量: {image_quality.get('quality_grade', '未知')}
- 清晰度: {image_quality.get('sharpness', 0):.2f}
- 亮度: {image_quality.get('brightness', 0):.2f}
- 对比度: {image_quality.get('contrast', 0):.2f}
- 人脸检测: 发现 {face_detection.get('faces_detected', 0)} 张人脸

**OCR提取文本**：
{extracted_text[:500] if extracted_text else "OCR识别失败或无文本"}

请基于以上信息进行分析。"""

                messages[0]["content"] = fallback_prompt
                analysis = await self.ollama_client.chat_async(
                    model=model_name,
                    messages=messages,
                    system=system_prompt
                )

            # 提取风险等级
            risk_level = self._extract_risk_level(analysis)

            return {
                "analysis_type": "id_card",
                "llm_analysis": analysis,
                "risk_level": risk_level,
                "image_quality": self.processor.analyze_image_quality(id_image),
                "extracted_text": self.processor.extract_text_from_image(id_image),
                "face_detection": face_detection if selfie_image else self.processor.detect_faces(id_image),
                "model_used": model_name,
                "has_selfie": selfie_image is not None
            }

        except Exception as e:
            logger.error(f"身份证分析失败: {e}")
            return {
                "analysis_type": "id_card",
                "error": f"身份证分析失败: {str(e)}",
                "risk_level": "ERROR"
            }

    async def analyze_receipt(
        self,
        transaction_features: Dict[str, float],
        receipt_image: Union[str, bytes],
        model_name: Optional[str] = None
    ) -> Dict[str, any]:
        """
        票据图像分析

        Args:
            transaction_features: 交易特征数据
            receipt_image: 票据图像路径或字节数据
            model_name: 使用的LLM模型名称

        Returns:
            Dict: 票据分析结果
        """
        try:
            model_name = model_name or self.default_multimodal_model

            # 图像预处理
            if isinstance(receipt_image, str):
                receipt = self.processor.load_image_from_path(receipt_image)
            else:
                receipt = self.processor.load_image_from_bytes(receipt_image)

            receipt = self.processor.resize_image(receipt)
            receipt = self.processor.enhance_image_quality(receipt)
            receipt_b64 = self.processor.encode_pil_image_to_base64(receipt)

            system_prompt = """你是一个票据真伪鉴定和财务审计专家。
请仔细分析提供的票据图像，重点关注：

1. **票据真实性检查**：
   - 票据格式是否符合规范
   - 印章、签名是否真实有效
   - 是否有伪造、篡改痕迹
   - 票据完整性检查

2. **财务信息验证**：
   - 金额数字和文字是否一致
   - 日期格式是否规范
   - 收款方和付款方信息
   - 税务信息（发票专用章等）

3. **图像质量评估**：
   - 清晰度和完整性
   - 关键信息是否可读
   - 是否有遮挡或模糊

请用中文提供详细分析，并给出票据真实性评估和风险等级。"""

            amount = transaction_features.get('Amount', 0)
            risk_score = transaction_features.get('risk_score', '未知')

            user_prompt = f"""请分析以下交易特征和票据图像：

**交易信息**：
- 交易金额: {amount:.2f}
- 系统风险评分: {risk_score}

**分析任务**：
1. 验证票据的真实性和完整性
2. 检查金额、日期等关键信息
3. 识别可能的伪造痕迹
4. 评估票据与交易的一致性
5. 给出风险等级和处理建议

请提供详细的票据分析报告。"""

            messages = [{"role": "user", "content": user_prompt}]

            # 调用多模态LLM
            if self.ollama_client.is_multimodal_model(model_name):
                analysis = await self.ollama_client.chat_multimodal_async(
                    model=model_name,
                    messages=messages,
                    images=[receipt_b64],
                    system=system_prompt
                )
            else:
                # 回退到文本模式
                image_quality = self.processor.analyze_image_quality(receipt)
                extracted_text = self.processor.extract_text_from_image(receipt)

                fallback_prompt = f"""{user_prompt}

**图像分析结果**：
- 图像质量等级: {image_quality.get('quality_grade', '未知')}
- 清晰度评分: {image_quality.get('sharpness', 0):.2f}
- 亮度水平: {image_quality.get('brightness', 0):.2f}

**OCR提取文本**：
{extracted_text[:800] if extracted_text else "OCR识别失败"}

请基于以上信息进行票据分析。"""

                messages[0]["content"] = fallback_prompt
                analysis = await self.ollama_client.chat_async(
                    model=model_name,
                    messages=messages,
                    system=system_prompt
                )

            risk_level = self._extract_risk_level(analysis)

            return {
                "analysis_type": "receipt",
                "llm_analysis": analysis,
                "risk_level": risk_level,
                "image_quality": self.processor.analyze_image_quality(receipt),
                "extracted_text": self.processor.extract_text_from_image(receipt),
                "amount_match": self._verify_amount_match(extracted_text, amount),
                "model_used": model_name
            }

        except Exception as e:
            logger.error(f"票据分析失败: {e}")
            return {
                "analysis_type": "receipt",
                "error": f"票据分析失败: {str(e)}",
                "risk_level": "ERROR"
            }

    async def comprehensive_analysis(
        self,
        transaction_features: Dict[str, float],
        images: Dict[str, Union[str, bytes]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, any]:
        """
        综合多模态分析

        Args:
            transaction_features: 交易特征数据
            images: 图像字典，如 {'id_card': image_data, 'receipt': image_data}
            model_name: 使用的LLM模型名称

        Returns:
            Dict: 综合分析结果
        """
        try:
            model_name = model_name or self.default_multimodal_model

            # 基础预测分析
            if self.model and self.scaler and self.feature_names:
                explanation_data = explain_prediction(
                    self.model, self.scaler, self.feature_names, transaction_features
                )
                base_explanation = generate_explanation_text(explanation_data)
                risk_score = explanation_data["prediction_score"]
                risk_label = explanation_data["prediction_label"]
            else:
                base_explanation = "模型预测结果不可用"
                risk_score = 0.5
                risk_label = 0

            # 图像分析任务
            image_analysis_tasks = []
            image_results = {}

            if images:
                if 'id_card' in images:
                    task = self.analyze_id_card(
                        transaction_features,
                        images['id_card'],
                        images.get('selfie'),
                        model_name
                    )
                    image_analysis_tasks.append(('id_card', task))

                if 'receipt' in images:
                    task = self.analyze_receipt(
                        transaction_features,
                        images['receipt'],
                        model_name
                    )
                    image_analysis_tasks.append(('receipt', task))

                # 并行执行图像分析
                if image_analysis_tasks:
                    results = await asyncio.gather(
                        *[task for _, task in image_analysis_tasks],
                        return_exceptions=True
                    )

                    for (image_type, _), result in zip(image_analysis_tasks, results):
                        if isinstance(result, Exception):
                            image_results[image_type] = {
                                "error": f"{image_type}分析失败: {str(result)}",
                                "risk_level": "ERROR"
                            }
                        else:
                            image_results[image_type] = result

            # 综合风险评估
            overall_risk = self._calculate_overall_risk(
                risk_score, risk_label, image_results
            )

            # 生成综合解释
            comprehensive_explanation = await self._generate_comprehensive_explanation(
                base_explanation, image_results, overall_risk, model_name
            )

            return {
                "transaction_prediction": {
                    "risk_score": risk_score,
                    "risk_label": risk_label,
                    "base_explanation": base_explanation
                },
                "image_analysis": image_results,
                "overall_risk": overall_risk,
                "comprehensive_explanation": comprehensive_explanation,
                "model_used": model_name,
                "analysis_timestamp": asyncio.get_event_loop().time()
            }

        except Exception as e:
            logger.error(f"综合分析失败: {e}")
            return {
                "error": f"综合分析失败: {str(e)}",
                "overall_risk": {"level": "ERROR", "score": 0.0}
            }

    def _extract_risk_level(self, analysis_text: str) -> str:
        """
        从分析文本中提取风险等级

        Args:
            analysis_text: LLM分析文本

        Returns:
            str: 风险等级 (LOW, MEDIUM, HIGH, CRITICAL)
        """
        text_lower = analysis_text.lower()

        if any(keyword in text_lower for keyword in ['高风险', '高风险等级', 'critical', '严重', '紧急']):
            return "HIGH"
        elif any(keyword in text_lower for keyword in ['中风险', '中等风险', 'moderate', '中等']):
            return "MEDIUM"
        elif any(keyword in text_lower for keyword in ['低风险', '低风险等级', 'low', '较低']):
            return "LOW"
        else:
            return "UNKNOWN"

    def _verify_amount_match(self, ocr_text: str, expected_amount: float) -> Dict[str, any]:
        """
        验证OCR提取的金额与期望金额是否匹配

        Args:
            ocr_text: OCR提取的文本
            expected_amount: 期望金额

        Returns:
            Dict: 金额匹配验证结果
        """
        try:
            # 提取金额数字
            amount_patterns = [
                r'¥(\d+\.?\d*)',
                r'(\d+\.?\d*)元',
                r'(\d+\.?\d*)',
                r'金额[：:]\s*(\d+\.?\d*)'
            ]

            extracted_amounts = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, ocr_text)
                extracted_amounts.extend([float(m) for m in matches])

            if not extracted_amounts:
                return {
                    "match": False,
                    "extracted_amounts": [],
                    "expected_amount": expected_amount,
                    "message": "未能在票据中找到金额信息"
                }

            # 检查是否有匹配的金额
            tolerance = 0.01  # 允许的误差范围
            matched_amounts = [
                amt for amt in extracted_amounts
                if abs(amt - expected_amount) <= tolerance
            ]

            return {
                "match": len(matched_amounts) > 0,
                "extracted_amounts": extracted_amounts,
                "matched_amounts": matched_amounts,
                "expected_amount": expected_amount,
                "tolerance": tolerance
            }

        except Exception as e:
            return {
                "match": False,
                "error": f"金额匹配验证失败: {str(e)}"
            }

    def _calculate_overall_risk(
        self,
        risk_score: float,
        risk_label: int,
        image_results: Dict[str, Dict]
    ) -> Dict[str, any]:
        """
        计算综合风险等级

        Args:
            risk_score: 基础风险分数
            risk_label: 基础风险标签
            image_results: 图像分析结果

        Returns:
            Dict: 综合风险评估
        """
        try:
            # 基础风险权重
            base_risk_weight = 0.6
            image_risk_weight = 0.4

            # 转换图像风险等级为分数
            image_risk_scores = []
            for image_type, result in image_results.items():
                if result.get("risk_level") == "HIGH":
                    image_risk_scores.append(0.8)
                elif result.get("risk_level") == "MEDIUM":
                    image_risk_scores.append(0.6)
                elif result.get("risk_level") == "LOW":
                    image_risk_scores.append(0.2)
                elif result.get("risk_level") == "ERROR":
                    image_risk_scores.append(0.5)  # 错误情况下给予中等风险
                else:
                    image_risk_scores.append(0.3)

            # 计算图像风险平均分
            avg_image_risk = sum(image_risk_scores) / len(image_risk_scores) if image_risk_scores else 0.3

            # 综合风险计算
            overall_score = (risk_score * base_risk_weight + avg_image_risk * image_risk_weight)

            # 确定风险等级
            if overall_score >= 0.8:
                risk_level = "HIGH"
            elif overall_score >= 0.6:
                risk_level = "MEDIUM"
            elif overall_score >= 0.3:
                risk_level = "LOW"
            else:
                risk_level = "VERY_LOW"

            return {
                "level": risk_level,
                "score": round(overall_score, 3),
                "base_risk_score": risk_score,
                "image_risk_score": avg_image_risk,
                "components": {
                    "transaction_features": {
                        "score": risk_score,
                        "weight": base_risk_weight
                    },
                    "image_analysis": {
                        "score": avg_image_risk,
                        "weight": image_risk_weight,
                        "details": {
                            img_type: result.get("risk_level", "UNKNOWN")
                            for img_type, result in image_results.items()
                        }
                    }
                }
            }

        except Exception as e:
            logger.error(f"综合风险计算失败: {e}")
            return {
                "level": "ERROR",
                "score": 0.0,
                "error": str(e)
            }

    async def _generate_comprehensive_explanation(
        self,
        base_explanation: str,
        image_results: Dict[str, Dict],
        overall_risk: Dict,
        model_name: str
    ) -> str:
        """
        生成综合解释报告

        Args:
            base_explanation: 基础模型解释
            image_results: 图像分析结果
            overall_risk: 综合风险评估
            model_name: 使用的模型名称

        Returns:
            str: 综合解释报告
        """
        try:
            system_prompt = """你是一个资深的风控专家，请根据提供的各方面分析结果，生成一份综合性的风险评估报告。

报告要求：
1. 综合分析交易特征和图像分析结果
2. 明确指出主要风险点和风险等级
3. 提供具体的风控建议和处理措施
4. 报告要结构清晰，重点突出
5. 控制在400字以内"""

            # 构建综合分析摘要
            image_summary = ""
            for image_type, result in image_results.items():
                risk_level = result.get("risk_level", "UNKNOWN")
                image_summary += f"- {image_type}分析: {risk_level}风险\n"

            user_prompt = f"""请基于以下分析结果生成综合风险评估报告：

**基础模型分析**：
{base_explanation}

**图像分析摘要**：
{image_summary}

**综合风险评估**：
- 风险等级: {overall_risk.get('level', 'UNKNOWN')}
- 风险分数: {overall_risk.get('score', 0):.3f}
- 基础特征风险: {overall_risk.get('components', {}).get('transaction_features', {}).get('score', 0):.3f}
- 图像分析风险: {overall_risk.get('components', {}).get('image_analysis', {}).get('score', 0):.3f}

请提供一份专业的综合风险评估报告，包括具体的处理建议。"""

            messages = [{"role": "user", "content": user_prompt}]

            # 使用异步方法生成解释
            if self.ollama_client.is_multimodal_model(model_name):
                explanation = await self.ollama_client.chat_async(
                    model=model_name,
                    messages=messages,
                    system=system_prompt
                )
            else:
                # 如果不是多模态模型，使用文本模型
                explanation = await self.ollama_client.chat_async(
                    model=self.fallback_text_model,
                    messages=messages,
                    system=system_prompt
                )

            return explanation

        except Exception as e:
            logger.error(f"综合解释生成失败: {e}")
            return f"综合解释生成失败: {str(e)}\n基础分析: {base_explanation}"