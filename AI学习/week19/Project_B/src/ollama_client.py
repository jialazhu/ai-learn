"""
Ollama API 客户端封装
支持文本和多模态（图像+文本）对话
"""
import json
import requests
import asyncio
import aiohttp
import logging
from typing import Optional, List, Dict, Union


class OllamaClient:
    """Ollama API 客户端，支持文本和多模态对话"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        初始化客户端

        Args:
            base_url: Ollama服务地址
        """
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
        self.logger = logging.getLogger(__name__)

        # 多模态模型列表
        self.multimodal_models = {
            'qwen3-vl:4b', 'qwen2-vl:7b', 'llava:latest',
            'llava-llama3:8b', 'bakllava:latest', 'moondream:latest'
        }
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        system: Optional[str] = None,
        timeout: int = 300
    ) -> str:
        """
        发送聊天请求
        
        Args:
            model: 模型名称
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            stream: 是否流式输出
            system: 系统提示词（可选）
            timeout: 超时时间（秒）
        
        Returns:
            str: 模型回复内容
        """
        payload = {
            "model": model,
            "messages": messages.copy(),
            "stream": stream
        }
        
        if system:
            payload["messages"].insert(0, {"role": "system", "content": system})
        
        resp = requests.post(self.chat_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        
        if stream:
            content = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                delta = data.get("message", {}).get("content", "")
                if delta:
                    content += delta
            return content
        else:
            data = resp.json()
            return data.get("message", {}).get("content", "")
    
    def explain_prediction(
        self,
        model: str,
        explanation_text: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        使用LLM解释预测结果
        
        Args:
            model: 模型名称
            explanation_text: SHAP解释文本
            system_prompt: 系统提示词（可选）
        
        Returns:
            str: LLM生成的解释报告
        """
        if system_prompt is None:
            system_prompt = """你是一个专业的风险控制分析师。请根据提供的模型预测结果和特征重要性分析，生成一份清晰、专业的中文解释报告。

要求：
1. 用通俗易懂的语言解释为什么模型给出这个预测结果
2. 重点说明哪些特征对预测结果影响最大
3. 给出风险提示或建议
4. 报告要简洁明了，控制在200字以内"""
        
        user_prompt = f"""请分析以下模型预测结果，并生成一份解释报告：

{explanation_text}

请用中文生成一份专业的解释报告。"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        return self.chat(model=model, messages=messages, system=system_prompt, stream=False)
    
    def generate_strategy(
        self,
        model: str,
        risk_summary: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        生成风控策略建议
        
        Args:
            model: 模型名称
            risk_summary: 风险摘要
            system_prompt: 系统提示词（可选）
        
        Returns:
            str: 策略建议
        """
        if system_prompt is None:
            system_prompt = """你是一个资深的风控策略专家。请根据提供的风险分析结果，给出可执行的运营和风控策略建议。

要求：
1. 建议要具体、可执行
2. 考虑实际业务场景
3. 给出3-5条建议
4. 用中文输出"""
        
        user_prompt = f"""根据以下风险分析结果，请给出风控策略建议：

{risk_summary}

请给出3-5条可执行的策略建议。"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        return self.chat(model=model, messages=messages, system=system_prompt, stream=False)
    
    def health_check(self) -> bool:
        """
        检查Ollama服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def is_multimodal_model(self, model: str) -> bool:
        """
        检查模型是否支持多模态

        Args:
            model: 模型名称

        Returns:
            bool: 是否支持多模态
        """
        return model in self.multimodal_models

    def chat_multimodal(
        self,
        model: str,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
        stream: bool = False,
        system: Optional[str] = None,
        timeout: int = 300
    ) -> str:
        """
        发送多模态聊天请求（文本+图像）

        Args:
            model: 模型名称（必须是多模态模型）
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            images: Base64编码的图像列表
            stream: 是否流式输出
            system: 系统提示词（可选）
            timeout: 超时时间（秒）

        Returns:
            str: 模型回复内容
        """
        if not self.is_multimodal_model(model):
            raise ValueError(f"模型 {model} 不支持多模态功能。支持的模型: {self.multimodal_models}")

        if not images:
            self.logger.warning("多模态调用未提供图像，将回退到文本模式")
            return self.chat(model, messages, stream, system, timeout)

        payload = {
            "model": model,
            "messages": messages.copy(),
            "stream": stream
        }

        # 添加图像数据到最后一条用户消息
        if images:
            for message in reversed(payload["messages"]):
                if message["role"] == "user":
                    message["images"] = images
                    break

        if system:
            payload["messages"].insert(0, {"role": "system", "content": system})

        try:
            resp = requests.post(self.chat_url, json=payload, timeout=timeout)
            resp.raise_for_status()

            if stream:
                content = ""
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    delta = data.get("message", {}).get("content", "")
                    if delta:
                        content += delta
                return content
            else:
                data = resp.json()
                return data.get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"多模态聊天请求失败: {e}")
            raise

    async def chat_multimodal_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
        stream: bool = False,
        system: Optional[str] = None,
        timeout: int = 300
    ) -> str:
        """
        异步发送多模态聊天请求

        Args:
            model: 模型名称（必须是多模态模型）
            messages: 消息列表
            images: Base64编码的图像列表
            stream: 是否流式输出
            system: 系统提示词（可选）
            timeout: 超时时间（秒）

        Returns:
            str: 模型回复内容
        """
        if not self.is_multimodal_model(model):
            raise ValueError(f"模型 {model} 不支持多模态功能")

        if not images:
            # 如果没有图像，使用异步文本聊天
            return await self.chat_async(model, messages, stream, system, timeout)

        payload = {
            "model": model,
            "messages": messages.copy(),
            "stream": stream
        }

        # 添加图像数据
        if images:
            for message in reversed(payload["messages"]):
                if message["role"] == "user":
                    message["images"] = images
                    break

        if system:
            payload["messages"].insert(0, {"role": "system", "content": system})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.chat_url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    resp.raise_for_status()

                    if stream:
                        content = ""
                        async for line in resp.content:
                            if not line:
                                continue
                            data = json.loads(line.decode())
                            delta = data.get("message", {}).get("content", "")
                            if delta:
                                content += delta
                        return content
                    else:
                        data = await resp.json()
                        return data.get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"异步多模态聊天请求失败: {e}")
            raise

    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        system: Optional[str] = None,
        timeout: int = 300
    ) -> str:
        """
        异步发送文本聊天请求

        Args:
            model: 模型名称
            messages: 消息列表
            stream: 是否流式输出
            system: 系统提示词（可选）
            timeout: 超时时间（秒）

        Returns:
            str: 模型回复内容
        """
        payload = {
            "model": model,
            "messages": messages.copy(),
            "stream": stream
        }

        if system:
            payload["messages"].insert(0, {"role": "system", "content": system})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.chat_url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    resp.raise_for_status()

                    if stream:
                        content = ""
                        async for line in resp.content:
                            if not line:
                                continue
                            data = json.loads(line.decode())
                            delta = data.get("message", {}).get("content", "")
                            if delta:
                                content += delta
                        return content
                    else:
                        data = await resp.json()
                        return data.get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"异步聊天请求失败: {e}")
            raise

    def explain_multimodal_prediction(
        self,
        model: str,
        explanation_text: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        使用多模态LLM解释预测结果（支持图像）

        Args:
            model: 模型名称（推荐多模态模型）
            explanation_text: SHAP解释文本
            images: Base64编码的图像列表（可选）
            system_prompt: 系统提示词（可选）

        Returns:
            str: LLM生成的解释报告
        """
        if system_prompt is None:
            if images:
                system_prompt = """你是一个专业的风险控制分析师和图像识别专家。
请根据提供的模型预测结果、特征重要性分析和相关图像，生成一份清晰、专业的中文解释报告。

要求：
1. 综合分析文本信息和图像信息
2. 用通俗易懂的语言解释为什么模型给出这个预测结果
3. 重点说明哪些特征对预测结果影响最大
4. 分析图像中的关键信息（如身份证、票据等）
5. 给出综合性的风险提示或建议
6. 报告要简洁明了，控制在300字以内"""
            else:
                system_prompt = """你是一个专业的风险控制分析师。请根据提供的模型预测结果和特征重要性分析，生成一份清晰、专业的中文解释报告。

要求：
1. 用通俗易懂的语言解释为什么模型给出这个预测结果
2. 重点说明哪些特征对预测结果影响最大
3. 给出风险提示或建议
4. 报告要简洁明了，控制在200字以内"""

        user_prompt = f"""请分析以下模型预测结果，并生成一份解释报告：

{explanation_text}

请用中文生成一份专业的解释报告。"""

        messages = [{"role": "user", "content": user_prompt}]

        # 根据是否有图像选择调用方式
        if images:
            return self.chat_multimodal(
                model=model,
                messages=messages,
                images=images,
                system=system_prompt,
                stream=False
            )
        else:
            return self.chat(
                model=model,
                messages=messages,
                system=system_prompt,
                stream=False
            )

    async def explain_multimodal_prediction_async(
        self,
        model: str,
        explanation_text: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        异步使用多模态LLM解释预测结果

        Args:
            model: 模型名称
            explanation_text: SHAP解释文本
            images: Base64编码的图像列表（可选）
            system_prompt: 系统提示词（可选）

        Returns:
            str: LLM生成的解释报告
        """
        if system_prompt is None:
            if images:
                system_prompt = """你是一个专业的风险控制分析师和图像识别专家。
请根据提供的模型预测结果、特征重要性分析和相关图像，生成一份清晰、专业的中文解释报告。

要求：
1. 综合分析文本信息和图像信息
2. 用通俗易懂的语言解释为什么模型给出这个预测结果
3. 重点说明哪些特征对预测结果影响最大
4. 分析图像中的关键信息（如身份证、票据等）
5. 给出综合性的风险提示或建议
6. 报告要简洁明了，控制在300字以内"""
            else:
                system_prompt = """你是一个专业的风险控制分析师。请根据提供的模型预测结果和特征重要性分析，生成一份清晰、专业的中文解释报告。

要求：
1. 用通俗易懂的语言解释为什么模型给出这个预测结果
2. 重点说明哪些特征对预测结果影响最大
3. 给出风险提示或建议
4. 报告要简洁明了，控制在200字以内"""

        user_prompt = f"""请分析以下模型预测结果，并生成一份解释报告：

{explanation_text}

请用中文生成一份专业的解释报告。"""

        messages = [{"role": "user", "content": user_prompt}]

        # 根据是否有图像选择调用方式
        if images:
            return await self.chat_multimodal_async(
                model=model,
                messages=messages,
                images=images,
                system=system_prompt,
                stream=False
            )
        else:
            return await self.chat_async(
                model=model,
                messages=messages,
                system=system_prompt,
                stream=False
            )

    def list_available_models(self) -> Dict[str, List[str]]:
        """
        获取可用模型列表

        Returns:
            Dict: 包含文本模型和多模态模型的字典
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()

            data = resp.json()
            all_models = [model["name"] for model in data.get("models", [])]

            # 分类模型
            text_models = []
            multimodal_models = []

            for model_name in all_models:
                if model_name in self.multimodal_models:
                    multimodal_models.append(model_name)
                else:
                    text_models.append(model_name)

            return {
                "all_models": all_models,
                "text_models": text_models,
                "multimodal_models": multimodal_models
            }
        except Exception as e:
            self.logger.error(f"获取模型列表失败: {e}")
            return {
                "all_models": [],
                "text_models": [],
                "multimodal_models": [],
                "error": str(e)
            }


if __name__ == "__main__":
    # 测试
    client = OllamaClient()
    
    # 健康检查
    if client.health_check():
        print("✓ Ollama服务可用")
    else:
        print("✗ Ollama服务不可用，请检查服务是否启动")
        exit(1)
    
    # 测试对话
    response = client.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    )
    print("\n模型回复：", response)

