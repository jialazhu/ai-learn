"""
多模态数据处理模块
支持图像预处理、Base64编码、OCR文本提取、图像质量分析等功能
"""
import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class MultimodalProcessor:
    """多模态数据处理器"""

    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图像文件编码为Base64字符串

        Args:
            image_path: 图像文件路径

        Returns:
            str: Base64编码的图像字符串
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            raise ValueError(f"无法编码图像文件 {image_path}: {str(e)}")

    def encode_pil_image_to_base64(self, image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """
        将PIL图像对象编码为Base64字符串

        Args:
            image: PIL图像对象
            format: 图像格式 (JPEG, PNG等)
            quality: 图像质量 (1-100)

        Returns:
            str: Base64编码的图像字符串
        """
        try:
            buffer = io.BytesIO()
            # 根据格式选择保存参数
            save_kwargs = {}
            if format.upper() == "JPEG":
                save_kwargs = {"format": "JPEG", "quality": quality, "optimize": True}
            elif format.upper() == "PNG":
                save_kwargs = {"format": "PNG", "optimize": True}
            else:
                save_kwargs = {"format": format}

            image.save(buffer, **save_kwargs)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"PIL图像编码失败: {e}")
            raise ValueError(f"PIL图像编码失败: {str(e)}")

    def load_image_from_path(self, image_path: str) -> Image.Image:
        """
        从文件路径加载图像

        Args:
            image_path: 图像文件路径

        Returns:
            PIL.Image.Image: 加载的图像对象
        """
        try:
            image = Image.open(image_path)
            # 转换为RGB模式（处理RGBA等格式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"图像加载失败: {e}")
            raise ValueError(f"无法加载图像文件 {image_path}: {str(e)}")

    def load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        从字节数据加载图像

        Args:
            image_bytes: 图像字节数据

        Returns:
            PIL.Image.Image: 加载的图像对象
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"从字节数据加载图像失败: {e}")
            raise ValueError(f"无法从字节数据加载图像: {str(e)}")

    def resize_image(
        self,
        image: Union[str, Image.Image],
        max_size: Tuple[int, int] = (1024, 1024),
        maintain_ratio: bool = True
    ) -> Image.Image:
        """
        调整图像尺寸

        Args:
            image: 图像路径或PIL图像对象
            max_size: 最大尺寸 (width, height)
            maintain_ratio: 是否保持宽高比

        Returns:
            PIL.Image.Image: 调整后的图像对象
        """
        try:
            if isinstance(image, str):
                image = self.load_image_from_path(image)

            if maintain_ratio:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(max_size, Image.Resampling.LANCZOS)

            return image
        except Exception as e:
            logger.error(f"图像尺寸调整失败: {e}")
            raise ValueError(f"图像尺寸调整失败: {str(e)}")

    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """
        增强图像质量

        Args:
            image: PIL图像对象

        Returns:
            PIL.Image.Image: 增强后的图像对象
        """
        try:
            # 锐化
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)

            # 对比度增强
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)

            # 亮度调整
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.05)

            return image
        except Exception as e:
            logger.error(f"图像质量增强失败: {e}")
            # 如果增强失败，返回原图像
            return image

    def extract_text_from_image(self, image: Union[str, Image.Image], language: str = 'chi_sim+eng') -> str:
        """
        从图像中提取文本（OCR功能）

        Args:
            image: 图像路径或PIL图像对象
            language: OCR语言设置

        Returns:
            str: 提取的文本内容
        """
        try:
            import pytesseract

            if isinstance(image, str):
                image = self.load_image_from_path(image)

            # 预处理图像以提高OCR准确性
            # 转换为灰度图
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image

            # 增强对比度
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer.enhance(2.0)

            # OCR识别
            text = pytesseract.image_to_string(enhanced_image, lang=language)

            return text.strip()

        except ImportError:
            logger.warning("pytesseract未安装，OCR功能不可用")
            return "OCR功能需要安装pytesseract和tesseract-ocr"
        except Exception as e:
            logger.error(f"OCR文本提取失败: {e}")
            return f"OCR文本提取失败: {str(e)}"

    def analyze_image_quality(self, image: Union[str, Image.Image]) -> Dict[str, any]:
        """
        分析图像质量指标

        Args:
            image: 图像路径或PIL图像对象

        Returns:
            Dict: 图像质量分析结果
        """
        try:
            if isinstance(image, str):
                image = self.load_image_from_path(image)

            # 转换为OpenCV格式
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            analysis = {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.width / image.height,
                "mode": image.mode,
                "format": image.format if image.format else "Unknown",
                "file_size": len(self.encode_pil_image_to_base64(image)) * 3 / 4,  # 估算文件大小
                "blur_detected": False,
                "noise_level": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "sharpness": 0.0,
                "quality_score": "unknown"
            }

            # 转换为灰度图进行分析
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 1. 模糊度检测 (Laplacian方差)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            analysis["blur_detected"] = laplacian_var < 100
            analysis["sharpness"] = float(laplacian_var)

            # 2. 噪声水平检测
            analysis["noise_level"] = float(np.std(gray))

            # 3. 亮度分析
            analysis["brightness"] = float(np.mean(gray))

            # 4. 对比度分析 (标准差)
            analysis["contrast"] = float(np.std(gray))

            # 5. 综合质量评分
            quality_score = 0
            # 清晰度评分 (0-30分)
            if laplacian_var > 500:
                quality_score += 30
            elif laplacian_var > 200:
                quality_score += 20
            elif laplacian_var > 100:
                quality_score += 10

            # 亮度评分 (0-25分)
            mean_brightness = np.mean(gray)
            if 50 <= mean_brightness <= 200:
                quality_score += 25
            elif 30 <= mean_brightness <= 220:
                quality_score += 15
            else:
                quality_score += 5

            # 对比度评分 (0-25分)
            std_dev = np.std(gray)
            if std_dev > 60:
                quality_score += 25
            elif std_dev > 40:
                quality_score += 20
            elif std_dev > 20:
                quality_score += 15
            elif std_dev > 10:
                quality_score += 10

            # 噪声评分 (0-20分)
            noise_level = np.std(gray)
            if noise_level < 30:
                quality_score += 20
            elif noise_level < 50:
                quality_score += 15
            elif noise_level < 70:
                quality_score += 10

            analysis["quality_score"] = quality_score

            # 质量等级
            if quality_score >= 80:
                analysis["quality_grade"] = "优秀"
            elif quality_score >= 60:
                analysis["quality_grade"] = "良好"
            elif quality_score >= 40:
                analysis["quality_grade"] = "一般"
            else:
                analysis["quality_grade"] = "较差"

            return analysis

        except Exception as e:
            logger.error(f"图像质量分析失败: {e}")
            return {
                "error": f"图像质量分析失败: {str(e)}",
                "quality_score": 0,
                "quality_grade": "分析失败"
            }

    def detect_faces(self, image: Union[str, Image.Image]) -> List[Dict[str, any]]:
        """
        检测图像中的人脸

        Args:
            image: 图像路径或PIL图像对象

        Returns:
            List[Dict]: 检测到的人脸信息列表
        """
        try:
            import cv2

            if isinstance(image, str):
                cv_image = cv2.imread(image)
            else:
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # 转换为灰度图
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 加载人脸检测器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face_results = []
            for i, (x, y, w, h) in enumerate(faces):
                face_info = {
                    "face_id": i + 1,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "confidence": 1.0,  # Haar检测器不直接提供置信度
                    "center_x": int(x + w/2),
                    "center_y": int(y + h/2)
                }
                face_results.append(face_info)

            return {
                "faces_detected": len(face_results),
                "faces": face_results,
                "image_size": (cv_image.shape[1], cv_image.shape[0])
            }

        except ImportError:
            logger.warning("OpenCV未安装，人脸检测功能不可用")
            return {"faces_detected": 0, "faces": [], "error": "需要安装OpenCV"}
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return {"faces_detected": 0, "faces": [], "error": f"人脸检测失败: {str(e)}"}

    def validate_image_format(self, filename: str) -> bool:
        """
        验证图像文件格式是否支持

        Args:
            filename: 文件名

        Returns:
            bool: 是否支持该格式
        """
        return any(filename.lower().endswith(ext) for ext in self.supported_formats)

    def get_image_info(self, image: Union[str, Image.Image]) -> Dict[str, any]:
        """
        获取图像基本信息

        Args:
            image: 图像路径或PIL图像对象

        Returns:
            Dict: 图像基本信息
        """
        try:
            if isinstance(image, str):
                image = self.load_image_from_path(image)

            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "size_bytes": len(self.encode_pil_image_to_base64(image)) * 3 / 4,
                "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
        except Exception as e:
            return {"error": f"获取图像信息失败: {str(e)}"}


# 创建全局实例
multimodal_processor = MultimodalProcessor()