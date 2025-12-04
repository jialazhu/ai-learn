"""
多模态数据分析师Agent - API服务器
提供图片上传、CSV分析和报告生成接口
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入Agent
from agents.data_analyst_agent import DataAnalystAgent
from core.controller import Controller

# 导入工具模块
from utils.file_processor import FileProcessor
from utils.report_exporter import ReportExporter
from utils.chart_generator import ChartGenerator
from utils.data_quality import DataQualityAnalyzer
from utils.json_utils import convert_to_serializable, safe_json_dump
import json
from datetime import datetime
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
controller = None
data_analyst_agent = None
report_exporter = None

# 创建目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

EXPORTS_DIR = Path("exports")
EXPORTS_DIR.mkdir(exist_ok=True)

def initialize_system():
    """初始化系统"""
    global controller, data_analyst_agent
    
    try:
        # 创建控制器
        controller = Controller()
        
        # 创建数据分析师Agent
        llm_model = os.environ.get("LLM_MODEL") or os.environ.get("QWEN_MODEL") or "gpt-4-vision-preview"
        data_analyst_agent = DataAnalystAgent(
            name="DataAnalyst",
            controller_reference=controller,
            llm_model=llm_model,
            temperature=0.0
        )
        
        # 初始化报告导出器
        global report_exporter
        report_exporter = ReportExporter(export_dir=str(EXPORTS_DIR))
        
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    initialize_system()
    yield
    # 关闭时清理（如果需要）
    logger.info("Application shutdown")

# 创建FastAPI应用
app = FastAPI(
    title="多模态数据分析师Agent API", 
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回Web界面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>多模态数据分析师Agent</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 40px;
                font-size: 1.1em;
            }
            .upload-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }
            .upload-box {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                background: #f8f9fa;
                transition: all 0.3s;
            }
            .upload-box:hover {
                border-color: #764ba2;
                background: #f0f0f0;
            }
            .upload-box h3 {
                color: #667eea;
                margin-bottom: 15px;
            }
            .file-input {
                margin: 15px 0;
            }
            .file-input input[type="file"] {
                padding: 10px;
                border: 2px solid #667eea;
                border-radius: 8px;
                width: 100%;
                cursor: pointer;
            }
            .preview {
                margin-top: 15px;
                max-width: 100%;
                max-height: 300px;
                border-radius: 8px;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 1.1em;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s;
                width: 100%;
                margin-top: 20px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            }
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .result-section {
                margin-top: 40px;
                padding: 30px;
                background: #f8f9fa;
                border-radius: 15px;
                display: none;
            }
            .result-section.show {
                display: block;
            }
            .result-section h2 {
                color: #333;
                margin-bottom: 20px;
            }
            .result-content {
                background: white;
                padding: 20px;
                border-radius: 10px;
                white-space: pre-wrap;
                line-height: 1.6;
                max-height: 600px;
                overflow-y: auto;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                color: #d32f2f;
                background: #ffebee;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                display: none;
            }
            .error.show {
                display: block;
            }
            .requirements-box {
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 15px;
                border: 2px solid #e0e0e0;
            }
            .requirements-box label {
                display: block;
                color: #333;
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            .requirements-box textarea {
                width: 100%;
                min-height: 120px;
                padding: 15px;
                border: 2px solid #667eea;
                border-radius: 10px;
                font-size: 1em;
                font-family: inherit;
                resize: vertical;
            }
            .requirements-box textarea:focus {
                outline: none;
                border-color: #764ba2;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
            }
            .file-info {
                margin-top: 10px;
                padding: 10px;
                background: white;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .file-info span {
                color: #666;
                font-size: 0.9em;
            }
            .remove-btn {
                background: #ff5252;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.85em;
                transition: all 0.3s;
            }
            .remove-btn:hover {
                background: #d32f2f;
                transform: scale(1.05);
            }
            .file-input-wrapper {
                position: relative;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 多模态数据分析师Agent</h1>
            <p class="subtitle">上传图表图片和数据文件（CSV/Excel/JSON），生成专业数据分析报告</p>
            <div style="text-align: center; margin-bottom: 20px; padding: 15px; background: #e8f4f8; border-radius: 10px;">
                <strong>✨ 新功能：</strong> 支持Excel/JSON格式 | 数据质量评估 | 自动生成图表 | 报告导出 | 历史记录
            </div>
            
            <form id="analysisForm">
                <div class="upload-section">
                    <div class="upload-box">
                        <h3>📈 上传图表图片</h3>
                        <div class="file-input-wrapper">
                            <div class="file-input">
                                <input type="file" id="imageInput" accept="image/*">
                            </div>
                            <div id="imageFileInfo" class="file-info" style="display: none;">
                                <span id="imageFileName"></span>
                                <button type="button" class="remove-btn" onclick="removeImage()">删除</button>
                            </div>
                        </div>
                        <img id="imagePreview" class="preview" style="display: none;">
                    </div>
                    
                    <div class="upload-box">
                        <h3>📄 上传数据文件</h3>
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                            支持格式：CSV, Excel (.xlsx, .xls), JSON
                        </div>
                        <div class="file-input-wrapper">
                            <div class="file-input">
                                <input type="file" id="csvInput" accept=".csv,.xlsx,.xls,.json">
                            </div>
                            <div id="csvFileInfo" class="file-info" style="display: none;">
                                <span id="csvFileName"></span>
                                <button type="button" class="remove-btn" onclick="removeCsv()">删除</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="requirements-box">
                    <label for="requirementsInput">💡 分析要求（可选）</label>
                    <textarea 
                        id="requirementsInput" 
                        placeholder="请输入您的分析要求，例如：&#10;- 重点关注销售额趋势&#10;- 分析异常值原因&#10;- 提供改进建议&#10;- 对比不同时间段的数据变化等..."
                    ></textarea>
                </div>
                
                <div style="margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 10px;">
                    <label style="display: flex; align-items: center; cursor: pointer;">
                        <input type="checkbox" id="generateCharts" style="margin-right: 10px; width: 20px; height: 20px;">
                        <span>📊 自动生成数据可视化图表</span>
                    </label>
                </div>
                
                <button type="submit" class="btn" id="analyzeBtn">
                    🚀 开始分析
                </button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px;">正在分析数据，请稍候...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result-section" id="resultSection">
                <h2>📋 分析报告</h2>
                <div class="result-content" id="resultContent"></div>
            </div>
        </div>
        
        <script>
            const imageInput = document.getElementById('imageInput');
            const csvInput = document.getElementById('csvInput');
            const imagePreview = document.getElementById('imagePreview');
            const csvPreview = document.getElementById('csvPreview');
            const form = document.getElementById('analysisForm');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const resultSection = document.getElementById('resultSection');
            const resultContent = document.getElementById('resultContent');
            const errorDiv = document.getElementById('error');
            
            // 图片预览和文件信息
            imageInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                    
                    // 显示文件信息
                    document.getElementById('imageFileName').textContent = `${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
                    document.getElementById('imageFileInfo').style.display = 'flex';
                }
            });
            
            // CSV文件信息
            csvInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    document.getElementById('csvFileName').textContent = `${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
                    document.getElementById('csvFileInfo').style.display = 'flex';
                }
            });
            
            // 删除图片
            function removeImage() {
                imageInput.value = '';
                imagePreview.src = '';
                imagePreview.style.display = 'none';
                document.getElementById('imageFileInfo').style.display = 'none';
            }
            
            // 删除CSV
            function removeCsv() {
                csvInput.value = '';
                document.getElementById('csvFileInfo').style.display = 'none';
            }
            
            // 表单提交
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const imageFile = imageInput.files[0];
                const csvFile = csvInput.files[0];
                
                if (!imageFile && !csvFile) {
                    showError('请至少上传一个图片或数据文件');
                    return;
                }
                
                // 显示加载状态
                analyzeBtn.disabled = true;
                loading.classList.add('show');
                resultSection.classList.remove('show');
                errorDiv.classList.remove('show');
                
                try {
                    const formData = new FormData();
                    if (imageFile) {
                        formData.append('image', imageFile);
                    }
                    if (csvFile) {
                        formData.append('csv', csvFile);
                    }
                    
                    // 添加用户要求
                    const requirements = document.getElementById('requirementsInput').value.trim();
                    if (requirements) {
                        formData.append('requirements', requirements);
                    }
                    
                    // 添加生成图表选项
                    const generateCharts = document.getElementById('generateCharts').checked;
                    if (generateCharts) {
                        formData.append('generate_charts', 'true');
                    }
                    
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        try {
                            displayResult(result);
                            resultSection.classList.add('show');
                        } catch (displayError) {
                            console.error('显示结果时出错:', displayError);
                            showError('显示结果时出错: ' + displayError.message);
                        }
                    } else {
                        showError(result.detail || result.message || '分析失败，请重试');
                    }
                } catch (error) {
                    console.error('请求错误:', error);
                    showError('网络错误: ' + (error.message || '未知错误'));
                } finally {
                    analyzeBtn.disabled = false;
                    loading.classList.remove('show');
                }
            });
            
            function displayResult(result) {
                // 安全检查
                if (!result || typeof result !== 'object') {
                    showError('返回的数据格式不正确');
                    return;
                }
                
                let html = '<div style="line-height: 1.8;">';
                
                // 显示主报告
                if (result.report) {
                    html += '<div style="margin-bottom: 30px;"><h3 style="color: #667eea; margin-bottom: 15px;">📋 分析报告</h3>';
                    html += '<div style="white-space: pre-wrap; background: #f8f9fa; padding: 20px; border-radius: 10px;">' + String(result.report || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div></div>';
                }
                
                // 显示图表分析（如果有）
                if (result.chart_analysis) {
                    html += '<div style="margin-bottom: 30px;"><h3 style="color: #667eea; margin-bottom: 15px;">📈 图表分析</h3>';
                    html += '<div style="white-space: pre-wrap; background: #f8f9fa; padding: 20px; border-radius: 10px;">' + String(result.chart_analysis || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div></div>';
                }
                
                // 显示CSV分析（如果有）
                if (result.csv_analysis) {
                    html += '<div style="margin-bottom: 30px;"><h3 style="color: #667eea; margin-bottom: 15px;">📊 数据分析</h3>';
                    html += '<div style="white-space: pre-wrap; background: #f8f9fa; padding: 20px; border-radius: 10px;">' + String(result.csv_analysis || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div></div>';
                }
                
                // 如果没有报告但有分析结果，显示分析结果
                if (!result.report && !result.chart_analysis && !result.csv_analysis && result.analysis) {
                    html += '<div style="margin-bottom: 30px;"><h3 style="color: #667eea; margin-bottom: 15px;">📋 分析结果</h3>';
                    html += '<div style="white-space: pre-wrap; background: #f8f9fa; padding: 20px; border-radius: 10px;">' + String(result.analysis || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div></div>';
                }
                
                // 显示数据质量评估
                if (result.quality_report && typeof result.quality_report === 'object') {
                    const quality = result.quality_report;
                    html += '<div style="margin-bottom: 30px;"><h3 style="color: #667eea; margin-bottom: 15px;">📊 数据质量评估</h3>';
                    html += '<div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">';
                    
                    // 安全地获取总体质量分数
                    const overallScore = (quality.overall_score !== undefined && quality.overall_score !== null) 
                        ? parseFloat(quality.overall_score) 
                        : null;
                    
                    if (overallScore !== null && !isNaN(overallScore)) {
                        const scoreColor = overallScore >= 80 ? '#28a745' : overallScore >= 60 ? '#ffc107' : '#dc3545';
                        html += `<p><strong>总体质量分数：</strong> <span style="font-size: 1.5em; color: ${scoreColor}">${overallScore.toFixed(1)}</span>/100</p>`;
                    } else {
                        html += '<p><strong>总体质量分数：</strong> <span style="font-size: 1.5em; color: #666">N/A</span></p>';
                    }
                    
                    if (quality.dimensions && typeof quality.dimensions === 'object') {
                        html += '<ul style="margin-top: 15px;">';
                        for (const [key, value] of Object.entries(quality.dimensions)) {
                            if (value && typeof value === 'object') {
                                const score = (value.score !== undefined && value.score !== null) 
                                    ? parseFloat(value.score) 
                                    : null;
                                const scoreText = (score !== null && !isNaN(score)) ? score.toFixed(1) : 'N/A';
                                const status = value.status || 'N/A';
                                html += `<li><strong>${key}:</strong> ${scoreText} - ${status}</li>`;
                            }
                        }
                        html += '</ul>';
                    }
                    html += '</div></div>';
                }
                
                // 显示异常值检测
                if (result.anomalies && typeof result.anomalies === 'object') {
                    const totalAnomalies = result.anomalies.total_anomalies;
                    if (totalAnomalies !== undefined && totalAnomalies !== null && totalAnomalies > 0) {
                        html += '<div style="margin-bottom: 30px;"><h3 style="color: #dc3545; margin-bottom: 15px;">⚠️ 异常值检测</h3>';
                        html += '<div style="background: #fff3cd; padding: 20px; border-radius: 10px;">';
                        html += `<p>检测到 <strong>${totalAnomalies}</strong> 个异常值</p>`;
                        if (result.anomalies.columns_with_anomalies && Array.isArray(result.anomalies.columns_with_anomalies)) {
                            html += '<p>涉及列：' + result.anomalies.columns_with_anomalies.join(', ') + '</p>';
                        }
                        html += '</div></div>';
                    }
                }
                
                // 显示生成的图表
                if (result.chart_urls && typeof result.chart_urls === 'object') {
                    html += '<div style="margin-bottom: 30px;"><h3 style="color: #667eea; margin-bottom: 15px;">📈 生成的可视化图表</h3>';
                    html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">';
                    for (const [name, url] of Object.entries(result.chart_urls)) {
                        if (url) {
                            html += `<div style="text-align: center;"><img src="${url}" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="${name || 'chart'}" onerror="this.style.display='none'"></div>`;
                        }
                    }
                    html += '</div></div>';
                }
                
                // 显示导出选项
                if (result.report_id) {
                    html += '<div style="margin-top: 30px; padding: 20px; background: #e8f4f8; border-radius: 10px;">';
                    html += '<h3 style="color: #667eea; margin-bottom: 15px;">💾 导出报告</h3>';
                    html += '<div style="display: flex; gap: 10px; flex-wrap: wrap;">';
                    html += `<button onclick="exportReport('${result.report_id}', 'markdown', this)" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">导出为 Markdown</button>`;
                    html += `<button onclick="exportReport('${result.report_id}', 'html', this)" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">导出为 HTML</button>`;
                    html += `<button onclick="exportReport('${result.report_id}', 'json', this)" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">导出为 JSON</button>`;
                    html += '</div></div>';
                }
                
                html += '</div>';
                resultContent.innerHTML = html;
            }
            
            function exportReport(reportId, format, buttonElement) {
                if (!reportId) {
                    showError('报告ID不存在，无法导出');
                    return;
                }
                
                // 显示导出提示
                const exportBtn = buttonElement || document.activeElement;
                const originalText = exportBtn ? exportBtn.textContent : '';
                if (exportBtn) {
                    exportBtn.disabled = true;
                    exportBtn.textContent = '导出中...';
                }
                
                // 创建下载链接
                const url = `/export/${reportId}?format=${format}`;
                const link = document.createElement('a');
                link.href = url;
                link.download = `report_${reportId}.${format}`;
                link.style.display = 'none';
                document.body.appendChild(link);
                
                // 尝试下载
                fetch(url)
                    .then(response => {
                        if (!response.ok) {
                            return response.json().then(err => {
                                throw new Error(err.detail || '导出失败');
                            });
                        }
                        return response.blob();
                    })
                    .then(blob => {
                        // 创建下载链接
                        const downloadUrl = window.URL.createObjectURL(blob);
                        link.href = downloadUrl;
                        link.click();
                        window.URL.revokeObjectURL(downloadUrl);
                        document.body.removeChild(link);
                        
                        // 恢复按钮
                        if (exportBtn) {
                            exportBtn.disabled = false;
                            exportBtn.textContent = originalText;
                        }
                    })
                    .catch(error => {
                        console.error('导出错误:', error);
                        showError('导出失败: ' + error.message);
                        if (exportBtn) {
                            exportBtn.disabled = false;
                            exportBtn.textContent = originalText;
                        }
                        if (link.parentNode) {
                            document.body.removeChild(link);
                        }
                    });
            }
            
            function showError(message) {
                errorDiv.textContent = message;
                errorDiv.classList.add('show');
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/analyze")
async def analyze_data(
    image: Optional[UploadFile] = File(None),
    csv: Optional[UploadFile] = File(None),
    requirements: Optional[str] = Form(None),
    generate_charts: Optional[bool] = Form(False)
):
    """分析数据并生成报告（支持多种文件格式）"""
    try:
        if not data_analyst_agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        image_base64 = None
        file_data = None
        filename = None
        
        # 处理图片
        if image:
            try:
                image_bytes = await image.read()
                if len(image_bytes) == 0:
                    logger.warning("Uploaded image file is empty")
                else:
                    import base64
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    # 检测图片格式
                    if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                        image_format = "PNG"
                    elif image_bytes.startswith(b'\xff\xd8\xff'):
                        image_format = "JPEG"
                    else:
                        image_format = "Unknown"
                    logger.info(f"Image uploaded: {image.filename}, format: {image_format}, size: {len(image_bytes)} bytes")
            except Exception as e:
                logger.error(f"Error reading image file: {str(e)}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"读取图片文件失败: {str(e)}")
        
        # 处理数据文件（支持CSV, Excel, JSON）
        if csv:
            file_bytes = await csv.read()
            filename = csv.filename
            file_data = file_bytes  # 保持bytes格式，由FileProcessor处理
            logger.info(f"Received data file: {filename}, size: {len(file_bytes)} bytes")
        else:
            logger.info("No data file provided")
        
        # 先处理数据文件（如果有），获取分析结果
        csv_analysis_result = None
        if file_data and filename:
            task_info = {
                "metadata": {
                    "task_type": "csv_analysis",
                    "file_data": file_data,
                    "filename": filename
                }
            }
            try:
                logger.info(f"Processing data file: {filename}, size: {len(file_data)} bytes")
                csv_result = data_analyst_agent._handle_csv_analysis_task(task_info)
                csv_analysis_result = csv_result
                
                # 检查是否有错误
                if "error" in csv_result:
                    error_msg = csv_result['error']
                    logger.error(f"Data file analysis error: {error_msg}")
                    # 如果错误严重，直接返回错误
                    if "需要安装" in error_msg or "无法读取" in error_msg:
                        raise HTTPException(status_code=400, detail=f"处理数据文件失败: {error_msg}")
                else:
                    analysis_len = len(csv_result.get('analysis', ''))
                    logger.info(f"Data file processed successfully: {filename}, analysis length: {analysis_len}")
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"处理数据文件时出错: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise HTTPException(status_code=500, detail=error_msg)
        
        # 生成分析报告（传入CSV分析结果，如果有）
        # 如果已经有CSV分析结果，将其传递给generate_report
        csv_analysis_text = None
        if csv_analysis_result and "analysis" in csv_analysis_result:
            csv_analysis_text = csv_analysis_result.get("analysis", "")
        
        result = data_analyst_agent.generate_report(
            image_base64=image_base64,
            csv_data=csv_analysis_text,  # 传入CSV分析结果文本
            user_requirements=requirements
        )
        
        # 合并CSV分析结果到最终结果
        if csv_analysis_result:
            if "error" in csv_analysis_result:
                logger.warning(f"CSV analysis had errors: {csv_analysis_result['error']}")
                # 即使有错误，也尝试获取质量报告（如果存在）
                if "quality_report" in csv_analysis_result:
                    result["quality_report"] = csv_analysis_result.get("quality_report", {})
                if "anomalies" in csv_analysis_result:
                    result["anomalies"] = csv_analysis_result.get("anomalies", {})
            else:
                # 合并结果
                result["csv_analysis"] = csv_analysis_result.get("analysis", "")
                result["quality_report"] = csv_analysis_result.get("quality_report", {})
                result["anomalies"] = csv_analysis_result.get("anomalies", {})
                result["data_summary"] = csv_analysis_result.get("data_summary", {})
                
                # 记录质量报告信息用于调试
                if result.get("quality_report"):
                    logger.info(f"Quality report generated: overall_score={result['quality_report'].get('overall_score', 'N/A')}")
                else:
                    logger.warning("Quality report is empty or missing")
            
            # 如果需要生成图表，需要从file_info获取DataFrame
            if generate_charts:
                try:
                    file_info = FileProcessor.read_file(file_data=file_data, filename=filename)
                    if file_info.get("data") is not None:
                        df = file_info["data"]
                        charts = ChartGenerator.generate_summary_charts(df, str(CHARTS_DIR))
                        result["generated_charts"] = list(charts.keys())
                        # 将图表路径转换为可访问的URL
                        result["chart_urls"] = {k: f"/charts/{Path(v).name}" for k, v in charts.items()}
                        logger.info(f"Generated {len(charts)} charts")
                except Exception as e:
                    logger.warning(f"Failed to generate charts: {str(e)}")
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 保存报告到历史记录
        try:
            report_id = save_report_history(result)
            result["report_id"] = report_id
        except Exception as e:
            logger.warning(f"Failed to save report history: {str(e)}")
        
        # 确保返回的数据可以JSON序列化
        serializable_result = convert_to_serializable(result)
        return JSONResponse(content=serializable_result)
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

def save_report_history(report_data: Dict[str, Any]) -> str:
    """保存报告到历史记录"""
    report_id = str(uuid.uuid4())
    report_file = REPORTS_DIR / f"{report_id}.json"
    
    history_entry = {
        "report_id": report_id,
        "created_at": datetime.now().isoformat(),
        "report": convert_to_serializable(report_data)
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        safe_json_dump(history_entry, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Report saved: {report_id}")
    return report_id

@app.get("/reports")
async def list_reports():
    """获取历史报告列表"""
    try:
        reports = []
        for report_file in REPORTS_DIR.glob("*.json"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    reports.append({
                        "report_id": data.get("report_id"),
                        "created_at": data.get("created_at"),
                        "preview": data.get("report", {}).get("report", "")[:200] + "..." if data.get("report", {}).get("report") else ""
                    })
            except Exception as e:
                logger.warning(f"Error reading report {report_file}: {str(e)}")
        
        # 按时间排序
        reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return {"reports": reports, "count": len(reports)}
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报告列表失败: {str(e)}")

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """获取指定报告"""
    try:
        report_file = REPORTS_DIR / f"{report_id}.json"
        if not report_file.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报告失败: {str(e)}")

@app.get("/export/{report_id}")
async def export_report(report_id: str, format: str = "markdown"):
    """导出报告为指定格式"""
    try:
        if not report_exporter:
            raise HTTPException(status_code=503, detail="Report exporter not initialized")
        
        report_file = REPORTS_DIR / f"{report_id}.json"
        if not report_file.exists():
            raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
        
        with open(report_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        report_data = data.get("report", {})
        
        if not report_data:
            raise HTTPException(status_code=400, detail="Report data is empty")
        
        # 根据格式导出
        if format == "markdown":
            filepath = report_exporter.export_markdown(report_data, f"report_{report_id}.md")
            media_type = "text/markdown"
        elif format == "json":
            filepath = report_exporter.export_json(report_data, f"report_{report_id}.json")
            media_type = "application/json"
        elif format == "html":
            filepath = report_exporter.export_html(report_data, f"report_{report_id}.html")
            media_type = "text/html"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Supported: markdown, json, html")
        
        # 检查文件是否成功创建
        if not Path(filepath).exists():
            raise HTTPException(status_code=500, detail="Failed to create export file")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            filepath,
            media_type=media_type,
            filename=Path(filepath).name,
            headers={"Content-Disposition": f"attachment; filename={Path(filepath).name}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"导出报告失败: {str(e)}")

@app.get("/charts/{filename}")
async def get_chart(filename: str):
    """获取生成的图表"""
    try:
        chart_path = CHARTS_DIR / filename
        if not chart_path.exists():
            raise HTTPException(status_code=404, detail="Chart not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(chart_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取图表失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "agent_initialized": data_analyst_agent is not None,
        "features": {
            "multi_format_support": True,
            "report_export": True,
            "history": True,
            "data_quality": True,
            "chart_generation": True
        }
    }

def main():
    """启动服务器"""
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    
    # 检查端口是否被占用
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
    except OSError:
        print(f"\n[错误] 端口 {port} 已被占用！")
        print(f"请执行以下操作之一：")
        print(f"1. 关闭占用端口 {port} 的其他程序")
        print(f"2. 修改 .env 文件中的 API_PORT 为其他端口（如 8001）")
        print(f"\n查找占用端口的进程：")
        print(f"  netstat -ano | findstr :{port}")
        return
    
    print("\n" + "="*60)
    print("  多模态数据分析师Agent API服务器")
    print("="*60)
    print(f"  服务地址: http://{host}:{port}")
    print(f"  Web界面: http://{host}:{port}/")
    print("="*60 + "\n")
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        print(f"\n[错误] 启动服务器失败: {str(e)}")
        print(f"请检查端口 {port} 是否可用")

if __name__ == "__main__":
    main()

