"""
使用本地 Ollama 的 qwen3:4b 模型进行对话

先确保本地已安装并拉取模型：
    ollama list          # 确认 qwen3:4b 已存在
    # 如未启动，运行 ollama serve（通常已随 Ollama 自启）

运行示例：
    python ollama_chat.py --prompt "你好，介绍一下这个项目"

可选参数：
    --model     模型名称（默认 qwen3:4b）
    --system    系统提示词
    --stream    开启流式输出
"""

import argparse
import json
import sys
import requests


def chat(model: str, prompt: str, system: str = "", stream: bool = False):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system} if system else None,
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
    }
    # 过滤掉 None 的 system
    payload["messages"] = [m for m in payload["messages"] if m]

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()

    if stream:
        # 流式响应：逐行打印 content
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            delta = data.get("message", {}).get("content", "")
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        print()
    else:
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        print(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="用户输入")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama 模型名称")
    parser.add_argument("--system", default="", help="系统提示词")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    args = parser.parse_args()

    chat(model=args.model, prompt=args.prompt, system=args.system, stream=args.stream)


if __name__ == "__main__":
    main()

