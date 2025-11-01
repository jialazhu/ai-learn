from __future__ import annotations
#启用延迟注解.数据集下载
from pathlib import Path
import json
# url标准请求库
import urllib.request
import urllib.error

#导入类型注解
from typing import List,Dict,Optional

import socket

def download_gomoku_dataset(save_path:str, dataset_type:str= "games") -> str:
    preset_urls = {
        "games": "https://raw.githubusercontent.com/example/gomoku-games/main/sample_games.json",
        "openings": "https://raw.githubusercontent.com/example/gomoku-games/main/opening_book.json"
    }

    url = preset_urls.get(dataset_type, preset_urls.get("games"))

    try:
        path = Path(save_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            #设置超时时间为10秒
            socket.setdefaulttimeout(10)
            urllib.request.urlretrieve(url, str(path))
            print(f"数据集已下载到: {path}")

            #验证下载后文件是否为json
            data =json.load(path.read_text(encoding="utf-8"))
            game_count = len(data) if isinstance(data,list) else 1
            print(f"数据集包含 {game_count} 条游戏记录")
            return f"数据集已下载到: {path}，包含 {game_count} 条游戏记录"
        except RuntimeError as e:
            print(e)
    except RuntimeError as e:
        raise e


