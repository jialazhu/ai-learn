"""
PvP（玩家对玩家）模式演示脚本
展示人类对人类的五子棋对局功能
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

# 确保脚本可直接运行：把当前目录加入 sys.path
_CUR_DIR = Path(__file__).resolve().parent
if str(_CUR_DIR) not in sys.path:
    sys.path.insert(0, str(_CUR_DIR))

# 强制标准输出为 UTF-8，避免 Windows 控制台中文乱码
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    pass

from run_demo import demo_play_with_player


def main():
    """主函数"""
    print("🎮 五子棋 PvP（玩家对玩家）模式")
    print("=" * 50)
    print("📋 游戏规则:")
    print("  • 黑棋先行（玩家1），白棋后行（玩家2）")
    print("  • 任意方向连成5子即获胜")
    print("  • 输入坐标格式：行,列（例如：7,7）")
    print("  • 坐标范围：0-14（15x15棋盘）")
    print("  • 输入 'hint' 获取AI建议")
    print("  • 输入 'quit' 退出游戏")
    print()
    print("💡 提示：如果已登录用户，游戏进度会自动记录到用户统计中")
    print()

    try:
        input("按回车键开始游戏...")
        demo_play_with_player()
    except KeyboardInterrupt:
        print("\n👋 游戏已退出")
    except Exception as exc:
        print(f"\n❌ 游戏出错: {exc}")


if __name__ == "__main__":
    main()