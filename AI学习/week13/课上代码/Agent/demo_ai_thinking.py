"""
AI思考功能演示脚本
展示增强的AI思考逻辑和决策过程
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加当前目录到路径
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from agent_builder import build_agent
from tools.ai_thinking import ai_think_and_decide, quick_analysis, get_thinking_engine
from tools.gomoku_game import init_game, make_move, get_current_board


def demo_ai_thinking():
    """演示AI思考功能"""
    print("🧠 AI思考功能演示")
    print("=" * 50)

    # 初始化游戏
    print("\n📋 初始化游戏...")
    init_game(15)

    # 模拟前几步
    print("🎮 模拟开局前几步...")

    # 黑棋走中心
    make_move(7, 7)
    print("   黑棋: (7,7) - 占据中心")

    # 白棋回应
    make_move(7, 8)
    print("   白棋: (7,8) - 邻近中心")

    # 黑棋继续
    make_move(8, 8)
    print("   黑棋: (8,8) - 发展攻势")

    print(f"\n📊 当前棋盘状态:")
    board = get_current_board()
    print(f"   已走步数: {len(board.move_history)}")
    print(f"   当前玩家: {'黑棋' if board.current_player.value == 1 else '白棋'}")

    # 演示快速分析
    print("\n⚡ 快速分析:")
    print("-" * 30)
    quick_result = quick_analysis()
    print(quick_result)

    # 演示深度思考
    print("\n🧠 深度思考分析:")
    print("-" * 30)
    thinking_result = ai_think_and_decide(3)
    print(thinking_result)

    print("\n✨ 演示完成！")


def demo_agent_with_thinking():
    """演示Agent使用新的AI思考工具"""
    print("\n🤖 Agent + AI思考工具演示")
    print("=" * 50)

    agent = build_agent()

    # 重置游戏
    print("\n🔄 重置游戏...")
    try:
        result = agent.invoke({
            "input": "请重置游戏并初始化一个新的15x15五子棋游戏"
        })
        print(result["output"])
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 模拟几步棋
    print("\n🎮 模拟对局...")
    moves = [(7, 7), (7, 8), (8, 8)]
    players = ["黑棋", "白棋", "黑棋"]

    for (row, col), player in zip(moves, players):
        try:
            result = agent.invoke({
                "input": f"请作为{player}，在位置({row},{col})落子"
            })
            print(f"   {player}: ({row},{col}) ✓")
        except Exception as e:
            print(f"   {player}: ({row},{col}) ❌ {e}")

    # 使用AI思考工具
    print("\n🧠 使用AI思考工具分析当前局面...")
    try:
        result = agent.invoke({
            "input": "请使用aiThinkAndDecide工具深度分析当前局面并给出最佳走法建议"
        })
        print("\n🤖 Agent + AI思考结果:")
        print(result["output"])
    except Exception as e:
        print(f"❌ AI思考失败: {e}")

    # 使用快速分析工具
    print("\n⚡ 使用快速分析工具...")
    try:
        result = agent.invoke({
            "input": "请使用quickAnalysis工具快速评估当前局面"
        })
        print("\n🤖 Agent + 快速分析结果:")
        print(result["output"])
    except Exception as e:
        print(f"❌ 快速分析失败: {e}")


def demo_thinking_engine_settings():
    """演示思考引擎的设置选项"""
    print("\n⚙️ 思考引擎设置演示")
    print("=" * 50)

    # 获取思考引擎
    engine = get_thinking_engine()

    # 显示当前设置
    print(f"📋 当前设置:")
    print(f"   详细输出: {engine.verbose}")
    print(f"   思考深度: {engine.current_depth}")
    print(f"   历史记录数: {len(engine.thinking_history)}")

    # 测试不同设置
    print(f"\n🔧 测试不同思考深度...")

    # 重置游戏到更复杂的状态
    from tools.gomoku_game import init_game, make_move, reset_game
    reset_game()
    init_game(15)

    # 创建一个更复杂的局面
    test_moves = [
        (7, 7), (7, 8), (8, 6), (8, 8), (6, 6),
        (6, 8), (9, 5), (9, 9), (5, 5), (5, 9)
    ]

    for i, (row, col) in enumerate(test_moves):
        make_move(row, col)
        player = "黑棋" if i % 2 == 0 else "白棋"
        print(f"   {player}: ({row},{col})")

    print(f"\n🧠 分析复杂局面...")
    result = ai_think_and_decide(5)
    print(result)


def interactive_demo():
    """交互式演示"""
    print("\n🎮 交互式AI思考演示")
    print("=" * 50)
    print("输入命令来体验AI思考功能:")
    print("  'think' - 深度思考分析")
    print("  'quick' - 快速分析")
    print("  'board' - 查看棋盘")
    print("  'move row,col' - 落子 (例如: move 5,5)")
    print("  'reset' - 重置游戏")
    print("  'quit' - 退出")

    from tools.gomoku_game import get_board_state, reset_game, init_game

    # 初始化游戏
    init_game(15)

    while True:
        try:
            cmd = input("\n👉 请输入命令: ").strip().lower()

            if cmd == 'quit' or cmd == 'q':
                print("👋 再见！")
                break
            elif cmd == 'think' or cmd == 't':
                print("\n🧠 AI深度思考中...")
                result = ai_think_and_decide()
                print(result)
            elif cmd == 'quick' or cmd == 'q':
                print("\n⚡ 快速分析中...")
                result = quick_analysis()
                print(result)
            elif cmd == 'board' or cmd == 'b':
                print("\n📊 当前棋盘:")
                board_info = get_board_state()
                print(board_info)
            elif cmd.startswith('move '):
                # 解析走子命令
                try:
                    coords = cmd[5:].strip()
                    if ',' in coords:
                        row, col = map(int, coords.split(','))
                    elif ' ' in coords:
                        row, col = map(int, coords.split())
                    else:
                        print("❌ 格式错误，请使用 'move row,col' 或 'move row col'")
                        continue

                    result = make_move(row, col)
                    print(f"🎯 走子结果: {result}")

                    # 自动进行快速分析
                    print("\n⚡ 走子后快速分析:")
                    quick_result = quick_analysis()
                    print(quick_result)

                except Exception as e:
                    print(f"❌ 走子失败: {e}")
            elif cmd == 'reset' or cmd == 'r':
                reset_game()
                init_game(15)
                print("🔄 游戏已重置")
            else:
                print("❌ 未知命令，请重新输入")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    print("🎯 AI思考功能演示程序")
    print("=" * 60)

    # 选择演示模式
    print("请选择演示模式:")
    print("1. 基础AI思考演示")
    print("2. Agent + AI思考演示")
    print("3. 思考引擎设置演示")
    print("4. 交互式演示")
    print("5. 全部演示")

    try:
        choice = input("👉 请选择 (1-5): ").strip()

        if choice == '1':
            demo_ai_thinking()
        elif choice == '2':
            demo_agent_with_thinking()
        elif choice == '3':
            demo_thinking_engine_settings()
        elif choice == '4':
            interactive_demo()
        elif choice == '5':
            demo_ai_thinking()
            demo_agent_with_thinking()
            demo_thinking_engine_settings()
            print("\n" + "="*60)
            print("🎉 全部演示完成！")
            print("现在可以尝试交互式演示...")
            interactive = input("👉 是否启动交互式演示？(y/n): ").strip().lower()
            if interactive in ['y', 'yes', '是']:
                interactive_demo()
        else:
            print("❌ 无效选择，运行基础演示...")
            demo_ai_thinking()

    except KeyboardInterrupt:
        print("\n👋 演示已取消")
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("尝试运行基础演示...")
        demo_ai_thinking()