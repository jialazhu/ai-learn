"""
五子棋AI自主下棋演示脚本
展示AI Agent自主思考并下棋的能力
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
from typing import List, Tuple, Union

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

from agent_builder import build_agent


# ===================== 胜负判定（纯函数） =====================

def _cell_value(cell: Union[int, object]) -> int:
    """支持 int 或带 .value 的枚举（如 Player.BLACK.value==1, Player.WHITE.value==2）"""
    return cell if isinstance(cell, int) else getattr(cell, "value", 0)


def count_in_direction(board_matrix: List[List[Union[int, object]]],
                       r: int, c: int, dr: int, dc: int, color: int) -> int:
    """以(r,c)为中心，沿(dr,dc)与反方向统计连续同色数量（含自身）"""
    rows = len(board_matrix)
    cols = len(board_matrix[0]) if rows > 0 else 0
    cnt = 1  # 包含自身

    # 正向
    rr, cc = r + dr, c + dc
    while 0 <= rr < rows and 0 <= cc < cols and _cell_value(board_matrix[rr][cc]) == color:
        cnt += 1
        rr += dr
        cc += dc

    # 反向
    rr, cc = r - dr, c - dc
    while 0 <= rr < rows and 0 <= cc < cols and _cell_value(board_matrix[rr][cc]) == color:
        cnt += 1
        rr -= dr
        cc -= dc

    return cnt


def has_five_in_a_row(board_matrix: List[List[Union[int, object]]],
                      last_move: Tuple[int, int],
                      exact_five: bool = False,
                      forbid_black_overline: bool = False) -> Tuple[bool, int]:
    """
    胜利判定（基于最后一步）：
    - exact_five=False：连续>=5即胜（自由五子）。
    - exact_five=True：必须等于5（一般用于严格“等五”。若需要连珠黑禁手，推荐 exact_five=False + forbid_black_overline=True）。
    - forbid_black_overline=True：黑方长连(>5)不算胜（连珠黑禁手的一部分）。
    返回：(是否胜利, 获胜颜色值 1=黑,2=白,0=无)
    """
    if not last_move:
        return (False, 0)

    r, c = last_move
    if r is None or c is None:
        return (False, 0)

    if not (0 <= r < len(board_matrix) and 0 <= c < len(board_matrix[0])):
        return (False, 0)

    color = _cell_value(board_matrix[r][c])
    if color == 0:
        return (False, 0)

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        cnt = count_in_direction(board_matrix, r, c, dr, dc, color)
        if exact_five:
            if cnt == 5:
                return (True, color)
            # 若黑禁手长连：长连（>5）不判胜
            if cnt > 5 and color == 1 and forbid_black_overline:
                continue
        else:
            if cnt >= 5:
                # 自由五子允许长连；如需禁黑长连：
                if cnt > 5 and color == 1 and forbid_black_overline:
                    continue
                return (True, color)

    return (False, 0)


# ===================== 打印与辅助 =====================

def _print_header(title: str) -> None:
    """打印标题"""
    print(f"\n╔{'═' * 58}╗")
    print(f"║ {title:^56} ║")
    print(f"╚{'═' * 58}╝\n")


def _print_step(step_num: int, step_name: str, status: str = "⚙") -> None:
    """打印步骤信息"""
    status_symbols = {"⚙": "⚙", "✓": "✓", "✗": "✗", "…": "…", "⚠": "⚠"}
    symbol = status_symbols.get(status, "⚙")
    print(f"{symbol} [{step_num}] {step_name}")


def _print_progress_bar(current: int, total: int, width: int = 30) -> None:
    """打印进度条"""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    percent = int(100 * current / total) if total > 0 else 0
    print(f"│{bar}│ {percent}%", end="\r", flush=True)
    if current >= total:
        print()


def _print_mini_board(board_state: str) -> None:
    """从棋盘状态文本中提取并显示简化棋盘（保留，便于兼容已有输出）"""
    lines = board_state.split("\n")
    in_board = False
    board_lines = []
    for line in lines:
        if "棋盘状态" in line or "●" in line or "○" in line:
            in_board = True
        if in_board and ("●" in line or "○" in line or "." in line):
            board_lines.append(line[:40])  # 只显示前40个字符
        if len(board_lines) >= 10:  # 最多显示10行
            break

    if board_lines:
        print("┌─ 棋盘（文本） ─────────────────────────────┐")
        for line in board_lines[:8]:
            print(f"│ {line[:38]:<38} │")
        print("└───────────────────────────────────────────┘")


def _print_mini_board_from_matrix(board_matrix: List[List[Union[int, object]]],
                                  center: Tuple[int, int] | None = None,
                                  view_size: int = 10) -> None:
    """直接从棋盘矩阵渲染简化棋盘，默认展示以最后一步为中心的 view_size×view_size 视窗"""
    rows = len(board_matrix)
    cols = len(board_matrix[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return

    # 视窗范围
    if center is None:
        r0 = 0
        c0 = 0
    else:
        cr, cc = center
        half = view_size // 2
        r0 = max(0, min(rows - view_size, cr - half))
        c0 = max(0, min(cols - view_size, cc - half))

    r1 = min(rows, r0 + view_size)
    c1 = min(cols, c0 + view_size)

    def ch(v: Union[int, object]) -> str:
        vv = _cell_value(v)
        return "●" if vv == 1 else ("○" if vv == 2 else ".")

    print("┌─ 棋盘（矩阵） ─────────────────────────────┐")
    for r in range(r0, r1):
        line = "".join(ch(board_matrix[r][c]) for c in range(c0, c1))
        print(f"│ {line:<38} │")
    print("└───────────────────────────────────────────┘")


def _extract_move_info_from_observation(observation: str) -> tuple[str, str]:
    """从观察字符串中提取走子信息"""
    import re

    # 匹配 "黑棋/白棋在位置 (x, y) 落子成功"
    if "落子成功" in observation:
        # 先检查黑棋
        match = re.search(r"黑棋在位置\s*\((\d+),\s*(\d+)\)", observation)
        if match:
            return ("●", f"({match.group(1)},{match.group(2)})")

        # 再检查白棋
        match = re.search(r"白棋在位置\s*\((\d+),\s*(\d+)\)", observation)
        if match:
            return ("○", f"({match.group(1)},{match.group(2)})")

    return ("", "")


def _print_move_summary(steps: list) -> None:
    """打印走子摘要"""
    moves = []
    positions_seen = set()

    for action, observation in steps:
        obs_str = str(observation)
        action_tool = getattr(action, "tool", None) or str(action)

        # 只处理 makeMove 的结果
        if "makeMove" in str(action_tool) or ("落子成功" in obs_str and "位置" in obs_str):
            symbol, pos = _extract_move_info_from_observation(obs_str)
            if symbol and pos and pos not in positions_seen:
                positions_seen.add(pos)
                moves.append((symbol, pos))

    if moves:
        print("\n┌─ 走子记录 ─────────────────────────────────┐")
        for i, (symbol, pos) in enumerate(moves[-10:], 1):  # 显示最后10步
            print(f"│ {i:2d}. {symbol} {pos:<30} │")
        print("└───────────────────────────────────────────┘")


# ===================== 演示：AI 自主下棋 =====================

def demo_autonomous_game() -> None:
    """演示AI自主下棋"""
    agent = build_agent()

    _print_header("五子棋 AI 自主下棋演示")

    # 任务1: 初始化游戏
    _print_step(1, "初始化游戏", "⚙")
    task1 = "请调用 initGame 工具初始化一个15x15的五子棋游戏。完成初始化后，直接给出最终答案，说明游戏已初始化。"
    
    try:
        result1 = agent.invoke({"input": task1})
        _print_step(1, "初始化完成", "✓")
    except Exception as exc:
        _print_step(1, f"初始化失败: {exc}", "✗")
        return

    # 任务2: 下载数据集（可选，如果已存在则跳过）
    dataset_path = Path(__file__).resolve().parent / "output" / "gomoku_dataset.json"

    if dataset_path.exists():
        _print_step(2, "数据集已存在，跳过下载", "…")
    else:
        _print_step(2, "下载数据集", "⚙")
        # 直接创建示例数据集，避免网络请求卡住
        try:
            from tools.dataset_downloader import _create_sample_dataset
            _create_sample_dataset(str(dataset_path), "games")
            _print_step(2, "数据集已创建", "✓")
        except Exception as exc:
            _print_step(2, "数据集创建失败，继续", "…")

    # 任务3: AI自主下棋（完整对局）
    _print_step(3, "AI自主对局", "⚙")
    print("\n┌───────────────────────────────────────────┐")
    print("│ ● = 黑棋  ○ = 白棋                      │")
    print("└───────────────────────────────────────────┘\n")
    
    task3 = (
        "请作为黑棋和白棋，自主完成一整局五子棋游戏，直到某一方真正形成五连获胜。\n"
        "\n"
        "【重要】禁止操作：\n"
        "- 禁止调用initGame工具（游戏已经在任务1中初始化，棋盘上有0个棋子）\n"
        "- 禁止调用resetGame工具（这会清空棋盘）\n"
        "- 只能使用makeMove工具在下棋，不能重新初始化\n"
        "\n"
        "【核心规则】获胜条件（必须严格遵守）：\n"
        "- 任意一方必须在横、竖或两条对角方向连成连续5枚同色棋子（不中断、不跨空格）即获胜\n"
        "- \"活五\"和\"冲五\"都算胜（只要形成连续5颗同色棋子）\n"
        "- 必须是同一颜色、连续、不间断的5颗棋子\n"
        "- 例如：● ● ● ● ●（连续5个黑棋）= 黑棋获胜\n"
        "- 例如：● ○ ● ○ ●（交替）= 没有五连，游戏继续\n"
        "- 只有在getBoardState返回的信息中明确显示\"游戏结束\"和\"获胜\"时，才算真正获胜\n"
        "- 如果棋盘上只是交替下棋（黑白黑白），没有形成连续5颗同色棋子，游戏必须继续\n"
        "\n"
        "游戏规则：\n"
        "- 黑棋先行，然后白棋，轮流下棋\n"
        "- 每次走子会在棋盘上增加一个棋子\n"
        "- 棋盘上的棋子会越来越多，每一步都会增加一个\n"
        "- 坐标范围是0-14（15x15棋盘）\n"
        "\n"
        "你的任务流程（必须严格按照此流程）：\n"
        "1. AI深度思考：使用aiThinkAndDecide工具进行深度分析，了解当前局面和最佳走法\n"
        "2. 查看当前棋盘状态：使用getBoardState工具确认棋盘状况\n"
        "3. 执行走子：基于AI思考结果，使用makeMove工具执行最佳走子，输入格式为'行,列'（例如'7,7'）\n"
        "4. 检查游戏状态：走子后必须立即使用getBoardState检查\n"
        "5. 验证是否获胜：检查是否有连续5颗同色棋子（横、竖、斜任一方向）\n"
        "6. 继续对局：如果游戏未结束，切换玩家（黑→白，白→黑），重复步骤1-5\n"
        "\n"
        "重要：每次轮到你时，都必须先使用aiThinkAndDecide工具展示完整的思考过程！"
        "\n"
        "重要规则：\n"
        "- 每一步走子后，棋盘上的棋子总数应该增加1（例如：0→1→2→3→4...）\n"
        "- 如果发现棋子数量减少或棋盘被清空，说明错误地调用了initGame或resetGame，这是错误的！\n"
        "- 只有在棋盘上真正形成连续5颗同色棋子时，游戏才会结束\n"
        "- 必须持续下棋直到真正分出胜负（形成连续5颗同色棋子），不能提前停止\n"
        "- 双方既要进攻也要防守，阻止对方形成五连"
    )
    
    try:
        # 实时显示执行过程
        print("┌─ 对局进行中 ──────────────────────────────┐")
        
        from tools.gomoku_game import get_current_board
        max_rounds = 50 
        round_num = 0
        
        while round_num < max_rounds:
            round_num += 1
            
            # 获取当前棋盘状态
            board = get_current_board()
            current_piece_count = len(board.move_history)
            
            # 检查是否已经形成五连
            if current_piece_count > 0:
                last_move = board.move_history[-1]
                has_win, winner_color = has_five_in_a_row(
                    board_matrix=board.board,
                    last_move=last_move,
                    exact_five=False,
                    forbid_black_overline=False
                )
                if has_win and winner_color > 0:
                    winner = "黑棋" if winner_color == 1 else "白棋"
                    _print_step(3, "对局完成", "✓")
                    print("\n╔══════════════════════════════════════╗")
                    symbol = "●" if winner == "黑棋" else "○"
                    print(f"║         {symbol} {winner}获胜！                 ║")
                    print("╚══════════════════════════════════════╝")
                    break
            
            # 构建任务：继续下棋
            if round_num == 1:
                current_task = task3
            else:
                # 后续轮次：继续下棋
                current_task = (
                    f"🧠 AI深度思考并走子。当前棋盘上有 {current_piece_count} 个棋子。\n"
                    f"作为当前玩家（{'黑棋' if board.current_player.value == 1 else '白棋'}），请进行深度思考分析。\n\n"
                    f"【重要】必须按照以下流程：\n"
                    f"1. 🧠 使用aiThinkAndDecide工具进行完整的深度思考分析\n"
                    f"2. 📊 使用getBoardState查看当前棋盘状态确认\n"
                    f"3. 🎯 基于思考结果使用makeMove执行最佳走子\n"
                    f"4. ✅ 检查是否形成五连获胜\n\n"
                    f"请展示完整的AI思考过程，让用户看到你的分析逻辑！"
                )
            
            result = agent.invoke({"input": current_task})
            
            # 显示AI思考过程和走子
            new_moves = 0
            round_has_output = False
            thinking_shown = False

            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    action_tool = getattr(action, "tool", None) or str(action)
                    obs_str = str(observation)

                    # 显示AI思考过程
                    if "aiThinkAndDecide" in str(action_tool) and not thinking_shown:
                        print("│ 🧠 AI思考分析:                           │")
                        # 简化显示思考结果的关键部分
                        if "推荐走法" in obs_str:
                            lines = obs_str.split('\n')
                            for line in lines:
                                if "推荐走法" in line or "决策原因" in line or "置信度" in line:
                                    if len(line) > 40:
                                        line = line[:37] + "..."
                                    print(f"│   {line:<37} │")
                        thinking_shown = True
                        round_has_output = True

                    # 显示走子结果
                    elif "makeMove" in str(action_tool) or ("落子成功" in obs_str and "位置" in obs_str):
                        symbol, pos = _extract_move_info_from_observation(obs_str)
                        if symbol and pos:
                            board_after = get_current_board()
                            if len(board_after.move_history) > current_piece_count:
                                new_moves += 1
                                move_num = len(board_after.move_history)
                                player = "黑棋" if symbol == "●" else "白棋"
                                print(f"│ 第{move_num:2d}手: {symbol} {player} {pos:<25} │")
                                round_has_output = True
            
            # 如果有走子输出，关闭边框
            if round_has_output and round_num == 1:
                print("└───────────────────────────────────────────┘")
                print(f"\n✓ 第{round_num}轮共走了 {new_moves} 步新棋\n")
            elif new_moves > 0:
                print("└───────────────────────────────────────────┘")
                print(f"✓ 第{round_num}轮共走了 {new_moves} 步新棋\n")
            
            # 检查是否达到迭代限制
            output = result.get("output", "")
            if "iteration limit" in output.lower() or "stopped due to" in output.lower():
                print(f"\n⚠ 第{round_num}轮：达到迭代限制（50步），当前棋子数：{len(get_current_board().move_history)}")
                print(f"⚠ 游戏尚未结束，继续第{round_num + 1}轮...")
                # 继续下一轮
                continue
            
            # 检查是否形成五连
            board = get_current_board()
            piece_count = len(board.move_history)
            if piece_count > 0:
                last_move = board.move_history[-1]
                has_win, winner_color = has_five_in_a_row(
                    board_matrix=board.board,
                    last_move=last_move,
                    exact_five=False,
                    forbid_black_overline=False
                )
                if has_win and winner_color > 0:
                    winner = "黑棋" if winner_color == 1 else "白棋"
                    _print_step(3, "对局完成", "✓")
                    print("\n╔══════════════════════════════════════╗")
                    symbol = "●" if winner == "黑棋" else "○"
                    print(f"║         {symbol} {winner}获胜！                 ║")
                    print("╚══════════════════════════════════════╝")
                    break
            
            # 如果没有新走子，可能 Agent 停止了
            if new_moves == 0:
                print(f"\n⚠ 第{round_num}轮未检测到新走子，可能Agent已停止")
                if round_num < max_rounds:
                    print(f"⚠ 继续第{round_num + 1}轮...")
        
        # 最终检查
        board = get_current_board()
        piece_count = len(board.move_history)
        last_move = board.move_history[-1] if piece_count > 0 else None

        # 渲染当前棋盘
        if piece_count > 0 and last_move:
            _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)

        has_win, winner_color = has_five_in_a_row(
            board_matrix=board.board,
            last_move=last_move if last_move else (0, 0),
            exact_five=False,
            forbid_black_overline=False
        )

        if has_win and winner_color > 0:
            winner = "黑棋" if winner_color == 1 else "白棋"
            if round_num >= max_rounds:
                _print_step(3, "对局完成", "✓")
                print("\n╔══════════════════════════════════════╗")
                symbol = "●" if winner == "黑棋" else "○"
                print(f"║         {symbol} {winner}获胜！                 ║")
                print("╚══════════════════════════════════════╝")
        else:
            _print_step(3, "对局未完成", "⚙")
            print(f"\n⚠ 游戏尚未结束：棋盘上共有 {piece_count} 个棋子")
            print("⚠ 没有任何一方形成连续五颗同色棋子")
            if round_num >= max_rounds:
                print(f"⚠ 已达到最大轮次限制（{max_rounds}轮），请手动继续或增加轮次限制")

        # 如需兼容旧的文本棋盘展示（可选）
        # 这里可以显示最后一次的结果（如果需要）
        pass

    except Exception as exc:
        _print_step(3, f"对局失败: {exc}", "✗")


# ===================== 演示：AI vs 人类 =====================

def demo_play_with_human() -> None:
    """演示AI与人类对局（人类先手）"""
    agent = build_agent()

    _print_header("五子棋 AI vs 人类 对局")

    # 初始化游戏
    _print_step(1, "初始化", "⚙")
    task_init = "请调用 initGame 工具初始化一个15x15的五子棋游戏。完成初始化后，直接给出最终答案。"
    
    try:
        result = agent.invoke({"input": task_init})
        _print_step(1, "就绪", "✓")
    except Exception as exc:
        _print_step(1, f"失败: {exc}", "✗")
        return

    from tools.gomoku_game import get_current_board
    
    # 对弈循环
    round_num = 0
    max_rounds = 100  # 最多100轮（200步）
    
    while round_num < max_rounds:
        round_num += 1
        board = get_current_board()
        
        # 检查是否已结束
        piece_count = len(board.move_history)
        if piece_count > 0:
            last_move = board.move_history[-1]
            has_win, winner_color = has_five_in_a_row(
                board_matrix=board.board,
                last_move=last_move,
                exact_five=False,
                forbid_black_overline=False
            )
            if has_win and winner_color > 0:
                winner = "黑棋" if winner_color == 1 else "白棋"
                _print_step(2, "对局完成", "✓")
                print("\n╔══════════════════════════════════════╗")
                symbol = "●" if winner == "黑棋" else "○"
                winner_name = "人类" if winner == "黑棋" else "AI"
                print(f"║         {symbol} {winner_name}({winner})获胜！                 ║")
                print("╚══════════════════════════════════════╝")
                _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)
                break
        
        # 检查当前玩家
        current_player = board.current_player
        if current_player.value == 1:  # 黑棋（人类）
            # 显示当前棋盘
            if piece_count > 0:
                last_move = board.move_history[-1]
                _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)
            
            print(f"\n┌─ 第{round_num}轮：人类(●) 走子 ────────────────────┐")
            print("│ 请输入走子位置，格式：行,列 (例如: 7,7)      │")
            print("│ 输入 'q' 退出                                │")
            print("└───────────────────────────────────────────┘")
            
            # 获取人类输入
            try:
                user_input = input("👉 请输入: ").strip()
                
                if user_input.lower() == 'q':
                    print("游戏已退出")
                    return
                
                # 解析输入
                if ',' in user_input:
                    parts = user_input.split(',')
                else:
                    parts = user_input.split()
                
                if len(parts) < 2:
                    print("❌ 输入格式错误，请使用 '行,列' 格式 (例如: 7,7)")
                    round_num -= 1  # 重试本轮
                    continue
                
                row = int(parts[0].strip())
                col = int(parts[1].strip())
                
                # 验证坐标范围
                if not (0 <= row < 15 and 0 <= col < 15):
                    print("❌ 坐标超出范围，请输入0-14之间的数字")
                    round_num -= 1
                    continue
                
                # 执行人类走子
                task_human = f"请调用 makeMove 工具执行走子，位置为 {row},{col}。完成走子后，直接给出最终答案。"
                
                try:
                    result = agent.invoke({"input": task_human})
                    board_after = get_current_board()
                    
                    # 检查是否成功走子
                    if len(board_after.move_history) > piece_count:
                        print(f"✓ 人类(●) 已落子: ({row}, {col})\n")
                    else:
                        print(f"❌ 走子失败：位置 ({row}, {col}) 可能已被占用")
                        round_num -= 1
                        continue
                except Exception as exc:
                    print(f"❌ 走子失败: {exc}")
                    round_num -= 1
                    continue
                    
            except ValueError:
                print("❌ 输入格式错误，请输入数字 (例如: 7,7)")
                round_num -= 1
                continue
            except KeyboardInterrupt:
                print("\n游戏已退出")
                return
            except Exception as exc:
                print(f"❌ 输入处理错误: {exc}")
                round_num -= 1
                continue
        
        else:  # 白棋（AI）
            # 显示当前棋盘
            board = get_current_board()
            if len(board.move_history) > 0:
                last_move = board.move_history[-1]
                _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)
            
            print(f"\n┌─ 第{round_num}轮：AI(○) 思考中 ────────────────────┐")
            print("│ 🤖 AI正在分析局面并计算最佳走法...        │")
            print("└───────────────────────────────────────────┘")
            
            task_ai = (
                "🧠 AI深度思考并应对。当前轮到白棋（你）。\n\n"
                "请严格按照以下流程：\n"
                "1. 使用aiThinkAndDecide工具进行深度思考分析\n"
                "2. 基于分析结果走出最佳一步\n"
                "3. 应对人类对手的威胁，创造自己的机会\n\n"
                "请展示完整的思考过程，然后走子。"
            )
            
            try:
                print("┌─ AI思考过程 ─────────────────────────────┐")
                result = agent.invoke({"input": task_ai})
                output = result.get("output", "")

                # 显示AI思考过程摘要
                if "intermediate_steps" in result:
                    for action, observation in result["intermediate_steps"]:
                        action_tool = getattr(action, "tool", None) or str(action)
                        if "aiThinkAndDecide" in str(action_tool):
                            obs_str = str(observation)
                            # 提取关键思考信息
                            if "推荐走法" in obs_str:
                                lines = obs_str.split('\n')
                                for line in lines:
                                    if any(keyword in line for keyword in ["推荐走法", "决策原因", "置信度"]):
                                        print(f"│ {line:<44} │")
                            break

                print("└───────────────────────────────────────────┘")

                # 提取AI走子信息
                import re
                move_match = re.search(r"\((\d+),\s*(\d+)\)", output)

                board_after = get_current_board()
                if len(board_after.move_history) > len(board.move_history):
                    # 成功走子
                    last_move = board_after.move_history[-1]
                    print(f"✓ AI(○) 已落子: ({last_move[0]}, {last_move[1]})\n")
                else:
                    print("⚠ AI未成功走子，请检查\n")
                    
            except Exception as exc:
                print(f"✗ AI走子失败: {exc}\n")
    
    # 如果达到最大轮次限制
    if round_num >= max_rounds:
        print(f"\n⚠ 已达到最大轮次限制（{max_rounds}轮）")
        board = get_current_board()
        if len(board.move_history) > 0:
            last_move = board.move_history[-1]
            _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)


# ===================== 演示：人类 vs 人类 =====================

def demo_play_with_player() -> None:
    """演示人类对人类对局（PvP模式）"""
    from tools.gomoku_game import get_current_board

    _print_header("五子棋 人类 vs 人类 对局")

    # 初始化游戏
    _print_step(1, "初始化", "⚙")

    # 直接创建游戏，不需要Agent
    from tools.gomoku_game import init_game
    try:
        init_game(15)
        _print_step(1, "游戏就绪", "✓")
    except Exception as exc:
        _print_step(1, f"初始化失败: {exc}", "✗")
        return

    # 对弈循环
    round_num = 0
    max_rounds = 150  # 最多150轮（300步）

    while round_num < max_rounds:
        round_num += 1
        board = get_current_board()

        # 检查是否已结束
        piece_count = len(board.move_history)
        if piece_count > 0:
            last_move = board.move_history[-1]
            has_win, winner_color = has_five_in_a_row(
                board_matrix=board.board,
                last_move=last_move,
                exact_five=False,
                forbid_black_overline=False
            )
            if has_win and winner_color > 0:
                winner = "黑棋" if winner_color == 1 else "白棋"
                player_name = "玩家1(●)" if winner == "黑棋" else "玩家2(○)"
                _print_step(2, "对局完成", "✓")
                print("\n╔══════════════════════════════════════╗")
                symbol = "●" if winner == "黑棋" else "○"
                print(f"║         {symbol} {player_name}获胜！                 ║")
                print("╚══════════════════════════════════════╝")
                _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)
                break

        # 检查当前玩家
        current_player = board.current_player
        if current_player.value == 1:  # 黑棋（玩家1）
            player_name = "玩家1"
            player_symbol = "●"
        else:  # 白棋（玩家2）
            player_name = "玩家2"
            player_symbol = "○"

        # 显示当前棋盘
        if piece_count > 0:
            last_move = board.move_history[-1]
            _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)
        else:
            print("┌─ 空棋盘 ─────────────────────────────┐")
            print("│ 游戏开始，黑棋先行！                      │")
            print("└───────────────────────────────────────────┘")

        print(f"\n┌─ 第{round_num}轮：{player_name}({player_symbol}) 走子 ────────────────────┐")
        print("│ 请输入走子位置，格式：行,列 (例如: 7,7)      │")
        print("│ 输入 'hint' 获取建议                        │")
        print("│ 输入 'quit' 退出                            │")
        print("└───────────────────────────────────────────┘")

        # 获取玩家输入
        try:
            user_input = input(f"👉 {player_name}请输入: ").strip()

            if user_input.lower() == 'quit':
                print("游戏已退出")
                return

            if user_input.lower() == 'hint':
                # 提供走子建议
                try:
                    from tools.evaluation import suggest_moves
                    suggestions = suggest_moves(3)
                    print(f"💡 建议走法: {suggestions}")
                    round_num -= 1  # 重试本轮
                    continue
                except Exception as exc:
                    print(f"❌ 获取建议失败: {exc}")
                    round_num -= 1
                    continue

            # 解析输入
            if ',' in user_input:
                parts = user_input.split(',')
            else:
                parts = user_input.split()

            if len(parts) < 2:
                print("❌ 输入格式错误，请使用 '行,列' 格式 (例如: 7,7)")
                round_num -= 1  # 重试本轮
                continue

            row = int(parts[0].strip())
            col = int(parts[1].strip())

            # 验证坐标范围
            if not (0 <= row < 15 and 0 <= col < 15):
                print("❌ 坐标超出范围，请输入0-14之间的数字")
                round_num -= 1
                continue

            # 执行走子
            from tools.gomoku_game import make_move
            move_result = make_move(row, col)

            if "成功" in move_result:
                print(f"✓ {player_name}({player_symbol}) 已落子: ({row}, {col})")

                # 更新游戏统计（如果有用户登录）
                try:
                    from user_manager import user_manager
                    current_user = user_manager.get_current_user()
                    if current_user:
                        print(f"💾 已为用户 '{current_user.username}' 记录游戏进度")
                except Exception:
                    pass

            else:
                print(f"❌ 走子失败：{move_result}")
                round_num -= 1  # 重试本轮
                continue

        except ValueError:
            print("❌ 输入格式错误，请输入数字 (例如: 7,7)")
            round_num -= 1
            continue
        except KeyboardInterrupt:
            print("\n游戏已退出")
            return
        except Exception as exc:
            print(f"❌ 输入处理错误: {exc}")
            round_num -= 1
            continue

    # 如果达到最大轮次限制
    if round_num >= max_rounds:
        print(f"\n⚠ 已达到最大轮次限制（{max_rounds}轮）")
        board = get_current_board()
        if len(board.move_history) > 0:
            last_move = board.move_history[-1]
            _print_mini_board_from_matrix(board.board, center=last_move, view_size=10)


# ===================== 主函数入口 =====================

def main() -> None:
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="五子棋AI Agent演示")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "human", "pvp"],
        default="auto",
        help="演示模式: auto=AI自主下棋, human=AI与人类对局, pvp=人类对人类",
    )

    args = parser.parse_args()

    if args.mode == "auto":
        demo_autonomous_game()
    elif args.mode == "pvp":
        demo_play_with_player()
    else:
        demo_play_with_human()


if __name__ == "__main__":
    main()