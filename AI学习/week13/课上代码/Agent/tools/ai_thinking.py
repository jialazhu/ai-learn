"""
AI思考逻辑模块
提供深度思考、策略分析和决策过程展示
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import json

from .gomoku_game import get_current_board, Player
from .evaluation import (
    evaluate_position,
    suggest_moves,
    _find_winning_moves,
    _find_blocking_moves,
    _find_attacking_moves,
    _find_threats
)


class ThinkingPhase(Enum):
    """思考阶段枚举"""
    SITUATION_ANALYSIS = "局面分析"
    THREAT_DETECTION = "威胁检测"
    OPPORTUNITY_SEARCH = "机会寻找"
    STRATEGIC_PLANNING = "策略规划"
    DECISION_MAKING = "最终决策"


@dataclass
class ThinkingStep:
    """思考步骤"""
    phase: ThinkingPhase
    description: str
    findings: List[str]
    confidence: float  # 置信度 0-1
    time_spent: float  # 思考时间(秒)


@dataclass
class MoveCandidate:
    """候选走法"""
    position: Tuple[int, int]
    reason: str
    priority: int
    win_probability: float
    risk_level: int  # 0-5 风险等级


class AIThinkingEngine:
    """AI思考引擎"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.thinking_history: List[ThinkingStep] = []
        self.current_depth = 3  # 默认思考深度

    def think_and_decide(self, max_suggestions: int = 5) -> Dict[str, Any]:
        """完整思考流程并返回决策结果

        Args:
            max_suggestions: 最大建议数量

        Returns:
            包含思考过程和最终决策的字典
        """
        start_time = time.time()
        board = get_current_board()

        if board.game_over:
            return {
                "status": "game_over",
                "message": "游戏已结束",
                "thinking_steps": []
            }

        # 清空历史记录
        self.thinking_history = []

        # === 第一阶段：局面分析 ===
        situation = self._analyze_situation(board)

        # === 第二阶段：威胁检测 ===
        threats = self._detect_threats(board)

        # === 第三阶段：机会寻找 ===
        opportunities = self._search_opportunities(board)

        # === 第四阶段：策略规划 ===
        strategy = self._plan_strategy(board, situation, threats, opportunities)

        # === 第五阶段：最终决策 ===
        final_decision = self._make_decision(board, strategy, max_suggestions)

        total_time = time.time() - start_time

        return {
            "status": "success",
            "current_player": "黑棋" if board.current_player == Player.BLACK else "白棋",
            "move_count": len(board.move_history),
            "thinking_time": round(total_time, 2),
            "thinking_steps": self._format_thinking_steps(),
            "analysis": {
                "situation": situation,
                "threats": threats,
                "opportunities": opportunities,
                "strategy": strategy
            },
            "decision": final_decision
        }

    def _analyze_situation(self, board) -> Dict[str, Any]:
        """第一阶段：分析当前局面"""
        step_start = time.time()
        findings = []

        # 基础信息
        move_count = len(board.move_history)
        current_player = "黑棋" if board.current_player == Player.BLACK else "白棋"

        findings.append(f"当前是{current_player}的回合")
        findings.append(f"已进行{move_count}步棋")

        # 开局/中局/残局判断
        if move_count <= 10:
            game_phase = "开局阶段"
            findings.append("当前处于开局阶段，应以占据中心和发展空间为主")
        elif move_count <= 50:
            game_phase = "中局阶段"
            findings.append("当前处于中局阶段，需要平衡攻防")
        else:
            game_phase = "残局阶段"
            findings.append("当前处于残局阶段，寻找决定性机会")

        # 棋盘密度分析
        total_cells = board.size * board.size
        density = move_count / total_cells
        findings.append(f"棋盘密度: {density:.1%} ({move_count}/{total_cells})")

        # 中心控制分析
        center_control = self._analyze_center_control(board)
        findings.extend(center_control["findings"])

        self._add_thinking_step(
            ThinkingPhase.SITUATION_ANALYSIS,
            f"分析当前局面 - {game_phase}",
            findings,
            0.9,
            time.time() - step_start
        )

        return {
            "phase": game_phase,
            "move_count": move_count,
            "density": density,
            "center_control": center_control
        }

    def _detect_threats(self, board) -> Dict[str, Any]:
        """第二阶段：检测威胁"""
        step_start = time.time()
        findings = []

        current_player = board.current_player
        opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK

        # 检测对手威胁
        opponent_threats = _find_threats(board, opponent)
        if opponent_threats:
            findings.append(f"⚠️ 发现{len(opponent_threats)}个对手威胁点需要防守")

            # 威胁等级分析
            urgent_threats = []
            for threat_pos in opponent_threats[:3]:
                findings.append(f"🔥 紧急威胁位置: {threat_pos}")
                urgent_threats.append(threat_pos)
        else:
            findings.append("✅ 未发现对手的直接威胁")

        # 检测自身威胁
        my_threats = _find_threats(board, current_player)
        if my_threats:
            findings.append(f"⭐ 发现{len(my_threats)}个己方威胁点可利用")
        else:
            findings.append("当前我方暂无直接威胁")

        # 检查即将形成的模式
        patterns = self._analyze_patterns(board)
        findings.extend(patterns["findings"])

        self._add_thinking_step(
            ThinkingPhase.THREAT_DETECTION,
            "检测攻防威胁",
            findings,
            0.85,
            time.time() - step_start
        )

        return {
            "opponent_threats": opponent_threats,
            "my_threats": my_threats,
            "urgent_threats": urgent_threats if opponent_threats else [],
            "patterns": patterns
        }

    def _search_opportunities(self, board) -> Dict[str, Any]:
        """第三阶段：寻找机会"""
        step_start = time.time()
        findings = []

        current_player = board.current_player

        # 寻找必胜走法
        winning_moves = _find_winning_moves(board)
        if winning_moves:
            findings.append(f"🎯 发现{len(winning_moves)}个必胜走法！")
            for move in winning_moves:
                findings.append(f"🏆 必胜位置: {move}")

        # 寻找攻击机会
        attacking_moves = _find_attacking_moves(board, current_player)
        if attacking_moves:
            findings.append(f"⚔️ 发现{len(attacking_moves)}个攻击机会")
        else:
            findings.append("暂无明显攻击机会")

        # 寻找好的防守位置
        opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK
        blocking_moves = _find_blocking_moves(board, opponent)
        if blocking_moves:
            findings.append(f"🛡️ 发现{len(blocking_moves)}个重要防守位置")

        # 寻找战略要点
        strategic_points = self._find_strategic_points(board)
        if strategic_points:
            findings.append(f"📍 发现{len(strategic_points)}个战略要点")

        self._add_thinking_step(
            ThinkingPhase.OPPORTUNITY_SEARCH,
            "寻找攻防机会",
            findings,
            0.8,
            time.time() - step_start
        )

        return {
            "winning_moves": winning_moves,
            "attacking_moves": attacking_moves,
            "blocking_moves": blocking_moves,
            "strategic_points": strategic_points
        }

    def _plan_strategy(self, board, situation, threats, opportunities) -> Dict[str, Any]:
        """第四阶段：制定策略"""
        step_start = time.time()
        findings = []

        # 根据游戏阶段制定策略
        phase = situation["phase"]

        if phase == "开局阶段":
            strategy = "开局策略"
            findings.append("📍 策略: 占据中心，保持棋子联系")
            findings.append("🎯 目标: 建立攻势基础，限制对手发展")

        elif phase == "中局阶段":
            strategy = "中局策略"
            if threats["opponent_threats"]:
                findings.append("📍 策略: 防守反击，优先化解威胁")
                findings.append("🎯 目标: 稳固防守后寻找反击机会")
            else:
                findings.append("📍 策略: 主动进攻，施加压力")
                findings.append("🎯 目标: 创造多重威胁，迫使对手防守")

        else:  # 残局阶段
            strategy = "残局策略"
            if opportunities["winning_moves"]:
                findings.append("📍 策略: 寻求决胜")
                findings.append("🎯 目标: 利用必胜机会结束游戏")
            else:
                findings.append("📍 策略: 精确计算")
                findings.append("🎯 目标: 避免失误，等待对手出错")

        # 优先级排序
        priority_order = self._determine_priority(threats, opportunities)
        findings.append(f"🔢 优先级: {' → '.join(priority_order)}")

        self._add_thinking_step(
            ThinkingPhase.STRATEGIC_PLANNING,
            f"制定{strategy}",
            findings,
            0.75,
            time.time() - step_start
        )

        return {
            "strategy": strategy,
            "priority_order": priority_order,
            "risk_assessment": self._assess_risks(board)
        }

    def _make_decision(self, board, strategy, max_suggestions: int) -> Dict[str, Any]:
        """第五阶段：最终决策"""
        step_start = time.time()
        findings = []

        candidates = []

        # 1. 必胜走法（最高优先级）
        winning_moves = _find_winning_moves(board)
        if winning_moves:
            for pos in winning_moves:
                candidates.append(MoveCandidate(
                    position=pos,
                    reason="必胜走法",
                    priority=100,
                    win_probability=1.0,
                    risk_level=0
                ))
            findings.append(f"🏆 选择必胜走法: {winning_moves[0]}")

        # 2. 紧急防守
        if not candidates:
            opponent = Player.WHITE if board.current_player == Player.BLACK else Player.BLACK
            blocking_moves = _find_blocking_moves(board, opponent)
            if blocking_moves:
                for pos in blocking_moves:
                    candidates.append(MoveCandidate(
                        position=pos,
                        reason="紧急防守",
                        priority=90,
                        win_probability=0.8,
                        risk_level=1
                    ))
                findings.append(f"🛡️ 选择防守位置: {blocking_moves[0]}")

        # 3. 攻击机会
        if not candidates:
            attacking_moves = _find_attacking_moves(board, board.current_player)
            if attacking_moves:
                for i, pos in enumerate(attacking_moves[:3]):
                    candidates.append(MoveCandidate(
                        position=pos,
                        reason=f"攻击机会#{i+1}",
                        priority=70 - i*5,
                        win_probability=0.6 - i*0.1,
                        risk_level=2
                    ))
                findings.append(f"⚔️ 选择攻击位置: {attacking_moves[0]}")

        # 4. 战略要点
        if not candidates:
            strategic_points = self._find_strategic_points(board)
            if strategic_points:
                for i, pos in enumerate(strategic_points[:3]):
                    candidates.append(MoveCandidate(
                        position=pos,
                        reason=f"战略要点#{i+1}",
                        priority=50 - i*3,
                        win_probability=0.4 - i*0.05,
                        risk_level=3
                    ))

        # 5. 默认中心位置
        if not candidates:
            center = board.size // 2
            default_pos = (center, center)
            if board.board[center][center] == Player.EMPTY:
                candidates.append(MoveCandidate(
                    position=default_pos,
                    reason="中心默认",
                    priority=30,
                    win_probability=0.3,
                    risk_level=2
                ))

        # 排序并选择最佳候选
        candidates.sort(key=lambda x: (-x.priority, -x.win_probability, x.risk_level))

        if candidates:
            best_choice = candidates[0]
            findings.append(f"🎯 最终决定: {best_choice.position} ({best_choice.reason})")
            confidence = min(0.95, best_choice.priority / 100)
        else:
            # 紧急情况：随机选择空位
            empty_positions = [(i, j) for i in range(board.size)
                             for j in range(board.size)
                             if board.board[i][j] == Player.EMPTY]
            if empty_positions:
                best_choice = MoveCandidate(
                    position=empty_positions[0],
                    reason="紧急选择",
                    priority=10,
                    win_probability=0.1,
                    risk_level=5
                )
                findings.append(f"⚠️ 紧急选择: {best_choice.position}")
                confidence = 0.3
            else:
                return {"status": "no_moves", "message": "棋盘已满"}

        self._add_thinking_step(
            ThinkingPhase.DECISION_MAKING,
            "最终决策",
            findings,
            confidence,
            time.time() - step_start
        )

        return {
            "status": "success",
            "chosen_position": best_choice.position,
            "reason": best_choice.reason,
            "confidence": confidence,
            "candidates": [
                {
                    "position": c.position,
                    "reason": c.reason,
                    "priority": c.priority,
                    "win_probability": c.win_probability,
                    "risk_level": c.risk_level
                } for c in candidates[:max_suggestions]
            ]
        }

    def _analyze_center_control(self, board) -> Dict[str, Any]:
        """分析中心控制情况"""
        center = board.size // 2
        findings = []

        # 检查中心区域控制
        center_area = []
        for i in range(max(0, center-1), min(board.size, center+2)):
            for j in range(max(0, center-1), min(board.size, center+2)):
                if board.board[i][j] != Player.EMPTY:
                    center_area.append((i, j, board.board[i][j]))

        black_count = sum(1 for _, _, p in center_area if p == Player.BLACK)
        white_count = sum(1 for _, _, p in center_area if p == Player.WHITE)

        if black_count > white_count:
            findings.append(f"黑棋控制中心优势 ({black_count}:{white_count})")
        elif white_count > black_count:
            findings.append(f"白棋控制中心优势 ({white_count}:{black_count})")
        else:
            findings.append(f"中心区域势均力敌 ({black_count}:{white_count})")

        return {
            "center_area": center_area,
            "black_control": black_count,
            "white_control": white_count,
            "findings": findings
        }

    def _analyze_patterns(self, board) -> Dict[str, Any]:
        """分析棋局模式"""
        findings = []

        # 这里可以扩展更复杂的模式识别
        # 现在只做基础分析

        return {
            "patterns_found": [],
            "findings": findings
        }

    def _find_strategic_points(self, board) -> List[Tuple[int, int]]:
        """寻找战略要点"""
        strategic_points = []
        center = board.size // 2

        # 优先考虑中心区域
        for i in range(max(0, center-2), min(board.size, center+3)):
            for j in range(max(0, center-2), min(board.size, center+3)):
                if board.board[i][j] == Player.EMPTY:
                    strategic_points.append((i, j))

        # 如果中心已满，寻找其他重要位置
        if len(strategic_points) < 5:
            for i in range(board.size):
                for j in range(board.size):
                    if board.board[i][j] == Player.EMPTY:
                        # 检查是否靠近现有棋子
                        if self._is_near_existing_pieces(board, i, j):
                            strategic_points.append((i, j))

        return strategic_points[:10]  # 最多返回10个

    def _is_near_existing_pieces(self, board, row: int, col: int) -> bool:
        """检查位置是否靠近现有棋子"""
        for i in range(max(0, row-1), min(board.size, row+2)):
            for j in range(max(0, col-1), min(board.size, col+2)):
                if board.board[i][j] != Player.EMPTY:
                    return True
        return False

    def _determine_priority(self, threats, opportunities) -> List[str]:
        """确定行动优先级"""
        priority = []

        if opportunities["winning_moves"]:
            priority.append("必胜")
        elif threats["urgent_threats"]:
            priority.append("紧急防守")
        elif opportunities["attacking_moves"]:
            priority.append("主动攻击")
        elif threats["opponent_threats"]:
            priority.append("防守")
        else:
            priority.append("战略布局")

        return priority

    def _assess_risks(self, board) -> Dict[str, Any]:
        """评估风险"""
        # 简单的风险评估
        move_count = len(board.move_history)

        if move_count < 10:
            risk_level = "低"
        elif move_count < 30:
            risk_level = "中"
        else:
            risk_level = "高"

        return {
            "risk_level": risk_level,
            "factors": ["棋局复杂度", "对手威胁程度", "机会成本"]
        }

    def _add_thinking_step(self, phase: ThinkingPhase, description: str,
                          findings: List[str], confidence: float, time_spent: float):
        """添加思考步骤"""
        step = ThinkingStep(
            phase=phase,
            description=description,
            findings=findings,
            confidence=confidence,
            time_spent=time_spent
        )
        self.thinking_history.append(step)

        if self.verbose:
            self._print_thinking_step(step)

    def _print_thinking_step(self, step: ThinkingStep):
        """打印思考步骤"""
        print(f"\n🧠 [{step.phase.value}] {step.description}")
        print(f"   置信度: {'⭐' * int(step.confidence * 5)} ({step.confidence:.1%})")
        print(f"   用时: {step.time_spent:.2f}秒")
        for finding in step.findings:
            print(f"   • {finding}")

    def _format_thinking_steps(self) -> List[Dict[str, Any]]:
        """格式化思考步骤用于输出"""
        return [
            {
                "phase": step.phase.value,
                "description": step.description,
                "findings": step.findings,
                "confidence": step.confidence,
                "time_spent": step.time_spent
            }
            for step in self.thinking_history
        ]


# 全局思考引擎实例
_thinking_engine = None


def get_thinking_engine() -> AIThinkingEngine:
    """获取思考引擎实例"""
    global _thinking_engine
    if _thinking_engine is None:
        _thinking_engine = AIThinkingEngine(verbose=True)
    return _thinking_engine


def ai_think_and_decide(max_suggestions: int = 5) -> str:
    """AI思考并决策的便捷函数

    Args:
        max_suggestions: 最大建议数量

    Returns:
        格式化的思考结果
    """
    engine = get_thinking_engine()
    result = engine.think_and_decide(max_suggestions)

    if result["status"] == "game_over":
        return "游戏已结束，无需思考"

    if result["status"] == "no_moves":
        return "棋盘已满，无法继续"

    # 格式化输出
    output = f"""
🤖 AI思考分析报告
==================

📊 基础信息:
• 当前玩家: {result['current_player']}
• 已走步数: {result['move_count']}步
• 思考用时: {result['thinking_time']}秒

🧠 思考过程:
"""

    for i, step in enumerate(result["thinking_steps"], 1):
        output += f"\n{i}. {step['phase']}: {step['description']}\n"
        output += f"   置信度: {'⭐' * int(step['confidence'] * 5)}\n"
        for finding in step['findings']:
            output += f"   • {finding}\n"

    output += f"\n🎯 最终决策:\n"
    output += f"   推荐走法: {result['decision']['chosen_position']}\n"
    output += f"   决策原因: {result['decision']['reason']}\n"
    output += f"   置信度: {'⭐' * int(result['decision']['confidence'] * 5)}\n"

    if len(result['decision']['candidates']) > 1:
        output += f"\n📝 其他备选方案:\n"
        for i, candidate in enumerate(result['decision']['candidates'][1:3], 2):
            output += f"   {i}. {candidate['position']} - {candidate['reason']} (优先级: {candidate['priority']})\n"

    return output.strip()


def quick_analysis() -> str:
    """快速局面分析"""
    engine = get_thinking_engine()
    board = get_current_board()

    if board.game_over:
        return "游戏已结束"

    current_player = "黑棋" if board.current_player == Player.BLACK else "白棋"
    move_count = len(board.move_history)

    # 快速威胁检测
    opponent = Player.WHITE if board.current_player == Player.BLACK else Player.BLACK
    opponent_threats = _find_threats(board, opponent)
    my_threats = _find_threats(board, board.current_player)
    winning_moves = _find_winning_moves(board)

    result = f"🔍 快速分析 ({current_player}回合, 第{move_count}步):\n"

    if winning_moves:
        result += f"🏆 发现必胜走法: {winning_moves[0]} - 强烈建议立即走此步！\n"
    elif opponent_threats:
        result += f"🚨 紧急威胁: 对手有{len(opponent_threats)}个威胁点，需要防守！\n"
        result += f"📍 威胁位置: {opponent_threats[:3]}\n"
    elif my_threats:
        result += f"⭐ 攻击机会: 我方有{len(my_threats)}个机会点可以发展\n"
    else:
        result += "😌 局面相对平稳，可以正常发展\n"

    return result