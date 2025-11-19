"""
用户相关工具
为Agent提供用户登录、注册和信息查询功能
"""

from __future__ import annotations

import json
from typing import Dict, Any
from user_manager import user_manager


def user_register(username: str, password: str, email: str = "") -> str:
    """
    注册新用户

    Args:
        username: 用户名 (3-20字符)
        password: 密码 (至少6字符)
        email: 邮箱地址 (可选)

    Returns:
        注册结果信息
    """
    result = user_manager.register_user(username, password, email)
    return result


def user_login(username: str, password: str) -> str:
    """
    用户登录

    Args:
        username: 用户名
        password: 密码

    Returns:
        登录结果信息，包含会话ID
    """
    result = user_manager.login(username, password)
    return result


def user_logout() -> str:
    """
    用户登出

    Returns:
        登出结果信息
    """
    result = user_manager.logout()
    return result


def get_user_info(username: str = "") -> str:
    """
    获取用户信息

    Args:
        username: 用户名 (可选，不提供则获取当前用户信息)

    Returns:
        用户信息的JSON字符串
    """
    try:
        info = user_manager.get_user_info(username if username else None)
        return json.dumps(info, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"获取用户信息失败: {str(e)}"}, ensure_ascii=False)


def get_current_user() -> str:
    """
    获取当前登录用户信息

    Returns:
        当前用户信息的JSON字符串
    """
    try:
        current_user = user_manager.get_current_user()
        if not current_user:
            return json.dumps({"error": "未登录"}, ensure_ascii=False)

        info = {
            "username": current_user.username,
            "email": current_user.email,
            "games_played": current_user.games_played,
            "games_won": current_user.games_won,
            "win_rate": current_user.win_rate
        }
        return json.dumps(info, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"获取当前用户信息失败: {str(e)}"}, ensure_ascii=False)


def change_user_password(old_password: str, new_password: str) -> str:
    """
    修改当前用户密码

    Args:
        old_password: 旧密码
        new_password: 新密码

    Returns:
        修改结果信息
    """
    result = user_manager.change_password(old_password, new_password)
    return result


def list_all_users() -> str:
    """
    列出所有用户 (管理员功能)

    Returns:
        所有用户列表的JSON字符串
    """
    try:
        users = user_manager.list_users()
        return json.dumps(users, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"获取用户列表失败: {str(e)}"}, ensure_ascii=False)


def get_session_info() -> str:
    """
    获取当前会话信息

    Returns:
        会话信息的JSON字符串
    """
    try:
        info = user_manager.get_session_info()
        return json.dumps(info, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"获取会话信息失败: {str(e)}"}, ensure_ascii=False)


def update_game_statistics(won: bool = False) -> str:
    """
    更新当前用户的游戏统计

    Args:
        won: 是否获胜 (True=胜利, False=失败)

    Returns:
        更新结果信息
    """
    current_user = user_manager.get_current_user()
    if not current_user:
        return "❌ 未登录，无法更新游戏统计"

    result = user_manager.update_game_stats(current_user.username, won)
    return result


def parse_register_args(args: str) -> str:
    """解析注册参数"""
    try:
        parts = [p.strip() for p in args.split(',', 2)]
        if len(parts) < 2:
            return "❌ 参数格式错误，应为: username,password,email"

        username = parts[0].strip('"\'')
        password = parts[1].strip('"\'')
        email = parts[2].strip('"\'') if len(parts) > 2 else ""

        return user_register(username, password, email)
    except Exception as e:
        return f"❌ 注册失败: {str(e)}"


def parse_login_args(args: str) -> str:
    """解析登录参数"""
    try:
        parts = [p.strip() for p in args.split(',', 1)]
        if len(parts) < 2:
            return "❌ 参数格式错误，应为: username,password"

        username = parts[0].strip('"\'')
        password = parts[1].strip('"\'')

        return user_login(username, password)
    except Exception as e:
        return f"❌ 登录失败: {str(e)}"


def parse_change_password_args(args: str) -> str:
    """解析修改密码参数"""
    try:
        parts = [p.strip() for p in args.split(',', 1)]
        if len(parts) < 2:
            return "❌ 参数格式错误，应为: old_password,new_password"

        old_password = parts[0].strip('"\'')
        new_password = parts[1].strip('"\'')

        return change_user_password(old_password, new_password)
    except Exception as e:
        return f"❌ 修改密码失败: {str(e)}"