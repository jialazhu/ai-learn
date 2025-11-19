"""
用户系统测试脚本
直接测试用户管理功能，不依赖Agent
"""

from __future__ import annotations

import json
from user_manager import user_manager


def test_user_system():
    """测试用户系统功能"""
    print("🧪 用户系统测试开始")
    print("=" * 50)

    # 测试1: 获取默认用户列表
    print("\n[1] 查看默认用户:")
    users = user_manager.list_users()
    print(json.dumps(users, ensure_ascii=False, indent=2))

    # 测试2: 注册新用户
    print("\n[2] 注册新用户:")
    result = user_manager.register_user("testuser", "testpass123", "test@example.com")
    print(f"注册结果: {result}")

    # 测试3: 重复注册相同用户名
    print("\n[3] 重复注册相同用户名:")
    result = user_manager.register_user("testuser", "anotherpass", "test2@example.com")
    print(f"注册结果: {result}")

    # 测试4: 用户登录
    print("\n[4] 用户登录:")
    result = user_manager.login("testuser", "testpass123")
    print(f"登录结果: {result}")

    # 测试5: 获取当前用户信息
    print("\n[5] 获取当前用户信息:")
    current_user = user_manager.get_current_user()
    if current_user:
        print(f"当前用户: {current_user.username}")
        info = user_manager.get_user_info()
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print("没有当前用户")

    # 测试6: 获取会话信息
    print("\n[6] 获取会话信息:")
    session_info = user_manager.get_session_info()
    print(json.dumps(session_info, ensure_ascii=False, indent=2))

    # 测试7: 更新游戏统计
    print("\n[7] 更新游戏统计:")
    result = user_manager.update_game_stats("testuser", True)  # 胜利
    print(f"更新胜利统计: {result}")
    result = user_manager.update_game_stats("testuser", False)  # 失败
    print(f"更新失败统计: {result}")

    # 测试8: 查看更新后的用户信息
    print("\n[8] 查看更新后的用户信息:")
    info = user_manager.get_user_info("testuser")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    # 测试9: 修改密码
    print("\n[9] 修改密码:")
    result = user_manager.change_password("testpass123", "newpass456")
    print(f"修改密码结果: {result}")

    # 测试10: 使用旧密码登录失败
    print("\n[10] 使用旧密码登录失败:")
    result = user_manager.login("testuser", "testpass123")
    print(f"旧密码登录结果: {result}")

    # 测试11: 使用新密码登录成功
    print("\n[11] 使用新密码登录成功:")
    result = user_manager.login("testuser", "newpass456")
    print(f"新密码登录结果: {result}")

    # 测试12: 用户登出
    print("\n[12] 用户登出:")
    result = user_manager.logout()
    print(f"登出结果: {result}")

    # 测试13: 验证登出后无当前用户
    print("\n[13] 验证登出后状态:")
    current_user = user_manager.get_current_user()
    print(f"当前用户: {current_user}")

    # 测试14: 查看所有用户统计
    print("\n[14] 查看所有用户统计:")
    users = user_manager.list_users()
    print(json.dumps(users, ensure_ascii=False, indent=2))

    print("\n" + "=" * 50)
    print("✅ 用户系统测试完成")


if __name__ == "__main__":
    test_user_system()