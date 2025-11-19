"""
用户系统演示脚本
展示用户注册、登录和会话管理功能
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

from agent_builder import build_agent


def print_header(title: str) -> None:
    """打印标题"""
    print(f"\n╔{'═' * 60}╗")
    print(f"║ {title:^58} ║")
    print(f"╚{'═' * 60}╝\n")


def print_step(step_num: int, step_name: str, status: str = "⚙") -> None:
    """打印步骤信息"""
    status_symbols = {"⚙": "⚙", "✓": "✓", "✗": "✗", "…": "…", "⚠": "⚠"}
    symbol = status_symbols.get(status, "⚙")
    print(f"{symbol} [{step_num}] {step_name}")


def demo_user_management():
    """演示用户管理功能"""
    agent = build_agent()

    print_header("用户登录系统演示")

    # 步骤1: 注册新用户
    print_step(1, "注册新用户", "⚙")
    task_register = (
        "请使用userRegister工具注册一个新用户testuser，密码为testpass123，"
        "邮箱为test@example.com。完成注册后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_register})
        print_step(1, "用户注册完成", "✓")
        print(f"结果: {result.get('output', '')[:100]}...")
    except Exception as exc:
        print_step(1, f"注册失败: {exc}", "✗")

    # 步骤2: 用户登录
    print_step(2, "用户登录", "⚙")
    task_login = (
        "请使用userLogin工具让用户testuser登录，密码为testpass123。"
        "完成登录后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_login})
        print_step(2, "用户登录成功", "✓")
        print(f"结果: {result.get('output', '')[:100]}...")
    except Exception as exc:
        print_step(2, f"登录失败: {exc}", "✗")

    # 步骤3: 获取当前用户信息
    print_step(3, "获取当前用户信息", "⚙")
    task_get_user = (
        "请使用getCurrentUser工具获取当前登录用户的信息。"
        "完成查询后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_get_user})
        print_step(3, "用户信息获取成功", "✓")
        print(f"结果: {result.get('output', '')[:200]}...")
    except Exception as exc:
        print_step(3, f"获取用户信息失败: {exc}", "✗")

    # 步骤4: 获取会话信息
    print_step(4, "获取会话信息", "⚙")
    task_session = (
        "请使用getSessionInfo工具获取当前会话的信息。"
        "完成查询后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_session})
        print_step(4, "会话信息获取成功", "✓")
        print(f"结果: {result.get('output', '')[:200]}...")
    except Exception as exc:
        print_step(4, f"获取会话信息失败: {exc}", "✗")

    # 步骤5: 查看所有用户（管理员功能）
    print_step(5, "查看所有用户", "⚙")
    task_list_users = (
        "请使用listUsers工具查看系统中的所有用户列表。"
        "完成查询后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_list_users})
        print_step(5, "用户列表获取成功", "✓")
        print(f"结果: {result.get('output', '')[:300]}...")
    except Exception as exc:
        print_step(5, f"获取用户列表失败: {exc}", "✗")

    # 步骤6: 修改密码
    print_step(6, "修改用户密码", "⚙")
    task_change_password = (
        "请使用changePassword工具修改当前用户的密码，"
        "旧密码为testpass123，新密码为newpass456。"
        "完成修改后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_change_password})
        print_step(6, "密码修改成功", "✓")
        print(f"结果: {result.get('output', '')[:100]}...")
    except Exception as exc:
        print_step(6, f"修改密码失败: {exc}", "✗")

    # 步骤7: 用户登出
    print_step(7, "用户登出", "⚙")
    task_logout = (
        "请使用userLogout工具让当前用户登出。"
        "完成登出后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_logout})
        print_step(7, "用户登出成功", "✓")
        print(f"结果: {result.get('output', '')[:100]}...")
    except Exception as exc:
        print_step(7, f"登出失败: {exc}", "✗")

    # 步骤8: 验证登出状态
    print_step(8, "验证登出状态", "⚙")
    task_verify_logout = (
        "请使用getCurrentUser工具验证当前用户状态。"
        "完成验证后给出最终答案。"
    )

    try:
        result = agent.invoke({"input": task_verify_logout})
        print_step(8, "登出状态验证成功", "✓")
        print(f"结果: {result.get('output', '')[:200]}...")
    except Exception as exc:
        print_step(8, f"验证登出状态失败: {exc}", "✗")

    print_header("用户系统演示完成")
    print("✅ 所有用户管理功能演示完成！")
    print("✅ 用户信息存储在内存中，重启程序后数据会重置")
    print("✅ 默认用户已创建：admin/password1, player1/password1, guest/guest123")


def demo_user_interaction():
    """演示用户交互功能"""
    agent = build_agent()

    print_header("用户交互模式演示")

    print("💡 使用提示:")
    print("   - 你可以与Agent对话来使用用户系统功能")
    print("   - 尝试说：'帮我注册一个用户'、'我要登录'、'查看我的信息'")
    print("   - 输入 'quit' 或 'exit' 退出")
    print()

    while True:
        try:
            user_input = input("👉 请输入命令: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break

            if not user_input:
                continue

            print(f"\n🤖 Agent思考中...")
            result = agent.invoke({"input": user_input})
            output = result.get("output", "")
            print(f"🤖 Agent: {output}\n")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as exc:
            print(f"❌ 错误: {exc}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="用户系统演示")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "interactive"],
        default="demo",
        help="演示模式: demo=自动演示功能, interactive=交互模式",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        demo_user_management()
    else:
        demo_user_interaction()


if __name__ == "__main__":
    main()