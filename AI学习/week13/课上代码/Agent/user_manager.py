"""
用户管理模块
管理用户登录、注册和会话，用户信息存储在内存中
"""

from __future__ import annotations

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class User:
    """用户数据类"""
    username: str
    password_hash: str
    email: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    games_played: int = 0
    games_won: int = 0

    @property
    def win_rate(self) -> float:
        """计算胜率"""
        if self.games_played == 0:
            return 0.0
        return round(self.games_won / self.games_played, 2)


@dataclass
class UserSession:
    """用户会话类"""
    user: User
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def update_activity(self):
        """更新最后活动时间"""
        self.last_activity = datetime.now()


class UserManager:
    """用户管理器 - 单例模式，数据存储在内存中"""

    _instance: Optional[UserManager] = None

    def __new__(cls) -> UserManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._users: Dict[str, User] = {}  # username -> User
        self._sessions: Dict[str, UserSession] = {}  # session_id -> UserSession
        self._current_session: Optional[UserSession] = None

        # 创建默认用户
        self._create_default_users()

        self._initialized = True

    def _create_default_users(self):
        """创建默认测试用户"""
        default_users = [
            ("admin", "admin123", "admin@gomoku.com"),
            ("player1", "password1", "player1@gomoku.com"),
            ("guest", "guest123", "guest@gomoku.com")
        ]

        for username, password, email in default_users:
            if username not in self._users:
                self.register_user(username, password, email)
                print(f"✓ 创建默认用户: {username}")

    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username: str, password: str, email: str = "") -> str:
        """
        注册新用户

        Args:
            username: 用户名
            password: 密码
            email: 邮箱（可选）

        Returns:
            注册结果信息
        """
        if not username or not password:
            return "❌ 用户名和密码不能为空"

        if len(username) < 3 or len(username) > 20:
            return "❌ 用户名长度必须在3-20个字符之间"

        if len(password) < 6:
            return "❌ 密码长度至少6个字符"

        if username in self._users:
            return f"❌ 用户名 '{username}' 已存在"

        # 创建新用户
        user = User(
            username=username,
            password_hash=self._hash_password(password),
            email=email
        )

        self._users[username] = user
        return f"✅ 用户 '{username}' 注册成功"

    def login(self, username: str, password: str) -> str:
        """
        用户登录

        Args:
            username: 用户名
            password: 密码

        Returns:
            登录结果信息
        """
        if not username or not password:
            return "❌ 用户名和密码不能为空"

        user = self._users.get(username)
        if not user:
            return f"❌ 用户名 '{username}' 不存在"

        if user.password_hash != self._hash_password(password):
            return "❌ 密码错误"

        # 创建会话
        session_id = hashlib.md5(
            f"{username}{datetime.now()}".encode()
        ).hexdigest()[:16]

        session = UserSession(user=user, session_id=session_id)
        session.update_activity()

        # 更新用户最后登录时间
        user.last_login = datetime.now()

        # 保存会话
        self._sessions[session_id] = session
        self._current_session = session

        return f"✅ 用户 '{username}' 登录成功 (会话ID: {session_id})"

    def logout(self, session_id: Optional[str] = None) -> str:
        """
        用户登出

        Args:
            session_id: 会话ID（可选，默认使用当前会话）

        Returns:
            登出结果信息
        """
        if session_id is None:
            if self._current_session:
                session_id = self._current_session.session_id
            else:
                return "❌ 未找到活跃会话"

        session = self._sessions.get(session_id)
        if not session:
            return f"❌ 会话ID '{session_id}' 不存在"

        username = session.user.username

        # 移除会话
        del self._sessions[session_id]

        # 如果是当前会话，清除当前会话
        if self._current_session and self._current_session.session_id == session_id:
            self._current_session = None

        return f"✅ 用户 '{username}' 已登出"

    def get_current_user(self) -> Optional[User]:
        """获取当前登录用户"""
        return self._current_session.user if self._current_session else None

    def get_user_info(self, username: str = None) -> Dict:
        """
        获取用户信息

        Args:
            username: 用户名（可选，默认使用当前用户）

        Returns:
            用户信息字典
        """
        if username is None:
            user = self.get_current_user()
            if not user:
                return {"error": "未登录"}
        else:
            user = self._users.get(username)
            if not user:
                return {"error": f"用户 '{username}' 不存在"}

        return {
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": user.last_login.strftime("%Y-%m-%d %H:%M:%S") if user.last_login else "从未登录",
            "games_played": user.games_played,
            "games_won": user.games_won,
            "win_rate": user.win_rate
        }

    def update_game_stats(self, username: str, won: bool) -> str:
        """
        更新用户游戏统计

        Args:
            username: 用户名
            won: 是否获胜

        Returns:
            更新结果信息
        """
        user = self._users.get(username)
        if not user:
            return f"❌ 用户 '{username}' 不存在"

        user.games_played += 1
        if won:
            user.games_won += 1

        return f"✅ 用户 '{username}' 游戏统计已更新"

    def list_users(self) -> List[Dict]:
        """列出所有用户（管理员功能）"""
        return [
            {
                "username": user.username,
                "email": user.email,
                "games_played": user.games_played,
                "games_won": user.games_won,
                "win_rate": user.win_rate
            }
            for user in self._users.values()
        ]

    def change_password(self, old_password: str, new_password: str) -> str:
        """
        修改当前用户密码

        Args:
            old_password: 旧密码
            new_password: 新密码

        Returns:
            修改结果信息
        """
        user = self.get_current_user()
        if not user:
            return "❌ 未登录"

        if user.password_hash != self._hash_password(old_password):
            return "❌ 旧密码错误"

        if len(new_password) < 6:
            return "❌ 新密码长度至少6个字符"

        user.password_hash = self._hash_password(new_password)
        return "✅ 密码修改成功"

    def get_session_info(self) -> Dict:
        """获取当前会话信息"""
        if not self._current_session:
            return {"error": "无活跃会话"}

        session = self._current_session
        return {
            "session_id": session.session_id,
            "username": session.user.username,
            "created_at": session.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "last_activity": session.last_activity.strftime("%Y-%m-%d %H:%M:%S")
        }


# 全局用户管理器实例
user_manager = UserManager()