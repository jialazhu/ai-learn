# 用户登录系统

## 功能概述

为五子棋AI Agent添加了完整的用户登录和会话管理系统，所有用户信息存储在内存中。

## 主要功能

### 1. 用户注册 (userRegister)
- 注册新用户账号
- 支持用户名、密码、邮箱
- 用户名长度3-20字符，密码至少6字符
- 防止重复用户名

**用法示例:**
```
请调用userRegister工具，输入: 'newuser,password123,user@example.com'
```

### 2. 用户登录 (userLogin)
- 用户身份验证
- 创建用户会话
- 记录登录时间

**用法示例:**
```
请调用userLogin工具，输入: 'username,password'
```

### 3. 用户登出 (userLogout)
- 结束用户会话
- 清除当前登录状态

**用法示例:**
```
请调用userLogout工具
```

### 4. 用户信息管理
- **getCurrentUser**: 获取当前登录用户信息
- **getUserInfo**: 获取指定用户信息
- **listUsers**: 列出所有用户（管理员功能）

**用法示例:**
```
请调用getCurrentUser工具获取当前用户信息
请调用getUserInfo工具，输入: 'admin'
请调用listUsers工具查看所有用户
```

### 5. 密码管理 (changePassword)
- 修改当前用户密码
- 验证旧密码

**用法示例:**
```
请调用changePassword工具，输入: 'oldpassword,newpassword123'
```

### 6. 游戏统计 (updateGameStats)
- 记录用户游戏场次和胜负
- 自动计算胜率

**用法示例:**
```
请调用updateGameStats工具，输入: 'true'  # 胜利
请调用updateGameStats工具，输入: 'false' # 失败
```

### 7. 会话管理 (getSessionInfo)
- 获取当前会话详细信息
- 会话ID、创建时间、最后活动时间

## 默认用户

系统自动创建三个默认测试用户：

| 用户名 | 密码 | 邮箱 |
|--------|------|------|
| admin | admin123 | admin@gomoku.com |
| player1 | password1 | player1@gomoku.com |
| guest | guest123 | guest@gomoku.com |

## 数据结构

### User 用户对象
```python
{
    "username": "用户名",
    "password_hash": "密码哈希",
    "email": "邮箱地址",
    "created_at": "创建时间",
    "last_login": "最后登录时间",
    "games_played": "游戏场次",
    "games_won": "胜利场次",
    "win_rate": "胜率"
}
```

### UserSession 会话对象
```python
{
    "user": "用户对象",
    "session_id": "会话ID",
    "created_at": "创建时间",
    "last_activity": "最后活动时间"
}
```

## 使用方法

### 1. 通过Agent对话使用
```bash
# 运行主程序
python run_demo.py --mode auto
# 或
python run_demo.py --mode human

# 然后与Agent对话：
# "帮我注册一个用户，用户名是myuser，密码是mypass123"
# "我要登录，用户名是admin，密码是admin123"
# "查看我的用户信息"
# "修改我的密码"
```

### 2. 演示脚本
```bash
# 自动演示用户系统功能
python demo_user_system.py --mode demo

# 交互式体验
python demo_user_system.py --mode interactive
```

### 3. 直接测试
```bash
# 运行用户系统测试
python test_user_system.py
```

## 文件结构

```
Agent/
├── user_manager.py      # 用户管理核心模块
├── tools/
│   └── user_tools.py    # 用户相关工具函数
├── demo_user_system.py  # 用户系统演示脚本
├── test_user_system.py  # 用户系统测试脚本
└── USER_SYSTEM.md       # 本说明文档
```

## 安全特性

1. **密码哈希**: 使用SHA256对密码进行哈希存储
2. **会话管理**: 唯一会话ID，防止会话冲突
3. **输入验证**: 用户名长度、密码强度验证
4. **状态管理**: 登录/登出状态完整跟踪

## 注意事项

1. **内存存储**: 所有用户数据存储在内存中，程序重启后数据会重置
2. **单例模式**: UserManager使用单例模式，确保全局唯一实例
3. **默认用户**: 每次启动会自动创建默认测试用户
4. **游戏统计**: 需要在游戏结束时手动调用updateGameStats更新统计

## 示例对话

```
用户: 我想注册一个新账号
Agent: 我来帮你注册新用户。请提供用户名、密码和邮箱。

用户: 用户名是zhangsan，密码是mypassword123，邮箱是zhang@example.com
Agent: 请调用userRegister工具，输入: 'zhangsan,mypassword123,zhang@example.com'
[调用工具后]
✅ 用户 'zhangsan' 注册成功

用户: 现在我想登录
Agent: 请调用userLogin工具，输入: 'zhangsan,mypassword123'
[调用工具后]
✅ 用户 'zhangsan' 登录成功 (会话ID: abc123def456)

用户: 查看我的信息
Agent: 请调用getCurrentUser工具
[调用工具后]
返回用户信息JSON...
```

## 集成说明

用户系统已完全集成到五子棋AI Agent中：

1. **工具集成**: 所有用户功能都已添加到Agent工具集
2. **Prompt更新**: Agent知道如何使用所有用户相关工具
3. **无缝体验**: 用户可以在游戏过程中随时管理账号
4. **统计关联**: 游戏结果可以关联到具体用户

现在你的五子棋AI Agent具备了完整的用户管理系统！🎉