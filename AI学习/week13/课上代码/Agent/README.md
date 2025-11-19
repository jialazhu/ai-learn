## 五子棋 AI 自主下棋 Agent 项目

### 功能概述

基于 **LangChain ReAct 架构**构建的五子棋 AI 智能体，能够：
- 🎮 自主思考并下棋（AI vs AI 完整对局）
- 🤖 与人类对局（AI 作为对手）
- 📊 在线下载棋谱数据集并学习开局模式
- 🧠 智能评估局面并推荐最佳走法
- 💾 保存和加载游戏状态

### 技术架构

- **框架**: LangChain ReAct（Reasoning + Acting）
- **LLM**: Qwen（兼容 OpenAI 接口）
- **游戏**: 五子棋（15x15 标准棋盘）
- **工具集**: 游戏管理、局面评估、数据集下载等

### 目录结构

```
Agent/
  README.md
  requirements.txt
  .env.example
  __init__.py
  config.py                    # 配置加载
  agent_builder.py            # Agent 构建
  run_demo.py                 # 主演示脚本
  demo_ai_thinking.py         # 🧠 AI思考功能演示（新增）
  tools/
    __init__.py
    gomoku_game.py            # 五子棋游戏核心逻辑
    dataset_downloader.py     # 数据集下载工具
    evaluation.py            # 局面评估工具
    ai_thinking.py            # 🧠 AI深度思考引擎（新增）
  output/
    gomoku_dataset.json      # 下载的数据集
    *.json                   # 保存的游戏记录
```

### 安装依赖

建议使用虚拟环境：

```bash
cd week1314/Agent
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
QWEN_API_KEY=你的_qwen_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-max
```

注意：不要将真实密钥写入代码库。推荐使用 `.env` 或系统环境变量。

### 使用方法

#### 1. AI 自主下棋（完整对局）

AI 将作为黑棋和白棋，自主完成一整局游戏：

```bash
python -m Agent.run_demo --mode auto
```

或直接运行：

```bash
cd week1314/Agent
python run_demo.py --mode auto
```

#### 2. AI vs 人类对局

人类先手，AI 作为对手：

```bash
python run_demo.py --mode human
```

#### 3. 玩家对玩家 (PvP) 模式 🆕

两个人类玩家对战，支持AI建议功能：

```bash
python run_demo.py --mode pvp
```

PvP模式特色：
- 👥 两人轮流对战
- 💡 可获取AI走子建议（输入 `hint`）
- 👤 支持用户登录和游戏统计
- 🎮 完整的交互式体验

#### 4. 用户系统演示 🆕

体验用户注册、登录和游戏统计功能：

```bash
# 自动演示用户功能
python demo_user_system.py --mode demo

# 交互式用户管理
python demo_user_system.py --mode interactive

# 用户系统测试
python test_user_system.py
```

用户功能包括：
- ✅ 用户注册、登录、登出
- 📊 游戏统计追踪（场次、胜负、胜率）
- 🔐 密码管理和会话控制
- 👤 个人信息查看

#### 5. AI 思考功能演示

体验新增的深度思考功能：

```bash
# 运行AI思考演示
python demo_ai_thinking.py
```

新功能包括：
- 🧠 **深度思考分析**: 五阶段思考流程（局面分析→威胁检测→机会寻找→策略规划→最终决策）
- ⚡ **快速分析**: 实时威胁检测和必胜机会识别
- 🎯 **智能决策**: 基于置信度和风险评估的走法选择
- 📊 **思考过程可视化**: 展示AI的完整推理链条

#### 4. 自定义使用

在 Python 代码中使用：

```python
from agent_builder import build_agent
from tools.ai_thinking import ai_think_and_decide, quick_analysis

agent = build_agent()

# 初始化游戏
result = agent.invoke({"input": "请初始化一个15x15的五子棋游戏"})

# 使用AI深度思考
thinking_result = ai_think_and_decide()
print(thinking_result)

# 快速分析当前局面
quick_result = quick_analysis()
print(quick_result)

# AI自主下棋（使用思考工具）
result = agent.invoke({
    "input": "请使用aiThinkAndDecide工具分析局面，然后走出最佳一步"
})
```

### Agent 工具集

Agent 可使用的工具包括：

1. **initGame**: 初始化新游戏
2. **getBoardState**: 查看当前棋盘状态（可视化）
3. **evaluatePosition**: 评估当前局面优劣
4. **suggestMoves**: 获取最佳走法建议（防守优先、攻击机会、优先级）
5. **makeMove**: 执行走子（格式：'row,col'）
6. **aiThinkAndDecide**: 🧠 **AI深度思考分析**（新增）
7. **quickAnalysis**: ⚡ **快速局面分析**（新增）
8. **downloadDataset**: 下载五子棋数据集（棋谱、开局库）
9. **loadDataset**: 加载并查看数据集信息
10. **analyzeOpening**: 分析开局模式和走法统计
11. **saveGame**: 保存当前游戏状态
12. **loadGame**: 加载之前的游戏状态
13. **resetGame**: 重置游戏，重新开始
14. **userRegister**: 👤 **用户注册**（新增）
15. **userLogin**: 👤 **用户登录**（新增）
16. **userLogout**: 👤 **用户登出**（新增）
17. **getCurrentUser**: 👤 **获取当前用户信息**（新增）
18. **changePassword**: 👤 **修改密码**（新增）
19. **updateGameStats**: 👤 **更新游戏统计**（新增）

### 数据集功能

Agent 支持在线下载五子棋数据集：

- **棋谱数据集**: 包含经典对局记录
- **开局库**: 常见开局模式
- **自动下载**: 如果网络下载失败，会自动创建示例数据集

示例：

```python
# 下载数据集
agent.invoke({
    "input": "请下载五子棋数据集并保存到 data/games.json，类型为 games"
})

# 分析开局
agent.invoke({
    "input": "请分析 data/games.json 中的开局模式"
})
```

### 下棋策略

Agent 的下棋策略包括：

1. **防守优先**: 优先阻止对方形成五连
2. **攻击机会**: 寻找形成威胁的机会
3. **中心位置**: 优先占据中心区域
4. **局面评估**: 实时评估威胁和机会
5. **自主思考**: 基于 ReAct 框架的推理-行动循环

### 输出说明

运行后会生成：

- `output/gomoku_dataset.json`: 下载或创建的棋谱数据集
- `output/*.json`: 保存的游戏记录（可后续加载）

### 可自定义项

- 修改 `QWEN_MODEL` 使用不同模型（例如 `qwen-turbo`）
- 在 `tools/` 目录添加新工具，并在 `agent_builder.py` 中注册
- 调整 `temperature` 参数控制策略随机性（在 `agent_builder.py` 中）
- 修改棋盘大小（默认 15x15）在 `gomoku_game.py` 中

### 注意事项

1. 确保已配置 `QWEN_API_KEY` 环境变量
2. 网络连接正常（用于下载数据集）
3. 棋盘坐标范围为 0-14（15x15 棋盘）
4. 五连即获胜（横、竖、斜任意方向）
5. 黑棋先行

### 🧠 AI 思考引擎详解

新增的 AI 思考引擎提供了人类般的思考过程：

#### 思考阶段
1. **局面分析** - 评估当前游戏阶段、棋盘密度、中心控制
2. **威胁检测** - 识别对手威胁、自身威胁、关键模式
3. **机会寻找** - 发现必胜走法、攻击机会、防守位置
4. **策略规划** - 制定开局/中局/残局策略、确定优先级
5. **最终决策** - 综合评估选择最佳走法

#### 智能特性
- 🎯 **置信度评估**: 每个决策都有置信度评分
- ⏱️ **思考时间统计**: 记录每个阶段的思考耗时
- 🔢 **优先级排序**: 多维度评估候选走法
- 🛡️ **风险评估**: 评估每步棋的风险等级
- 📊 **过程可视化**: 完整展示思考链条

#### 示例思考输出
```
🤖 AI思考分析报告
==================

📊 基础信息:
• 当前玩家: 黑棋
• 已走步数: 8步
• 思考用时: 0.45秒

🧠 思考过程:

1. 局面分析: 分析当前局面 - 开局阶段
   置信度: ⭐⭐⭐⭐⭐ (90.0%)
   • 当前是黑棋的回合
   • 已进行8步棋
   • 当前处于开局阶段，应以占据中心和发展空间为主

2. 威胁检测: 检测攻防威胁
   置信度: ⭐⭐⭐⭐ (85.0%)
   • ✅ 未发现对手的直接威胁
   • ⭐ 发现2个己方威胁点可利用

3. 机会寻找: 寻找攻防机会
   置信度: ⭐⭐⭐⭐ (80.0%)
   • ⚔️ 发现3个攻击机会

🎯 最终决策:
   推荐走法: (8, 6)
   决策原因: 攻击机会#1
   置信度: ⭐⭐⭐⭐
```

### 项目特色

- ✅ **完整的 ReAct 实现**: 思考-行动-观察循环
- ✅ **深度思考引擎**: 五阶段AI推理过程
- ✅ **智能决策系统**: 基于置信度和风险评估
- ✅ **自主决策**: AI 能够独立思考和下棋
- ✅ **数据集学习**: 支持在线下载和学习开局模式
- ✅ **智能评估**: 局面分析和最佳走法推荐
- ✅ **游戏管理**: 保存、加载、重置功能完善
- ✅ **可视化思考**: 完整展示AI的思考过程

### 🚀 运行效果

运行 `python run_demo.py --mode auto` 后，你会看到增强的AI思考过程：

```
🧠 AI自主下棋演示
=======================================

[1] 初始化游戏 ⚙
✓ 初始化完成

┌─ 对局进行中 ──────────────────────────────┐
│ 🧠 AI思考分析:                           │
│   推荐走法: (8, 6)                     │
│   决策原因: 攻击机会#1                  │
│   置信度: ⭐⭐⭐⭐                        │
│ 第 1手: ● 黑棋 (7, 7)                 │
│ 第 2手: ○ 白棋 (7, 8)                 │
└───────────────────────────────────────────┘

✓ 第1轮共走了 2 步新棋
```

**新特性展示**：
- 🧠 **每步棋都有深度思考分析**
- 📊 **实时显示推荐走法和决策原因**
- ⭐ **置信度评估展示AI决策的可信度**
- 🎯 **策略透明化，用户可以看到AI的推理过程**

运行 `python run_demo.py --mode human` 与AI对局时，AI同样会展示完整的思考过程。

### 示例输出

运行 `python run_demo.py --mode auto` 后，你会看到：

```
=== Agent 思考过程 ===
Thought: 我需要使用aiThinkAndDecide工具深度分析当前局面...
Action: aiThinkAndDecide
Action Input: 5
Observation: 🤖 AI思考分析报告...
推荐走法: (7, 7)
决策原因: 中心默认
置信度: ⭐⭐⭐
...

=== Agent 最终答案 ===
我已经完成了10步棋的对局。每一步都经过深度思考分析...
```

### 故障排查

如果遇到问题：

1. **API 错误**: 检查 `.env` 文件中的 `QWEN_API_KEY` 是否正确
2. **网络错误**: 检查网络连接，数据集下载失败会自动创建示例数据
3. **导入错误**: 确保已安装所有依赖：`pip install -r requirements.txt`
4. **编码错误**: Windows 系统确保控制台支持 UTF-8
