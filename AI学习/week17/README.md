# 电影推荐系统

基于物品协同过滤和内容推荐的电影推荐系统，实现了多种推荐算法和完整的测试用例。

## 项目结构

```
week17/
├── recommendation_system.py  # 主推荐系统实现
├── test_recommendation.py    # 测试文件
├── README.md                # 说明文档
└── 推荐系统代码作业.md       # 作业要求文档
```

## 功能特性

### 1. 物品协同过滤推荐 (Item-based Collaborative Filtering)
- 使用余弦相似度计算物品间相似度
- 基于用户历史评分进行推荐
- 生成个性化推荐理由

**相似度计算公式：**
```
sim(A, B) = (用户同时喜欢A和B的数量) / sqrt(喜欢A的用户数 × 喜欢B的用户数)
```

### 2. 基于内容的推荐 (Content-based Recommendation)
- 从电影特征（类型、导演、演员）中提取内容特征
- 构建用户偏好画像
- 基于特征匹配进行推荐

### 3. 混合推荐 (Hybrid Recommendation)
- 结合协同过滤和基于内容的推荐
- 支持权重调整
- 提供更全面的推荐结果

## 快速开始

### 环境要求
- Python 3.7+
- numpy
- scikit-learn (可选，如果不安装会使用简化的TF-IDF实现)

### 安装依赖
```bash
pip install numpy scikit-learn
```

### 运行测试
```bash
python test_recommendation.py
```

### 使用示例

```python
from recommendation_system import RecommendationSystem

# 准备数据
ratings = {
    1: {101: 5, 102: 4, 103: 3},
    2: {101: 4, 104: 5, 105: 4}
}

movies = {
    101: {'title': '星际穿越', 'genres': ['科幻', '冒险'], 'director': '诺兰', 'actors': ['马修·麦康纳', '安妮·海瑟薇']},
    102: {'title': '盗梦空间', 'genres': ['科幻', '动作'], 'director': '诺兰', 'actors': ['莱昂纳多', '玛丽昂·歌迪亚']},
    103: {'title': '泰坦尼克号', 'genres': ['爱情', '剧情'], 'director': '卡梅隆', 'actors': ['莱昂纳多', '凯特·温斯莱特']},
    104: {'title': '阿凡达', 'genres': ['科幻', '冒险'], 'director': '卡梅隆', 'actors': ['萨姆·沃辛顿', '佐伊·索尔达娜']},
    105: {'title': '黑客帝国', 'genres': ['科幻', '动作'], 'director': '沃卓斯基姐妹', 'actors': ['基努·里维斯', '劳伦斯·菲什伯恩']}
}

# 初始化推荐系统
rs = RecommendationSystem(ratings, movies)

# 协同过滤推荐
cf_recs = rs.item_based_recommend(user_id=1, top_n=3)

# 基于内容的推荐
content_recs = rs.content_based_recommend(user_id=1, top_n=3)

# 混合推荐
hybrid_recs = rs.hybrid_recommend(user_id=1, top_n=3, cf_weight=0.6)

# 展示推荐结果
rs.display_recommendations(1, cf_recs, "协同过滤")
```

## 算法说明

### 1. 物品协同过滤算法

**核心思想：** 基于用户对物品的历史评分，计算物品间的相似度，然后根据用户已评分物品的相似物品进行推荐。

**实现步骤：**
1. 构建物品-用户倒排表
2. 使用余弦相似度计算物品间相似度
3. 对于目标用户，基于其高评分电影寻找相似电影
4. 按加权相似度排序生成推荐列表

### 2. 基于内容的推荐算法

**核心思想：** 使用TF-IDF提取电影的内容特征（类型、导演、演员），构建用户偏好画像，通过余弦相似度推荐与用户画像匹配的电影。

**实现步骤：**
1. 使用TF-IDF向量化器提取电影特征向量
   - 支持1-gram和2-gram特征
   - 自动计算TF-IDF权重
2. 根据用户历史评分构建TF-IDF用户画像
   - 使用评分作为权重
   - 向量归一化处理
3. 计算候选电影与用户画像的余弦相似度
4. 按相似度排序生成推荐

### 3. 混合推荐算法

**核心思想：** 结合协同过滤和基于内容两种算法的优势，通过加权融合提供更准确的推荐。

**融合策略：**
- 协同过滤权重：cf_weight (默认0.6)
- 基于内容权重：1 - cf_weight

## 数据格式

### 评分数据格式
```python
ratings = {
    user_id: {movie_id: rating},
    # 例如：
    1: {101: 5, 102: 4, 103: 3},
    2: {101: 4, 104: 5, 105: 4}
}
```

### 电影数据格式
```python
movies = {
    movie_id: {
        'title': '电影名称',
        'genres': ['类型1', '类型2'],  # 或 '类型1|类型2'
        'director': '导演姓名',
        'actors': ['演员1', '演员2']  # 或 '演员1|演员2'
    }
}
```

## API 文档

### RecommendationSystem 类

#### 初始化
```python
RecommendationSystem(ratings, movies)
```

#### 主要方法

- `compute_item_similarity()`: 计算物品相似度矩阵
- `item_based_recommend(user_id, top_n=10)`: 物品协同过滤推荐
- `build_user_profile(user_id)`: 构建用户画像
- `content_based_recommend(user_id, top_n=10)`: 基于内容的推荐
- `hybrid_recommend(user_id, top_n=10, cf_weight=0.6)`: 混合推荐
- `display_recommendations(user_id, recommendations, algorithm_name)`: 展示推荐结果

## 测试用例

测试文件 `test_recommendation.py` 包含以下测试：

1. **物品相似度计算测试**
   - 验证相似度矩阵计算的正确性
   - 展示电影间相似度关系

2. **协同过滤推荐测试**
   - 测试用户1和用户2的Top-3推荐
   - 验证推荐理由的合理性

3. **基于内容的推荐测试**
   - 测试用户画像构建
   - 验证基于特征匹配的推荐结果

4. **混合推荐测试**
   - 测试不同权重组合的推荐效果
   - 验证融合策略的有效性

5. **边界情况测试**
   - 测试不存在用户的情况
   - 测试参数边界值

6. **推荐结果验证**
   - 验证推荐结果的合理性
   - 检查是否推荐已评分电影

## 运行结果示例

```
=== 用户1的物品协同过滤推荐结果 ===
1. 《黑暗骑士》 - 推荐分数: 4.2
   类型: ['动作', '犯罪', '剧情'] | 导演: 诺兰
   推荐理由: 因为您喜欢《盗梦空间》，所以推荐《黑暗骑士》

2. 《指环王》 - 推荐分数: 3.8
   类型: ['奇幻', '冒险'] | 导演: 彼得·杰克逊
   推荐理由: 因为您喜欢《星际穿越》，所以推荐《指环王》
```

## 项目特点

1. **完整的算法实现**：涵盖三种主流推荐算法
2. **TF-IDF特征提取**：使用先进的TF-IDF算法进行内容特征提取
   - 支持scikit-learn和简化实现两种方式
   - 自动处理1-gram和2-gram特征
   - 余弦相似度计算
3. **详细的测试用例**：包含各种场景和边界情况
4. **清晰的代码结构**：函数命名规范，注释详细
5. **可扩展性**：易于添加新的推荐算法
6. **实用的推荐理由**：为每个推荐生成个性化解释

## 扩展建议

1. **冷启动问题**：为新用户提供基于热门电影的推荐
2. **实时更新**：支持用户评分的实时更新
3. **评估指标**：添加准确率、召回率等评估指标
4. **性能优化**：对大规模数据进行优化
5. **更多特征**：加入年份、评分、标签等更多特征

## 作者

AI Assistant - 2024年12月