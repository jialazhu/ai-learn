# -*- coding: utf-8 -*-
"""
推荐系统测试文件
测试物品协同过滤、基于内容的推荐和混合推荐算法
Date: 2024-12-04
"""

from recommendation_system import RecommendationSystem
import numpy as np


def create_test_data():
    """
    创建测试数据
    :return: 评分数据和电影数据
    """
    # 评分数据
    ratings = {
        1: {101: 5, 102: 4, 103: 3},
        2: {101: 4, 104: 5, 105: 4},
        3: {102: 5, 103: 4, 106: 5},
        4: {101: 5, 103: 4, 107: 3},
        5: {102: 4, 104: 5, 108: 4},
        6: {101: 3, 105: 5, 109: 4, 110: 5},
        7: {102: 5, 106: 4, 111: 3},
        8: {103: 4, 107: 5, 112: 4},
        9: {104: 5, 108: 3, 113: 5},
        10: {105: 4, 109: 5, 114: 4}
    }

    # 电影数据
    movies = {
        101: {'title': '星际穿越', 'genres': ['科幻', '冒险'], 'director': '诺兰', 'actors': ['马修·麦康纳', '安妮·海瑟薇']},
        102: {'title': '盗梦空间', 'genres': ['科幻', '动作', '悬疑'], 'director': '诺兰', 'actors': ['莱昂纳多', '玛丽昂·歌迪亚']},
        103: {'title': '泰坦尼克号', 'genres': ['爱情', '剧情'], 'director': '卡梅隆', 'actors': ['莱昂纳多', '凯特·温斯莱特']},
        104: {'title': '阿凡达', 'genres': ['科幻', '冒险', '动作'], 'director': '卡梅隆', 'actors': ['萨姆·沃辛顿', '佐伊·索尔达娜']},
        105: {'title': '黑客帝国', 'genres': ['科幻', '动作'], 'director': '沃卓斯基姐妹', 'actors': ['基努·里维斯', '劳伦斯·菲什伯恩']},
        106: {'title': '指环王', 'genres': ['奇幻', '冒险'], 'director': '彼得·杰克逊', 'actors': ['伊利亚·伍德', '维果·莫腾森']},
        107: {'title': '肖申克的救赎', 'genres': ['剧情', '犯罪'], 'director': '弗兰克·德拉邦特', 'actors': ['蒂姆·罗宾斯', '摩根·弗里曼']},
        108: {'title': '教父', 'genres': ['剧情', '犯罪'], 'director': '弗朗西斯·科波拉', 'actors': ['马龙·白兰度', '阿尔·帕西诺']},
        109: {'title': '黑暗骑士', 'genres': ['动作', '犯罪', '剧情'], 'director': '诺兰', 'actors': ['克里斯蒂安·贝尔', '希斯·莱杰']},
        110: {'title': '搏击俱乐部', 'genres': ['剧情', '惊悚'], 'director': '大卫·芬奇', 'actors': ['布拉德·皮特', '爱德华·诺顿']},
        111: {'title': '千与千寻', 'genres': ['动画', '奇幻', '冒险'], 'director': '宫崎骏', 'actors': ['柊瑠美', '入野自由']},
        112: {'title': '龙猫', 'genres': ['动画', '家庭', '奇幻'], 'director': '宫崎骏', 'actors': ['日高法子', '坂本千夏']},
        113: {'title': '天空之城', 'genres': ['动画', '冒险', '奇幻'], 'director': '宫崎骏', 'actors': ['田中真弓', '横泽启子']},
        114: {'title': '疯狂动物城', 'genres': ['动画', '喜剧', '冒险'], 'director': '拜伦·霍华德', 'actors': ['金妮弗·古德温', '杰森·贝特曼']}
    }

    return ratings, movies


def test_item_similarity(recommendation_system):
    """
    测试物品相似度计算
    """
    print("\n" + "=" * 50)
    print("测试物品相似度计算")
    print("=" * 50)

    similarity_matrix = recommendation_system.compute_item_similarity()

    # 展示几个电影之间的相似度
    test_movies = [101, 102, 103]
    for movie_a in test_movies:
        title_a = recommendation_system.get_movie_info(movie_a)['title']
        print(f"\n《{title_a}》与其他电影的相似度：")

        # 获取相似度最高的前5个电影
        similar_movies = sorted(
            similarity_matrix[movie_a].items(),
            key=lambda x: x[1],
            reverse=True
        )[1:6]  # 排除自身

        for movie_b, similarity in similar_movies:
            title_b = recommendation_system.get_movie_info(movie_b)['title']
            print(f"  《{title_b}》: {similarity:.3f}")


def test_collaborative_filtering(recommendation_system):
    """
    测试协同过滤推荐
    """
    print("\n" + "=" * 50)
    print("测试物品协同过滤推荐")
    print("=" * 50)

    # 测试用户1和用户2的推荐（Top-3）
    test_users = [1, 2]

    for user_id in test_users:
        print(f"\n用户{user_id}的历史评分：")
        user_ratings = recommendation_system.ratings[user_id]
        for movie_id, rating in user_ratings.items():
            movie_title = recommendation_system.get_movie_info(movie_id)['title']
            print(f"  《{movie_title}》: {rating}分")

        recommendations = recommendation_system.item_based_recommend(user_id, top_n=3)
        recommendation_system.display_recommendations(user_id, recommendations, "物品协同过滤")


def test_content_based(recommendation_system):
    """
    测试基于内容的推荐
    """
    print("\n" + "=" * 50)
    print("测试基于内容的推荐")
    print("=" * 50)

    # 测试用户1和用户2的推荐（Top-3）
    test_users = [1, 2]

    for user_id in test_users:
        # 构建并展示用户画像
        user_profile = recommendation_system.build_user_profile(user_id)
        print(f"\n用户{user_id}的偏好画像：")

        # 展示TF-IDF用户画像信息
        if user_profile is not None:
            print("  用户画像已构建为TF-IDF向量")
            print(f"  向量维度: {user_profile.shape}")
            print(f"  向量范数: {np.linalg.norm(user_profile):.3f}")
        else:
            print("  无法构建用户画像")

        recommendations = recommendation_system.content_based_recommend(user_id, top_n=3)
        recommendation_system.display_recommendations(user_id, recommendations, "基于内容")


def test_hybrid_recommendation(recommendation_system):
    """
    测试混合推荐
    """
    print("\n" + "=" * 50)
    print("测试混合推荐")
    print("=" * 50)

    # 测试用户1和用户2的混合推荐（Top-3）
    test_users = [1, 2]

    for user_id in test_users:
        # 测试不同权重的混合推荐
        weights = [0.3, 0.5, 0.7]

        for cf_weight in weights:
            print(f"\n--- 协同过滤权重: {cf_weight}, 基于内容权重: {1-cf_weight} ---")
            recommendations = recommendation_system.hybrid_recommend(user_id, top_n=3, cf_weight=cf_weight)
            recommendation_system.display_recommendations(user_id, recommendations, f"混合推荐(权重{cf_weight})")


def test_edge_cases(recommendation_system):
    """
    测试边界情况
    """
    print("\n" + "=" * 50)
    print("测试边界情况")
    print("=" * 50)

    # 测试不存在的用户
    print("\n1. 测试不存在的用户ID:")
    recommendations = recommendation_system.item_based_recommend(999, top_n=3)
    print(f"   用户999的推荐结果: {recommendations}")

    # 测试基于内容推荐的N=1情况
    print("\n2. 测试基于内容推荐的top_n=1（应自动调整为2）:")
    recommendations = recommendation_system.content_based_recommend(1, top_n=1)
    print(f"   用户1的推荐结果数量: {len(recommendations)}")

    # 测试空评分的用户画像
    print("\n3. 测试用户画像构建:")
    empty_profile = recommendation_system.build_user_profile(999)
    print(f"   不存在用户的画像: {empty_profile}")


def validate_recommendations(recommendation_system):
    """
    验证推荐结果的合理性
    """
    print("\n" + "=" * 50)
    print("验证推荐结果合理性")
    print("=" * 50)

    # 对用户1进行详细分析
    user_id = 1
    print(f"\n用户{user_id}的推荐分析：")

    # 用户历史评分
    user_ratings = recommendation_system.ratings[user_id]
    print("历史评分电影：")
    for movie_id, rating in user_ratings.items():
        movie_info = recommendation_system.get_movie_info(movie_id)
        print(f"  《{movie_info['title']}》 - 类型: {movie_info['genres']} - 导演: {movie_info['director']} - 评分: {rating}")

    # 协同过滤推荐分析
    print("\n协同过滤推荐分析：")
    cf_recs = recommendation_system.item_based_recommend(user_id, top_n=3)
    for movie_id, score, reason in cf_recs:
        movie_info = recommendation_system.get_movie_info(movie_id)
        print(f"  《{movie_info['title']}》 - 分数: {score} - 理由: {reason}")

    # 基于内容推荐分析
    print("\n基于内容推荐分析：")
    content_recs = recommendation_system.content_based_recommend(user_id, top_n=3)
    for movie_id, score, reason in content_recs:
        movie_info = recommendation_system.get_movie_info(movie_id)
        print(f"  《{movie_info['title']}》 - 分数: {score} - 理由: {reason}")

    # 验证推荐合理性
    print("\n推荐合理性验证：")

    # 检查是否推荐了用户已经评分的电影
    recommended_movies = set()
    for rec_list in [cf_recs, content_recs]:
        for movie_id, _, _ in rec_list:
            recommended_movies.add(movie_id)

    overlap = recommended_movies & set(user_ratings.keys())
    if overlap:
        print(f"  ❌ 发现推荐了已评分的电影: {overlap}")
    else:
        print("  ✅ 没有推荐用户已评分的电影")

    # 检查推荐分数的合理性
    if cf_recs:
        cf_scores = [score for _, score, _ in cf_recs]
        print(f"  协同过滤推荐分数范围: {min(cf_scores):.3f} - {max(cf_scores):.3f}")

    if content_recs:
        content_scores = [score for _, score, _ in content_recs]
        print(f"  基于内容推荐分数范围: {min(content_scores):.3f} - {max(content_scores):.3f}")


def main():
    """
    主测试函数
    """
    print("开始推荐系统测试...")

    # 创建测试数据
    ratings, movies = create_test_data()

    # 初始化推荐系统
    recommendation_system = RecommendationSystem(ratings, movies)

    print(f"测试数据: {len(ratings)}个用户, {len(movies)}部电影")

    # 运行各项测试
    test_item_similarity(recommendation_system)
    test_collaborative_filtering(recommendation_system)
    test_content_based(recommendation_system)
    test_hybrid_recommendation(recommendation_system)
    test_edge_cases(recommendation_system)
    validate_recommendations(recommendation_system)

    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()