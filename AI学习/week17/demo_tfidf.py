# -*- coding: utf-8 -*-
"""
TF-IDF推荐系统演示脚本
展示基于TF-IDF特征提取的推荐效果
Date: 2024-12-04
"""

from recommendation_system import RecommendationSystem
import numpy as np


def create_demo_data():
    """
    创建演示数据 - 更多样化的电影类型
    """
    ratings = {
        1: {101: 5, 103: 4, 106: 5, 109: 4},    # 喜欢科幻、冒险、动作
        2: {102: 5, 104: 4, 107: 5, 111: 3},    # 喜欢剧情、爱情
        3: {105: 5, 108: 4, 112: 5, 114: 4},    # 喜欢动画、奇幻
        4: {110: 5, 113: 4, 115: 5, 116: 3},    # 喜欢悬疑、惊悚
        5: {101: 4, 105: 3, 109: 5, 112: 4}     # 混合偏好
    }

    movies = {
        101: {'title': '星际穿越', 'genres': ['科幻', '冒险', '剧情'], 'director': '诺兰', 'actors': ['马修·麦康纳', '安妮·海瑟薇']},
        102: {'title': '罗马假日', 'genres': ['爱情', '剧情'], 'director': '威廉·惠勒', 'actors': ['格利高里·派克', '奥黛丽·赫本']},
        103: {'title': '盗梦空间', 'genres': ['科幻', '动作', '悬疑'], 'director': '诺兰', 'actors': ['莱昂纳多', '玛丽昂·歌迪亚']},
        104: {'title': '泰坦尼克号', 'genres': ['爱情', '剧情', '冒险'], 'director': '卡梅隆', 'actors': ['莱昂纳多', '凯特·温斯莱特']},
        105: {'title': '千与千寻', 'genres': ['动画', '奇幻', '冒险'], 'director': '宫崎骏', 'actors': ['柊瑠美', '入野自由']},
        106: {'title': '黑客帝国', 'genres': ['科幻', '动作'], 'director': '沃卓斯基姐妹', 'actors': ['基努·里维斯', '劳伦斯·菲什伯恩']},
        107: {'title': '肖申克的救赎', 'genres': ['剧情', '犯罪'], 'director': '弗兰克·德拉邦特', 'actors': ['蒂姆·罗宾斯', '摩根·弗里曼']},
        108: {'title': '疯狂动物城', 'genres': ['动画', '喜剧', '冒险'], 'director': '拜恩·霍华德', 'actors': ['金妮弗·古德温', '杰森·贝特曼']},
        109: {'title': '黑暗骑士', 'genres': ['动作', '犯罪', '剧情'], 'director': '诺兰', 'actors': ['克里斯蒂安·贝尔', '希斯·莱杰']},
        110: {'title': '盗梦侦探', 'genres': ['悬疑', '惊悚', '动画'], 'director': '今敏', 'actors': ['林原惠美', '古谷彻']},
        111: {'title': '阿甘正传', 'genres': ['剧情', '爱情'], 'director': '罗伯特·泽米吉斯', 'actors': ['汤姆·汉克斯', '罗宾·怀特']},
        112: {'title': '龙猫', 'genres': ['动画', '家庭', '奇幻'], 'director': '宫崎骏', 'actors': ['日高法子', '坂本千夏']},
        113: {'title': '致命ID', 'genres': ['悬疑', '惊悚', '犯罪'], 'director': '詹姆斯·曼高德', 'actors': ['约翰·库萨克', '雷·利奥塔']},
        114: {'title': '天空之城', 'genres': ['动画', '冒险', '奇幻'], 'director': '宫崎骏', 'actors': ['田中真弓', '横泽启子']},
        115: {'title': '禁闭岛', 'genres': ['悬疑', '惊悚', '剧情'], 'director': '马丁·斯科塞斯', 'actors': ['莱昂纳多', '马克·鲁法洛']},
        116: {'title': '楚门的世界', 'genres': ['剧情', '科幻'], 'director': '彼得·威尔', 'actors': ['金·凯瑞', '劳拉·琳妮']}
    }

    return ratings, movies


def demonstrate_tfidf_features(recommendation_system):
    """
    演示TF-IDF特征提取
    """
    print("\n" + "=" * 60)
    print("TF-IDF特征提取演示")
    print("=" * 60)

    # 展示几个电影的特征向量
    sample_movies = [101, 102, 105, 110]

    for movie_id in sample_movies:
        movie_info = recommendation_system.get_movie_info(movie_id)
        feature_text = recommendation_system._extract_movie_features(movie_id)

        print(f"\n电影《{movie_info['title']}》的特征：")
        print(f"类型: {movie_info['genres']}")
        print(f"导演: {movie_info['director']}")
        print(f"演员: {movie_info['actors'][:2]}...")  # 只显示前2个演员
        print(f"TF-IDF特征文本: {feature_text}")


def demonstrate_user_profiles(recommendation_system):
    """
    演示用户画像构建
    """
    print("\n" + "=" * 60)
    print("TF-IDF用户画像演示")
    print("=" * 60)

    for user_id in [1, 2, 3, 4]:
        user_ratings = recommendation_system.ratings[user_id]
        print(f"\n用户{user_id}的评分记录：")
        for movie_id, rating in user_ratings.items():
            movie_title = recommendation_system.get_movie_info(movie_id)['title']
            print(f"  《{movie_title}》: {rating}分")

        # 构建用户画像
        user_profile = recommendation_system.build_user_profile(user_id)
        if user_profile is not None:
            print(f"用户画像向量维度: {user_profile.shape}")
            print(f"向量范数: {np.linalg.norm(user_profile):.3f}")

            # 计算与其他用户的相似度
            if user_id < 4:
                other_user_id = user_id + 1
                other_profile = recommendation_system.build_user_profile(other_user_id)
                if other_profile is not None:
                    similarity = np.dot(user_profile, other_profile)
                    print(f"与用户{other_user_id}的相似度: {similarity:.3f}")


def demonstrate_content_similarity(recommendation_system):
    """
    演示基于内容相似度的推荐
    """
    print("\n" + "=" * 60)
    print("基于内容相似度的推荐演示")
    print("=" * 60)

    # 选择一个用户进行详细分析
    user_id = 1
    user_ratings = recommendation_system.ratings[user_id]

    print(f"用户{user_id}的历史评分：")
    for movie_id, rating in user_ratings.items():
        movie_title = recommendation_system.get_movie_info(movie_id)['title']
        print(f"  《{movie_title}》: {rating}分")

    # 获取基于内容的推荐
    recommendations = recommendation_system.content_based_recommend(user_id, top_n=5)

    print(f"\n基于TF-IDF内容相似度的推荐结果：")
    for i, (movie_id, score, reason) in enumerate(recommendations, 1):
        movie_info = recommendation_system.get_movie_info(movie_id)
        print(f"{i}. 《{movie_info['title']}》 - 相似度分数: {score}")
        print(f"   类型: {movie_info['genres']} | 导演: {movie_info['director']}")
        print(f"   推荐理由: {reason}")

    # 计算电影间的相似度
    print(f"\n电影内容相似度分析：")
    high_rated_movies = [mid for mid, rating in user_ratings.items() if rating >= 4]

    if len(high_rated_movies) >= 2:
        movie_a = high_rated_movies[0]
        movie_b = high_rated_movies[1]

        # 获取这两部电影之间的相似度
        if hasattr(recommendation_system, 'tfidf_matrix'):
            idx_a = list(recommendation_system.movies.keys()).index(movie_a)
            idx_b = list(recommendation_system.movies.keys()).index(movie_b)

            if hasattr(recommendation_system.tfidf_matrix, 'toarray'):  # sklearn版本
                vec_a = recommendation_system.tfidf_matrix[idx_a].toarray().flatten()
                vec_b = recommendation_system.tfidf_matrix[idx_b].toarray().flatten()
            else:  # 简化版本
                vec_a = recommendation_system.tfidf_matrix[idx_a]
                vec_b = recommendation_system.tfidf_matrix[idx_b]

            similarity = np.dot(vec_a, vec_b) / (
                np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8
            )

            title_a = recommendation_system.get_movie_info(movie_a)['title']
            title_b = recommendation_system.get_movie_info(movie_b)['title']
            print(f"《{title_a}》与《{title_b}》的内容相似度: {similarity:.3f}")


def main():
    """
    主演示函数
    """
    print("TF-IDF推荐系统演示")
    print("展示基于TF-IDF特征提取的电影推荐效果")

    # 创建演示数据
    ratings, movies = create_demo_data()

    # 初始化推荐系统
    print("\n正在初始化TF-IDF推荐系统...")
    recommendation_system = RecommendationSystem(ratings, movies)

    print(f"数据规模: {len(ratings)}个用户, {len(movies)}部电影")

    # 演示各项功能
    demonstrate_tfidf_features(recommendation_system)
    demonstrate_user_profiles(recommendation_system)
    demonstrate_content_similarity(recommendation_system)

    print("\n" + "=" * 60)
    print("TF-IDF推荐系统演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()