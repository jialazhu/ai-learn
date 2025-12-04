# -*- coding: utf-8 -*-
"""
电影推荐系统
实现物品协同过滤和基于内容的推荐算法
Date: 2024-12-04
"""

import math
import numpy as np
from collections import defaultdict, Counter

# 尝试导入sklearn的TF-IDF，如果没有安装则提供简单实现
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告：scikit-learn未安装，将使用简化的TF-IDF实现")


class RecommendationSystem:
    def __init__(self, ratings, movies):
        """
        初始化推荐系统
        :param ratings: 评分数据字典 {user_id: {movie_id: rating}}
        :param movies: 电影数据字典 {movie_id: {title, genres, director, actors}}
        """
        self.ratings = ratings
        self.movies = movies
        self.item_similarity = {}
        self.movie_features = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._init_tfidf()

    def _init_tfidf(self):
        """
        初始化TF-IDF向量化器
        """
        print("正在初始化TF-IDF特征提取器...")

        # 提取所有电影的特征文本
        movie_texts = []
        movie_ids = []

        for movie_id in self.movies.keys():
            feature_text = self._extract_movie_features(movie_id)
            movie_texts.append(feature_text)
            movie_ids.append(movie_id)

        if SKLEARN_AVAILABLE:
            # 使用sklearn的TF-IDF向量化器
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),  # 使用1-gram和2-gram
                min_df=1,  # 最小文档频率
                max_df=0.8  # 最大文档频率
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(movie_texts)
            print(f"TF-IDF特征矩阵形状: {self.tfidf_matrix.shape}")
        else:
            # 简化的TF-IDF实现
            self._build_simple_tfidf(movie_texts)

        print("TF-IDF特征提取器初始化完成")

    def _build_simple_tfidf(self, movie_texts):
        """
        构建简化的TF-IDF矩阵
        """
        # 构建词汇表
        vocabulary = set()
        for text in movie_texts:
            vocabulary.update(text.split())
        vocabulary = list(vocabulary)
        self.vocabulary = {word: i for i, word in enumerate(vocabulary)}

        # 计算TF-IDF矩阵
        n_docs = len(movie_texts)
        n_vocab = len(vocabulary)
        self.tfidf_matrix = np.zeros((n_docs, n_vocab))

        # 计算文档频率
        doc_freq = np.zeros(n_vocab)
        for text in movie_texts:
            words_in_doc = set(text.split())
            for word in words_in_doc:
                if word in self.vocabulary:
                    doc_freq[self.vocabulary[word]] += 1

        # 计算TF-IDF
        for doc_idx, text in enumerate(movie_texts):
            words = text.split()
            word_count = Counter(words)
            total_words = len(words)

            for word, count in word_count.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf = count / total_words
                    idf = math.log(n_docs / doc_freq[word_idx])
                    self.tfidf_matrix[doc_idx, word_idx] = tf * idf

        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movies.keys())}

    def compute_item_similarity(self):
        """
        计算物品相似度矩阵
        使用余弦相似度：sim(A, B) = (用户同时喜欢A和B的数量) / sqrt(喜欢A的用户数 × 喜欢B的用户数)
        :return: 相似度矩阵字典 {movie_id: {similar_movie_id: similarity_score}}
        """
        print("正在计算物品相似度矩阵...")

        # 构建物品-用户倒排表
        item_users = defaultdict(set)
        for user_id, user_ratings in self.ratings.items():
            for movie_id, rating in user_ratings.items():
                # 只考虑用户评分较高的电影（评分>=4）
                if rating >= 4:
                    item_users[movie_id].add(user_id)

        # 计算物品间的余弦相似度
        similarity_matrix = {}
        movie_ids = list(self.movies.keys())

        for i, movie_a in enumerate(movie_ids):
            similarity_matrix[movie_a] = {}
            users_a = item_users.get(movie_a, set())

            for movie_b in movie_ids:
                if movie_a == movie_b:
                    similarity_matrix[movie_a][movie_b] = 1.0
                    continue

                users_b = item_users.get(movie_b, set())

                # 计算余弦相似度
                common_users = len(users_a & users_b)
                if common_users == 0:
                    similarity_matrix[movie_a][movie_b] = 0.0
                else:
                    similarity = common_users / math.sqrt(len(users_a) * len(users_b))
                    similarity_matrix[movie_a][movie_b] = similarity

        self.item_similarity = similarity_matrix
        print("物品相似度矩阵计算完成！")
        return similarity_matrix

    def item_based_recommend(self, user_id, top_n=10):
        """
        基于物品的协同过滤推荐
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        if user_id not in self.ratings:
            return []

        if not self.item_similarity:
            self.compute_item_similarity()

        # 获取用户已评分的电影
        user_ratings = self.ratings[user_id]
        rated_movies = set(user_ratings.keys())

        # 计算候选电影的推荐分数
        movie_scores = defaultdict(float)
        movie_reasons = defaultdict(list)

        for rated_movie, rating in user_ratings.items():
            if rating < 4:  # 只基于用户喜欢的电影进行推荐
                continue

            # 获取与已评分电影相似的其他电影
            similar_movies = self.item_similarity.get(rated_movie, {})

            for candidate_movie, similarity in similar_movies.items():
                if candidate_movie in rated_movies or similarity <= 0:
                    continue

                # 加权评分 = 相似度 × 用户评分
                weighted_score = similarity * rating
                movie_scores[candidate_movie] += weighted_score

                # 生成推荐理由
                if len(movie_reasons[candidate_movie]) < 3:  # 最多保留3个理由
                    movie_title = self.movies[rated_movie]['title']
                    candidate_title = self.movies[candidate_movie]['title']
                    reason = f"因为您喜欢《{movie_title}》，所以推荐《{candidate_title}》"
                    movie_reasons[candidate_movie].append((reason, similarity))

        # 按分数排序并生成最终推荐结果
        recommendations = []
        for movie_id, score in sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            # 选择相似度最高的理由
            best_reason = max(movie_reasons[movie_id], key=lambda x: x[1])[0] if movie_reasons[movie_id] else "基于您的观影偏好推荐"
            recommendations.append((movie_id, round(score, 3), best_reason))

        return recommendations

    def _extract_movie_features(self, movie_id):
        """
        提取电影特征向量
        :param movie_id: 电影ID
        :return: 特征字符串
        """
        if movie_id in self.movie_features:
            return self.movie_features[movie_id]

        movie = self.movies.get(movie_id, {})
        features = []

        # 添加类型特征
        genres = movie.get('genres', [])
        if isinstance(genres, list):
            features.extend([f"类型:{genre}" for genre in genres])
        elif isinstance(genres, str):
            if '|' in genres:
                features.extend([f"类型:{genre}" for genre in genres.split('|')])
            else:
                features.append(f"类型:{genres}")

        # 添加导演特征
        director = movie.get('director', '')
        if director:
            features.append(f"导演:{director}")

        # 添加演员特征
        actors = movie.get('actors', [])
        if isinstance(actors, list):
            features.extend([f"演员:{actor}" for actor in actors[:3]])  # 只取前3个演员
        elif isinstance(actors, str):
            if '|' in actors:
                features.extend([f"演员:{actor}" for actor in actors.split('|')[:3]])
            else:
                features.append(f"演员:{actors}")

        # 合并特征
        feature_text = ' '.join(features)
        self.movie_features[movie_id] = feature_text
        return feature_text

    def build_user_profile(self, user_id):
        """
        构建用户画像
        使用TF-IDF向量根据用户历史评分记录构建用户偏好画像
        :param user_id: 用户ID
        :return: 用户偏好向量
        """
        if user_id not in self.ratings:
            return None

        user_ratings = self.ratings[user_id]
        user_profile_vector = None

        if SKLEARN_AVAILABLE:
            # 使用sklearn的TF-IDF矩阵
            for movie_id, rating in user_ratings.items():
                if rating < 3:  # 忽略低评分
                    continue

                movie_idx = list(self.movies.keys()).index(movie_id)
                movie_vector = self.tfidf_matrix[movie_idx].toarray().flatten()

                # 使用评分作为权重
                weighted_vector = movie_vector * (rating / 5.0)  # 归一化评分

                if user_profile_vector is None:
                    user_profile_vector = weighted_vector
                else:
                    user_profile_vector += weighted_vector

            # 归一化用户画像向量
            if user_profile_vector is not None:
                norm = np.linalg.norm(user_profile_vector)
                if norm > 0:
                    user_profile_vector = user_profile_vector / norm
        else:
            # 使用简化的TF-IDF实现
            for movie_id, rating in user_ratings.items():
                if rating < 3:  # 忽略低评分
                    continue

                movie_idx = self.movie_id_to_idx[movie_id]
                movie_vector = self.tfidf_matrix[movie_idx]

                # 使用评分作为权重
                weighted_vector = movie_vector * (rating / 5.0)  # 归一化评分

                if user_profile_vector is None:
                    user_profile_vector = weighted_vector
                else:
                    user_profile_vector += weighted_vector

            # 归一化用户画像向量
            if user_profile_vector is not None:
                norm = np.linalg.norm(user_profile_vector)
                if norm > 0:
                    user_profile_vector = user_profile_vector / norm

        return user_profile_vector

    def content_based_recommend(self, user_id, top_n=10):
        """
        基于内容的推荐
        使用TF-IDF向量计算电影与用户画像的余弦相似度
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐（N不可以等于1）
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        if user_id not in self.ratings:
            return []

        if top_n <= 1:
            top_n = 2  # 确保N不等于1

        # 构建用户画像
        user_profile = self.build_user_profile(user_id)
        if user_profile is None:
            return []

        # 获取用户已评分的电影
        rated_movies = set(self.ratings[user_id].keys())

        # 计算候选电影与用户画像的余弦相似度
        movie_scores = []
        movie_ids = list(self.movies.keys())

        for idx, movie_id in enumerate(movie_ids):
            if movie_id in rated_movies:
                continue

            if SKLEARN_AVAILABLE:
                # 使用sklearn计算余弦相似度
                movie_vector = self.tfidf_matrix[idx].toarray().flatten()
                similarity = np.dot(user_profile, movie_vector) / (
                    np.linalg.norm(user_profile) * np.linalg.norm(movie_vector) + 1e-8
                )
            else:
                # 使用简化的余弦相似度计算
                movie_vector = self.tfidf_matrix[idx]
                similarity = np.dot(user_profile, movie_vector) / (
                    np.linalg.norm(user_profile) * np.linalg.norm(movie_vector) + 1e-8
                )

            if similarity > 0:
                # 生成推荐理由
                reason = self._generate_content_reason(user_id, movie_id, similarity)
                movie_scores.append((movie_id, round(similarity, 3), reason))

        # 按相似度排序并返回Top-N推荐
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        return movie_scores[:top_n]

    def _generate_content_reason(self, user_id, movie_id, similarity):
        """
        生成基于内容的推荐理由
        """
        user_ratings = self.ratings[user_id]
        user_high_rated_movies = [
            mid for mid, rating in user_ratings.items()
            if rating >= 4 and mid != movie_id
        ]

        if not user_high_rated_movies:
            return "基于您的观影偏好推荐"

        # 找到最相似的高评分电影
        best_similarity = 0
        best_movie_id = None

        for high_rated_movie_id in user_high_rated_movies:
            if SKLEARN_AVAILABLE:
                movie_idx = list(self.movies.keys()).index(movie_id)
                high_rated_idx = list(self.movies.keys()).index(high_rated_movie_id)

                movie_vec = self.tfidf_matrix[movie_idx].toarray().flatten()
                high_rated_vec = self.tfidf_matrix[high_rated_idx].toarray().flatten()

                sim = np.dot(movie_vec, high_rated_vec) / (
                    np.linalg.norm(movie_vec) * np.linalg.norm(high_rated_vec) + 1e-8
                )
            else:
                movie_idx = self.movie_id_to_idx[movie_id]
                high_rated_idx = self.movie_id_to_idx[high_rated_movie_id]

                movie_vec = self.tfidf_matrix[movie_idx]
                high_rated_vec = self.tfidf_matrix[high_rated_idx]

                sim = np.dot(movie_vec, high_rated_vec) / (
                    np.linalg.norm(movie_vec) * np.linalg.norm(high_rated_vec) + 1e-8
                )

            if sim > best_similarity:
                best_similarity = sim
                best_movie_id = high_rated_movie_id

        if best_movie_id and best_similarity > 0.3:  # 相似度阈值
            movie_title = self.movies[movie_id]['title']
            liked_title = self.movies[best_movie_id]['title']
            return f"基于内容特征，这部电影与您喜欢的《{liked_title}》很相似"

        return "基于您的观影偏好推荐"

    def hybrid_recommend(self, user_id, top_n=10, cf_weight=0.6):
        """
        混合推荐
        结合协同过滤和基于内容的推荐
        :param user_id: 用户ID
        :param top_n: 返回Top-N推荐
        :param cf_weight: 协同过滤权重（0-1）
        :return: 推荐列表 [(movie_id, score, reason), ...]
        """
        if user_id not in self.ratings:
            return []

        # 获取两种推荐结果
        cf_recommendations = self.item_based_recommend(user_id, top_n * 2)
        content_recommendations = self.content_based_recommend(user_id, top_n * 2)

        # 合并推荐结果
        movie_scores = {}
        movie_reasons = {}

        # 处理协同过滤推荐
        for movie_id, score, reason in cf_recommendations:
            movie_scores[movie_id] = score * cf_weight
            movie_reasons[movie_id] = reason

        # 处理基于内容的推荐
        content_weight = 1 - cf_weight
        for movie_id, score, reason in content_recommendations:
            if movie_id in movie_scores:
                movie_scores[movie_id] += score * content_weight
                movie_reasons[movie_id] = f"混合推荐：{movie_reasons[movie_id]}；{reason}"
            else:
                movie_scores[movie_id] = score * content_weight
                movie_reasons[movie_id] = f"混合推荐：{reason}"

        # 排序并返回Top-N推荐
        recommendations = []
        for movie_id, score in sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            recommendations.append((movie_id, round(score, 3), movie_reasons[movie_id]))

        return recommendations

    def get_movie_info(self, movie_id):
        """
        获取电影信息
        :param movie_id: 电影ID
        :return: 电影信息字典
        """
        return self.movies.get(movie_id, {})

    def display_recommendations(self, user_id, recommendations, algorithm_name="推荐算法"):
        """
        展示推荐结果
        :param user_id: 用户ID
        :param recommendations: 推荐列表
        :param algorithm_name: 算法名称
        """
        print(f"\n=== 用户{user_id}的{algorithm_name}推荐结果 ===")
        if not recommendations:
            print("暂无推荐")
            return

        for i, (movie_id, score, reason) in enumerate(recommendations, 1):
            movie_info = self.get_movie_info(movie_id)
            title = movie_info.get('title', f'电影{movie_id}')
            genres = movie_info.get('genres', [])
            director = movie_info.get('director', '未知')

            print(f"{i}. 《{title}》 - 推荐分数: {score}")
            print(f"   类型: {genres} | 导演: {director}")
            print(f"   推荐理由: {reason}")
            print()