# 实践任务：带大家探索词向量的世界，模型训练，信息打印，结果可视化，加载模型...
# 1.环境的导入
import matplotlib

from AI学习.week4.词向量 import word2ver_model

# matplotlib.use('TkAgg')  # 解决中文显示问题
import os  # 导入操作系统接口的模块，用于处理文件路径
import gensim  # 强大的NLP工具包
# Word2Vec 词向量模型, FastText 需要训练作对比的模型, KeyedVectors，用于帮助我们加载预训练好的模型Glove
from gensim.models import Word2Vec, FastText, KeyedVectors
# 降维 PCA和t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 2.准备训练数据
sentences = [
    ["natural", "language", "processing", "is", "fascinating"],
    ["word", "embeddings", "capture", "semantic", "meaning"],
    ["king", "queen", "man", "woman", "royalty"],
    ["similar", "words", "have", "close", "vectors"],
    ["machine", "learning", "models", "learn", "patterns"]
]

# 3.训练模型
# word2vec_model = Word2Vec(
#     sentences=sentences,  # 拟定的数据导入
#     vector_size=100,  # 词向量维度
#     window=5,  # 设置上下文窗口的大小，模型在学习一个词向量的时候，回去考虑前后各五个词的语境
#     min_count=1,  # 最小词频，至少要出现一次
#     workers=4,  # 并行线程，同时使用4个CPU核心进行并行处理，加快训练速度
#     epochs=50  # 训练轮次，所以训练50轮
# )
#
# fasttext_model = FastText(
#     sentences=sentences,  # 拟定的数据导入
#     vector_size=100,  # 词向量维度
#     window=5,  # 设置上下文窗口的大小，模型在学习一个词向量的时候，回去考虑前后各五个词的语境
#     min_count=1,  # 最小词频，至少要出现一次
#     workers=4,  # 并行线程，同时使用4个CPU核心进行并行处理，加快训练速度
#     epochs=50,  # 训练轮次，所以训练50轮
#     min_n=3,  # fasttext模型特有的参数，子词的最小长度
#     max_n=6  # fasttext模型特有的参数，子词的最大长度，用于去学习词内部长度在3-6之间的子词信息
# )
#
# # 对于训练好的模型，进行相应的保存（本地），方便下次使用
# word2vec_model.save('word2vec.model.bin')
# fasttext_model.save('fasttext.model.bin')
word2vec_model = KeyedVectors.load('skip-gram.model.bin')
fasttext_model = KeyedVectors.load('fasttext.model.bin')

print(f"\n打印Word2Vec模型信息:")  # 需要打印词汇量，向量维度，训练轮次
print(f" - 词汇量: {len(word2vec_model.wv)}")  # .wx是一个属性，表示存储模型所有的词向量和词汇表信息
print(f" - 向量维度: {word2vec_model.vector_size}")
print(f" - 训练轮次: {word2vec_model.epochs}")

print(f"\nFastText模型信息:")
print(f" - 词汇量: {len(fasttext_model.wv)}")
print(f" - 向量维度: {fasttext_model.vector_size}")
print(f" - 子词长度范围: {fasttext_model.wv.min_n}-{fasttext_model.wv.max_n}")


# 现在的模型非常的基础，使用数据集也非常的基础，虽然已经训练好了，但是没有经过任何的优化，所以这模型的效果必然是非常差！
# 4.词向量可视化
# 定义可视化的核心函数 visualize_vectors
def visualize_vectors(model, words, method='pca'):  # 这里的参数 是形式参数
    vectors_model = model.wv if hasattr(model, 'wv') else model  # 关键的兼容性处理，如果模型对象有wv属性，那么就使用wv，返回True，否则使用model，返回False
    vectors = [vectors_model[word] for word in words]  # 通过遍历模型对象，获取到词向量，使用列表的推导式，将词向量存储到列表中
    vectors = np.array(vectors)  # 转换为numpy数组，为了后续进行数学运算，是一个语法上的规范，是标准格式

    # # 降维到2D 还需要对降维进行选择，这里使用的是PCA和t-SNE两种方法
    if method == 'pca':
        # PCA降维（线性，保持全局结构）
        reducer = PCA(n_components=2)  # 降维到2D，n_components=2，表示降维到2D
        title = 'Word Vector Visualization (PCA)'
    else:
        # t-SNE降维（非线性，保持局部结构）
        reducer = TSNE(n_components=2, perplexity=min(5, len(words) - 1),
                       learning_rate=200, random_state=42)
        title = 'Word Vector Visualization (t-SNE)'
    result = reducer.fit_transform(vectors)  # 使用降维模型，对词向量进行降维处理，返回降维后的结果
    return result, title  # 返回降维后的结果，和标题


# 执行可视化和结果分析
test_words = ['king', 'queen', 'man', 'woman', 'royalty']  # 定义需要可视化的词
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("词向量可视化对比： PCA和t-SNE", fontsize=16)  # 设置标题

# PCA降维
result_pca, _ = visualize_vectors(word2vec_model, test_words, method='pca')
ax1.scatter(result_pca[:, 0], result_pca[:, 1])  # 绘制散点图
for i, word in enumerate(test_words):
    ax1.annotate(word, xy=(result_pca[i, 0], result_pca[i, 1]), fontsize=14)
ax1.set_title('PCA降维 (保持全局结构)')

# t_SNE降维
result_tsne, _ = visualize_vectors(word2vec_model, test_words, method='t-sne')
ax2.scatter(result_tsne[:, 0], result_tsne[:, 1])  # 绘制散点图
for i, word in enumerate(test_words):
    ax2.annotate(word, xy=(result_tsne[i, 0], result_tsne[i, 1]), fontsize=14)
ax2.set_title('tSNE降维 (保持局部结构)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# 做类比性测试
def word_vector_similarity(model, word1, word2, word3):
    try:
        vector_model = model.wv if hasattr(model, 'wv') else model
        result = vector_model.most_similar(positive=[word2, word3], negative=[word1], topn=3)
        print(f"打印结果：{word2}和{word3}比{word1}更接近的词是：{result}")
    except Exception as e:
        print(f"类比失败：{e}")


word_vector_similarity(word2vec_model, test_words[2], test_words[3], test_words[0])

# TODO
# 5.使用预训练的Glove模型
# glove2word2vec('glove.6B.100d.txt', 'glove.6B.100d.word2vec.txt')
# glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)
# glove_model.save('glove.6B.100d.word2vec.model.bin')
glove_model = KeyedVectors.load('glove.6B.100d.word2vec.model.bin')
word_vector_similarity(glove_model, test_words[2], test_words[3], test_words[0])

# 6.做结果分析对比以及可视化
glove_fig, (glove_ax1, glove_ax2) = plt.subplots(1, 2, figsize=(16, 6))
glove_fig.suptitle("GLOVE词向量可视化对比： PCA和t-SNE", fontsize=16)  # 设置标题

# PCA降维
result_pca, _ = visualize_vectors(glove_model, test_words, method='pca')
glove_ax1.scatter(result_pca[:, 0], result_pca[:, 1])  # 绘制散点图
for i, word in enumerate(test_words):
    glove_ax1.annotate(word, xy=(result_pca[i, 0], result_pca[i, 1]), fontsize=14)
glove_ax1.set_title('PCA降维 (保持全局结构)')

# t_SNE降维
result_tsne, _ = visualize_vectors(glove_model, test_words, method='t-sne')
glove_ax2.scatter(result_tsne[:, 0], result_tsne[:, 1])  # 绘制散点图
for i, word in enumerate(test_words):
    glove_ax2.annotate(word, xy=(result_tsne[i, 0], result_tsne[i, 1]), fontsize=14)
glove_ax2.set_title('tSNE降维 (保持局部结构)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()