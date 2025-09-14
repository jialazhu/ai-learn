import matplotlib

# matplotlib.use('Agg')  # 解决中文乱码问题
import os
import gensim  # 强大的NLP工具包
from gensim.models import word2vec, FastText, KeyedVectors, Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from gensim.scripts.glove2word2vec import glove2word2vec
from torch.distributions.constraints import positive

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

#训练数据
# sentences = [
#     ["natural", "language", "processing", "is", "fascinating"],
#     ["word", "embeddings", "capture", "semantic", "meaning"],
#     ["king", "queen", "man", "woman", "royalty"],
#     ["similar", "words", "have", "close", "vectors"],
#     ["machine", "learning", "models", "learn", "patterns"]
# ]
#
# word2ver_model = Word2Vec(
#     sentences = sentences,
#     vector_size = 100, #词向量维度
#     window = 5, #上下文窗口大小 模型在学习一个词向量会考虑前后各5个词的含义
#     min_count = 1, #最小词频 模型会忽略出现次数小于1的词
#     workers = 4, # 并行训练线程数 默认为1 建议根据CPU核心数设置
#     epochs = 50, # 训练轮数
# )
#
# fastText_model = FastText(
#     sentences = sentences,
#     vector_size = 100,
#     window = 5,
#     min_count = 1,
#     workers = 4,
#     epochs = 50,
#     sg = 1, # 1表示使用skip-gram模型 0表示使用CBOW模型
#     min_n = 3, # 最小子词长度 模型会将每个词切分成多个子词 最小子词长度为3
#     max_n = 6, # 最大子词长度 模型会将每个词切分成多个子词 最大子词长度为6
#     word_ngrams = 1, # 是否使用n-gram模型 0表示不使用 1表示使用
# )

# 将训练好的模型保存到本地文件
word2ver_model = KeyedVectors.load('skip-gram.model.bin')
fastText_model = KeyedVectors.load('fastText.model.bin')
#打印词汇量长度和词向量维度 和训练轮数
print(word2ver_model.wv.vectors.shape)
print(fastText_model.wv.vectors.shape)
print(word2ver_model.wv)
print(fastText_model.wv)
print(word2ver_model.epochs)
print(fastText_model.epochs)

#定义可视化核心函数
def visualize_vectors(model,words,method = 'pca'):
    vectors_model = model.wv if hasattr(model,'wv') else model #兼容性处理 如果有wv属性.就使用wv.否则用model
    vectors = [vectors_model[word] for word in words]
    vectors = np.array( vectors) #转换为num数组 语法规范
    #数据降维
    if method == 'pca':
        reducer = PCA(n_components=2) #降维为2维
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(5, len(words)-1)) #降维为2维 困惑度为30 迭代次数为300
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    result = reducer.fit_transform(vectors) #降维后的结果
    return result

def buld_plc(model , words):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('词向量可视化对比')
    pca_title = visualize_vectors(model, words, 'pca')
    ax1.scatter(*pca_title.T, alpha=0.5)
    for i, word in enumerate(words):
        ax1.annotate(word, (pca_title[i, 0], pca_title[i, 1]))
    ax1.set_title('PCA')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')

    tsne_title = visualize_vectors(model, words, 'tsne')
    ax2.scatter(*tsne_title.T, alpha=0.5)
    for i, word in enumerate(words):
        ax2.annotate(word, (tsne_title[i, 0], tsne_title[i, 1]))
    ax2.set_title('t-SNE')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#执行可视化和结果分析
words = ['king','man','woman','queen']
# buld_plc(word2ver_model,words)
# buld_plc(fastText_model,words)




def word_vector_similarity(model,word1,word2 , word3):
    try:
        vector_model = model.wv if hasattr(model, 'wv') else model
        result = vector_model.most_similar(positive=[word1,word2], negative=[word3],topn = 3)
        print(f"{word1} 和 {word2} 和 {word3} 的相似度: {result}")
    except Exception as e:
        print(e)
        print(f"计算 {word1} 和 {word2} 和 {word3} 相似度时出错")

word_vector_similarity(word2ver_model,'woman','king','man')

#使用预训练的GloVe模型
# glove2word2vec('glove.6B.100d.txt', 'glove.6B.100d.word2vec.txt')
# glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)
# glove_model.save('glove.6B.100d.word2vec.model.bin')
glove_model = KeyedVectors.load('glove.6B.100d.word2vec.model.bin')
word_vector_similarity(glove_model,'woman','king','man')

# buld_plc(glove_model,words)
words = ['龙井','绿茶','大红袍','红茶']
# china_model = glove2word2vec('vectors.txt', 'vectors.word2vec.txt')
# china_model = KeyedVectors.load_word2vec_format('vectors.word2vec.txt', binary=False)
# china_model.save('vectors.word2vec.model.bin')
china_model = KeyedVectors.load("vectors.word2vec.model.bin")
word_vector_similarity(china_model,words[0],words[1],words[2])
buld_plc(china_model,words)