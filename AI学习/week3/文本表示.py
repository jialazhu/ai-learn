import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# 实现tf-idf的核心工具  实现词袋模型
corpus = [
    '小李 今天 很帅',
    '小李 心情 不好',
    '小李 心情 真好',
    '小红 心情 不好'
]
#词袋模型
co = CountVectorizer()
x_co =co.fit_transform(corpus)
print(co.get_feature_names_out() )
print(x_co)
print(x_co.toarray())

#tf-idf
tf =TfidfVectorizer()
tiidf = tf.fit_transform(corpus)
print(tf.get_feature_names_out())
print(tiidf)
print(tiidf.toarray().round(2))