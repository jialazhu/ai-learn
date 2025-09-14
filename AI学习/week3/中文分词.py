import warnings
# 忽略告警信息展示
warnings.filterwarnings("ignore")
# 专门用于中文分词
import jieba
jieba.setLogLevel(20) # 只显示报错信息

import jieba.analyse

# content = '我来到北京市清华大学'
content = '我上周去试驾了问界M9,鸿蒙座舱太好用了'


set_list = jieba.cut(content, cut_all=False)
print(f"精确模式:","/".join(set_list))
set_all = jieba.cut(content, cut_all=True)
print(f"全模式：","/".join(set_all))
set_search = jieba.cut_for_search(content)
print(f"搜索引擎模式",list(set_search))

jieba.add_word("问界M9")
jieba.add_word("鸿蒙座舱")
print("添加自定义词:","/".join(jieba.cut(content)))

words = "我的天呢.这个电影是真的好看呢.还要继续看"
stop_word = ['我','.','真','的','呢']
def stop_word_set(word_list,stop_word_list):
    return [word for word in word_list if word not in stop_word_list]
print("原始分词:",list(jieba.cut(words)))
print("去除停用词:",list(stop_word_set(jieba.cut(words),stop_word)))
