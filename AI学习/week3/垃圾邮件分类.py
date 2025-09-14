import warnings
warnings.filterwarnings("ignore")

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

jieba.setLogLevel(20)

data = {
    'text': [
        'Hi, you have won a lottery, click here to claim',  # 垃圾邮件
        'Lunch meeting tomorrow at 12pm?',  # 正常邮件
        'URGENT! Your account has been compromised!',  # 垃圾邮件
        'Can you please review the document?',  # 正常邮件
        'Free Viagra, cheap Cialis, order now!',  # 垃圾邮件
        '恭喜您中奖！请点击链接领取大奖！',  # 垃圾邮件
        '你好，下周的会议纪要发你了。',  # 正常邮件
        '【xx贷】急用钱？马上到账！',  # 垃圾邮件
        '周末一起去打球吗？',  # 正常邮件
        '发票，代开，增值税。'  # 垃圾邮件
    ],
    'label': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1: 垃圾邮件, 0: 正常邮件
}

df = pd.DataFrame(data)
print(df)

def preprocess_text(text):
    text = text.lower()
    words = jieba.lcut(text)
    return " ".join(words)


df['proce'] = df['text'].apply(preprocess_text)
print(df)

tfidf = TfidfVectorizer(max_features=30)
x = tfidf.fit_transform(df['proce'])
y = df['label']

model = MultinomialNB()
model.fit(x,y)

mail = [
    '明天记得来公司',  # 垃圾邮件
    'Lunch meeting 12pm?',  # 正常邮件
    'URGENT! compromised!',  # 垃圾邮件
    'Free Viagra, cheap Cialis, order now!',  # 垃圾邮件
    '骗子',  # 垃圾邮件
    '明天天气很好,记得带防晒衣',  # 正常邮件
]

for text in mail:
    textx = preprocess_text(text)
    xx = tfidf.transform([textx])
    pred = model.predict(xx)
    test = "垃圾邮件" if pred == 1  else "正常邮件"
    print(text, test)