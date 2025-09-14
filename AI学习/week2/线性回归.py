
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import  fetch_california_housing


# 或者更安全的方式
import os
# 设置SSL证书路径，使用certifi提供的证书文件
# os.environ['SSL_CERT_FILE'] = certifi.where()

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['price1'] = housing.target
df.info()

X = df.drop("price1",axis=1) # 1 删除列 0 删除行
y = df['price1']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#线性回归
# 创建线性回归模型实例
model = LinearRegression()
# 使用训练数据拟合模型
model.fit(X_train, y_train)
# 输出模型的系数和截距
print(f"系数和截距为：{model.coef_},{model.intercept_}")
# 使用测试数据进行预测
y_predict = model.predict(X_test)
# 计算均方误差和R2得分
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
# 输出评估结果
print(f"均方误差为：{mse:.3f}, r2:{r2:.3f}")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 需先安装字体:ml-citation{ref="4" data="citationList"}


def plot_regression_results(y_true, y_pred, title):
    # 创建一个新的图形窗口，设置图形大小为10x6英寸
    plt.figure(figsize=(10, 6))

    # 计算实际值和预测值的最大最小值，用于绘制参考线
    max_val = max(y_true.max(), y_pred.max())  # 获取最大值
    min_val = min(y_true.min(), y_pred.min())  # 获取最小值
    
    # 绘制一条从左下到右上的虚线，表示理想预测情况（预测值等于实际值）
    plt.plot([min_val, max_val], [min_val, max_val],
             '--', color='gray', alpha=0.5, label='理想预测线')

    # 绘制散点图，横轴是实际值，纵轴是预测值
    plt.scatter(y_true, y_pred,
                c='royalblue', alpha=0.6,           # 设置点的颜色和透明度
                edgecolors='w', linewidth=0.5,      # 设置点的边框颜色和线宽
                label='样本点')                     # 设置图例标签

    # 计算R2得分并添加到图中
    r2 = r2_score(y_true, y_pred)                   # 计算R2得分
    plt.text(0, 1.1, f'$R^2 = {r2:.3f}$',       # 在图中指定位置添加文本
             transform=plt.gca().transAxes,         # 使用相对坐标系
             fontsize=12,                           # 设置字体大小
             bbox=dict(facecolor='white', alpha=0.8)) # 添加白色背景框

    # 设置图形标题和坐标轴标签
    plt.title(title, fontsize=14, pad=20)           # 设置标题及与图形的距离
    plt.xlabel('实际房价（标准化）', fontsize=12)    # 设置x轴标签
    plt.ylabel('预测房价（标准化）', fontsize=12)    # 设置y轴标签
    
    # 添加网格线和图例
    plt.grid(True, linestyle='--', alpha=0.3)       # 添加虚线网格
    plt.legend(loc='upper left')                    # 在左上角显示图例
    plt.tight_layout()                              # 自动调整子图参数，使图形更紧凑
    return plt                                      # 返回图形对象



plot = plot_regression_results(y_test, y_predict,
                             '加州房价预测结果')
plot.show()