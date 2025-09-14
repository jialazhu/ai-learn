import pandas as pd
import numpy as np
import matplotlib

# matplotlib.use('Agg')  # 设置非交互式后端，避免图形显示问题
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

print("=== 多元线性回归作业 ===")
print("请完成TODO标记的部分\n")

# 加载加州房价数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
print("数据集信息:")
print(f"样本数: {X.shape[0]}")
print(f"特征数: {X.shape[1]}")
print(f"房价范围: {y.min():.2f} - {y.max():.2f}")

print(X.isnull().sum())
# 数据探索
print("\n数据探索:")
print("前5行数据:")
print(X.head())
print("\n基本统计信息:")
print(X.describe())
print("\n=== 基础多元线性回归 ===")

# TODO 1: 实现多元线性回归（约8行代码）
# 1. 划分训练集和测试集（比例8:2）
# 2. 训练线性回归模型
# 3. 在测试集上预测
# 4. 计算MSE和R²分数

# 你的代码：
# X_train, X_test, y_train, y_test = train_test_split(...)
# model = LinearRegression()
# model.fit(...)
# y_pred = model.predict(...)
# mse = mean_squared_error(...)
# r2 = r2_score(...)
# print(f"MSE: {mse:.4f}")
# print(f"R²: {r2:.4f}")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

print("\n=== 特征工程与改进 ===")

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# TODO 2: 使用标准化后的特征重新训练模型（约6行代码）
# 1. 划分标准化后的数据
# 2. 训练新模型
# 3. 预测并评估性能
# 4. 比较标准化前后的效果

# 你的代码：
# X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(...)
# model_scaled = LinearRegression()
# model_scaled.fit(...)
# y_pred_scaled = model_scaled.predict(...)
# mse_scaled = mean_squared_error(...)
# r2_scaled = r2_score(...)
# print(f"标准化后 MSE: {mse_scaled:.4f}")
# print(f"标准化后 R²: {r2_scaled:.4f}")

print(f"标准化后数据====")
# for i , e in enumerate(X_scaled):
#     print(i,e)

X_scaled_train,X_scaled_test,y_scaled_train,y_scaled_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled_train, y_scaled_train)
y_scaled_pred = model_scaled.predict(X_scaled_test)
mse_scaled = mean_squared_error(y_scaled_test, y_scaled_pred)
r2_scaled = r2_score(y_scaled_test, y_scaled_pred)
print(f"标准化后 MSE: {mse_scaled:.4f}")
print(f"标准化后 R²: {r2_scaled:.4f}")

print("\n=== 结果分析 ===")

# 特征重要性分析
def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    coefficients = model.coef_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    feature_importance = feature_importance.sort_values('coefficient', key=abs, ascending=False)
    print("特征重要性（按系数绝对值排序）:")
    print(feature_importance)
    return feature_importance

# 调用函数分析特征重要性
print("\n原始模型特征重要性:")
feature_importance_original = analyze_feature_importance(model, X.columns)

print("\n标准化模型特征重要性:")
feature_importance_scaled = analyze_feature_importance(model_scaled, X.columns)

# 可视化结果
def plot_results(y_true, y_pred, title):
    """绘制预测结果"""
    plt.figure(figsize=(10, 4))
    
    # 预测值vs真实值
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{title} - 预测值vs真实值')

    # 残差图
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title(f'{title} - 残差图')

    plt.tight_layout()
    plt.show()
    # 保存图片而不是显示
    plt.savefig(f'{title}_结果分析.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存为: {title}_结果分析.png")

print("\n")
plot_results(y_test,y_pred,"原始数据线性回归")
plot_results(y_scaled_test,y_scaled_pred,"标准化数据线性回归")

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

print(f"\n== 计算原始数据评估指标 ==")
mae , rmse =calculate_metrics(y_test, y_pred)
print(f"mae:{mae:.4f},rmse:{rmse:.4f}")

print(f"\n== 计算标准化数据评估指标 ==")
mae , rmse =calculate_metrics(y_scaled_test, y_scaled_pred)
print(f"mae:{mae:.4f},rmse:{rmse:.4f}")

print("\n作业完成")