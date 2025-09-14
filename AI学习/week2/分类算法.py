# 导入鸢尾花数据集加载函数
from sklearn.datasets import load_iris
# 导入数据集划分函数，用于将数据划分为训练集和测试集
from sklearn.model_selection import train_test_split
# 导入逻辑回归分类器
from sklearn.linear_model import LogisticRegression
# 导入支持向量机分类器
from sklearn.svm import SVC
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 导入K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入准确率评估函数
from sklearn.metrics import accuracy_score ,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 加载鸢尾花数据集
iris = load_iris()
# 提取特征数据和目标标签
X, y = iris.data, iris.target
# 将数据集划分为训练集和测试集，测试集占30%，使用分层抽样确保各类别比例一致，设置随机种子以保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)  # 分层抽样

# 初始化逻辑回归模型，设置最大迭代次数为200以确保收敛
log_reg = LogisticRegression(max_iter=200)
# 初始化K近邻分类器，设置邻居数为3
knn = KNeighborsClassifier(n_neighbors=3)
# 初始化支持向量机分类器，使用线性核函数
svm = SVC(kernel='linear')
# 初始化决策树分类器，设置随机种子以保证结果可复现
tree = DecisionTreeClassifier(random_state=42)

# 创建模型字典，便于统一训练和评估
models = {
    '逻辑回归': log_reg,
    'KNN': knn,
    'SVM': svm,
    '决策树': tree
}

# 遍历所有模型，进行训练、预测和评估
for name, model in models.items():
    # 使用训练数据拟合模型
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算预测准确率
    acc = accuracy_score(y_test, y_pred)
    # 打印模型名称及其准确率
    print(f"{name}的准确率是:{acc:.4f}")

# 定义新的待预测数据样本
new_data = [[3.2, 5.4, 5.6, 0.3]]
# 使用KNN模型对新数据进行预测
perd_new_data = knn.predict(new_data)
# 获取预测结果对应的类别名称
class_name = iris.target_names[perd_new_data[0]]
# 打印预测的类别
print(f"预测的类别是:{class_name}")

#使用matplotlib绘画坐标图

# 对测试集进行预测
y_pred = knn.predict(X_test)
# 计算预测准确率
acc = accuracy_score(y_test, y_pred)


import seaborn as sns

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制热力图
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels='x',
            yticklabels='y')
plt.title('分类混淆矩阵 (准确率={:.2f}%)'.format(acc*100))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()