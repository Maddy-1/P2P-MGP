import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 读取数据
df = pd.read_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/scaled_data.csv', encoding='ISO-8859-1')

# 提取特征和目标变量
X = df.drop(columns=['loan_status','funded_amnt'])
y = df['loan_status']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 创建RFE对象，指定模型和希望保留的特征数量
rfe = RFE(model, n_features_to_select=40)  # 选择希望保留的特征数量

# 对训练数据进行递归特征消除
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# 训练模型并预测
model.fit(X_train_rfe, y_train)
y_pred = model.predict(X_test_rfe)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 打印选择的特征
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)