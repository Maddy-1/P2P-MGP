import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import PTP_5
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix

logit_model_lasso = sm.Logit(PTP_5.y_train, PTP_5.X_train_sm)
result_lasso = logit_model_lasso.fit_regularized(method='l1', alpha=0.1)

X_test_sm = sm.add_constant(PTP_5.X_test)  # 添加截距项
y_pred = result_lasso.predict(X_test_sm)
y_pred_binary = (y_pred > 0.73).astype(int)  # 将概率转换为二进制预测

# 计算混淆矩阵
conf_matrix = confusion_matrix(PTP_5.y_test, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# 计算精度
accuracy = accuracy_score(PTP_5.y_test, y_pred_binary)
print("\nAccuracy:", accuracy)

auc = roc_auc_score(PTP_5.y_test, y_pred)
print("\nAUC:", auc)

# 计算 F1 分数
f1 = f1_score(PTP_5.y_test, y_pred_binary)
print("\nF1 Score:", f1)

# 打印分类报告
classification_rep = classification_report(PTP_5.y_test, y_pred_binary)
print("\nClassification Report:")
print(classification_rep)
