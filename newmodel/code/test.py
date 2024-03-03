import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score
import PTP_5

y_train_pred = PTP_5.result.predict(PTP_5.X_train_sm)
y_train_pred_binary = (y_train_pred > 0.73).astype(int)

conf_matrix = confusion_matrix(PTP_5.y_train, y_train_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# 计算精度
accuracy = accuracy_score(PTP_5.y_train, y_train_pred_binary)
print("\nAccuracy:", accuracy)

auc = roc_auc_score(PTP_5.y_train, y_train_pred)
print("\nAUC:", auc)

# 计算 F1 分数
f1 = f1_score(PTP_5.y_train, y_train_pred_binary)
print("\nF1 Score:", f1)

classification_rep = classification_report(PTP_5.y_train, y_train_pred_binary)
print("\nClassification Report:")
print(classification_rep)

#undersampler = RandomUnderSampler(random_state=42)
#X_test_pred_undersampled, y_test_pred_undersampled = undersampler.fit_resample(PTP_5.X_test, PTP_5.y_test)

X_test_sm = sm.add_constant(PTP_5.X_test)  # 添加截距项
y_test_pred = PTP_5.result.predict(X_test_sm)
y_test_pred_binary = (y_test_pred > 0.73).astype(int)  # 将概率转换为二进制预测

# 计算混淆矩阵
conf_matrix = confusion_matrix(PTP_5.y_test, y_test_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# 计算精度
accuracy = accuracy_score(PTP_5.y_test, y_test_pred_binary)
print("\nAccuracy:", accuracy)

auc = roc_auc_score(PTP_5.y_test, y_test_pred)
print("\nAUC:", auc)

# 计算 F1 分数
f1 = f1_score(PTP_5.y_test, y_test_pred_binary)
print("\nF1 Score:", f1)

classification_rep = classification_report(PTP_5.y_test, y_test_pred_binary)
print("\nClassification Report:")
print(classification_rep)

