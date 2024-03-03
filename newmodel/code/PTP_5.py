import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score

df = pd.read_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/scaled_data.csv', encoding='ISO-8859-1')

# 提取特征和目标变量

X = df[['loan_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc',
       'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
       'revol_bal', 'total_acc', 'collections_12_mths_ex_med', 'tot_cur_bal',
       'total_bal_il', 'open_rv_24m', 'max_bal_bc', 'total_rev_hi_lim',
       'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths',
       'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
       'mort_acc', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
       'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl', 'num_rev_accts',
       'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_op_past_12m',
       'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 'tot_hi_cred_lim',
       'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
       'term', 'sub_grade', 'home_ownership', 'verification_status']]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()


#undersampler = RandomUnderSampler(random_state=42)
#X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

#X_train_undersampled_sm = sm.add_constant(X_train_undersampled)  # 添加截距项
#logit_model_undersampled = sm.Logit(y_train_undersampled, X_train_undersampled_sm)
#result = logit_model_undersampled.fit()

#X_test_sm = sm.add_constant(X_test)  # 添加截距项
#y_pred_undersampled = result.predict(X_test_sm)
#y_pred_binary = (y_pred_undersampled > 0.5).astype(int)  # 将概率转换为二进制预测

# 计算混淆矩阵
#conf_matrix = confusion_matrix(y_test, y_pred_binary)
#print("Confusion Matrix:")
#print(conf_matrix)

# 计算精度
#accuracy = accuracy_score(y_test, y_pred_binary)
#print("\nAccuracy:", accuracy)

#auc = roc_auc_score(y_test, y_pred_undersampled)
#print("\nAUC:", auc)

# 计算 F1 分数
#f1 = f1_score(y_test, y_pred_binary)
#print("\nF1 Score:", f1)

# 打印分类报告
#classification_rep = classification_report(y_test, y_pred_binary)
#print("\nClassification Report:")
#print(classification_rep)

# 提取摘要表格数据
#summary_data = result.summary().tables[1].data[1:]  # 去掉表头

# 提取列名
#column_names = [cell.strip() for cell in result.summary().tables[1].data[0]]

# 将摘要信息保存为表格
#summary_df = pd.DataFrame(summary_data, columns=column_names)

# 将表格保存为CSV文件
#summary_df.to_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/model_summary_1.csv', index=False)

