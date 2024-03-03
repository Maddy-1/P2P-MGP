import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/scaled_data.csv', encoding='ISO-8859-1')

# 假设你的数据框是 df
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

# 创建随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 拟合模型
model.fit(X, y)

# 获取特征重要性
feature_names = X.columns
feature_importance = model.feature_importances_

# 创建包含特征重要性的数据框
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# 将特征重要性导出到CSV文件
feature_importance_df.to_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/feature_importance.csv', index=False)