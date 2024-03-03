import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


df = pd.read_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/scaled_data.csv', encoding='ISO-8859-1')

# 提取特征和目标变量
X = df.drop(columns=['loan_status'])
y = df['loan_status']

features_to_exclude =['term','sub_grade','emp_length','home_ownership','verification_status','application_type']

X_filtered = X.drop(columns=features_to_exclude)
X_sm = sm.add_constant(X_filtered)

# 计算VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_sm.columns
vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]

# 打印VIF数据
#print(vif_data)

vif_data.to_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/vif_data.csv', index=False)