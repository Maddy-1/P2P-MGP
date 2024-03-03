import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df = pd.read_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/PTP_dataset_4.csv', encoding='ISO-8859-1')

X = df.drop(columns=['loan_status','il_util'])
y = df['loan_status']

features_to_exclude =['term','sub_grade','emp_length','home_ownership','verification_status','application_type']

X_filtered = X.drop(columns=features_to_exclude)

# Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_filtered)

df_result = pd.concat([pd.DataFrame(X_scaled, columns=X_filtered.columns), X.drop(columns=X_filtered.columns)], axis=1)
df_result['loan_status'] = df['loan_status']
# 保存到新的CSV文件
df_result.to_csv('C:/Users/13097/Desktop/lending_club/DAM Project 3/Data/scaled_data.csv', index=False)