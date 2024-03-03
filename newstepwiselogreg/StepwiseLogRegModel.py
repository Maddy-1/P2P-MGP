import statsmodels.api as sm
import pandas as pd
from sklearn import linear_model

def model(xtrain, ytrain):
    logreg = linear_model.LogisticRegression()
    logreg.fit(xtrain, ytrain)
    xtrain_model = sm.add_constant(xtrain)
    print(xtrain_model)
    logit_model = sm.Logit(ytrain, xtrain_model)
    print(logit_model)
    result = logit_model.fit()
    print(result)
    return None


x = pd.read_csv('EvenCleanerLoanStats2016Q2.csv')
y = x['loan_status']
for i in ['loan_status', 'member_id', 'mths_since_last_record', 'dti_joint']:
    x.pop(i)

model(x, y)