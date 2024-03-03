import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import pandas as pd


def model(xtrain, ytrain):
    logreg = LogisticRegression()
    logreg.fit(xtrain, ytrain)
    xtrain_model = sm.add_constant(xtrain)
    print(xtrain_model)
    logit_model = sm.Logit(ytrain, xtrain_model)
    print(logit_model)
    result = logit_model.fit()
    print(result)
    return None


