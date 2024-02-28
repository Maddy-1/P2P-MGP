import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from mlxtend.feature_selection import SequentialFeatureSelector
from DataCorrectness.DataClean import DataClean as DC

# from GradeConverter import subgrade_converter
# from ModifyLoanStatus import modify_loan_status
# from EmptyValueCheck import empty_check
# from RemoveValues import remove_rows


data14 = pd.read_csv("LoanStats2007_11.csv", low_memory=False)

def stepwiseLog(df):

  # clean data
  df = DC.complete_data_clean(df.iloc[: , 1:])

  y = df["loan_status"]
  X = df.drop("loan_status", axis = 1)

  # Perform stepwise regression
  sfs = SequentialFeatureSelector(linear_model.LogisticRegression(),
                                  k_features=3,
                                  forward=True,
                                  scoring='accuracy',
                                  cv=None)
  selected_features = sfs.fit(X, y)

  # Create a dataframe with only the selected features
  df_selected = DC.relevant_data(df)
  df_selected = df.drop("loan_status", axis = 1)

  # Split the data into train and test sets
  X_train, X_test,\
      y_train, y_test = train_test_split(
          df_selected, y,
          test_size=0.3,
          random_state=42)
  
  # Fit a logistic regression model using the selected features
  logreg = linear_model.LogisticRegression()
  logreg.fit(X_train, y_train)
  
  # Make predictions using the test set
  y_pred = logreg.predict(X_test)
  
  # Evaluate the model performance
  return y_pred
