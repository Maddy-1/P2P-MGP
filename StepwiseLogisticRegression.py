import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from mlxtend.feature_selection import SequentialFeatureSelector
from DataCorrectness.DataClean import DataClean
from DataCorrectness.ModelParameters import ModelParameters
# from GradeConverter import subgrade_converter
from DataCorrectness.ModifyData.ModifyLoanStatus import modify_loan_status
# from EmptyValueCheck import empty_check
# from RemoveValues import remove_rows
from typing import List

data14 = pd.read_csv("LoanStats2007_11.csv", low_memory=False)

def stepwiseLog(df: pd.DataFrame, outlier_detection_factors: List[str]):

  # get data
  stdsclr = StandardScaler()
  data = df.iloc[:, 1:]
  factors = data.columns.tolist()
  factors.remove('loan_status')
  model = ModelParameters(data, factors, 'loan_status')
  dc = DataClean(model, factors_to_check_outliers=outlier_detection_factors)
  #clean data
  cleaned_model = dc.complete_data_clean()
  #cleaned_model is type = ModelParameters. we set df to be the cleaned data
  df = cleaned_model.data

  y =modify_loan_status(df)["loan_status"]
  X = stdsclr.fit_transform(df.drop("loan_status", axis = 1))

  # Perform stepwise regression
  sfs = SequentialFeatureSelector(linear_model.LogisticRegression(),
                                  k_features=3,
                                  forward=True,
                                  scoring='accuracy',
                                  cv=None)
  selected_features = sfs.fit(X, y)

  # Create a dataframe with only the selected features
  new_model = ModelParameters(df, sfs.k_feature_names_, "loan_status")
  df_selected = DataClean(new_model).relevant_data()
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
