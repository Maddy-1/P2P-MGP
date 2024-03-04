from StepwiseLogisticRegression import stepwiseLog
import pandas as pd

data = pd.read_csv("LoanStats2007_11.csv", low_memory=False)

print(stepwiseLog(data, ['dti']))