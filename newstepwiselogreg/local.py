from DataCorrectness.DataClean import DataClean
from DataCorrectness.ModelParameters import ModelParameters
import pandas as pd

data = pd.read_csv("LoanStats2016Q2.csv", low_memory=False)
fact = data.loc[:, ~(data.isna().sum() / data.shape[0] > 0.1)].columns.tolist()
relevant_parameter_list = data.columns.to_list()
relevant_parameter_list.remove('loan_status')
model = ModelParameters(data, relevant_parameter_list, 'loan_status')
dc = DataClean(model, fact)
cleaned_data = dc.complete_data_clean()
new_par_list = cleaned_data.data.columns.to_list()
newdc = DataClean(cleaned_data, new_par_list).int_and_float_data().data
newdc.to_csv("EvenCleanerLoanStats2016Q2.csv")