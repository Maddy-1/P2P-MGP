from DataCorrectness.ModelParameters import ModelParameters
from DataCorrectness.ModifyData.EditDataFrame import get_specific_columns, drop_specific_columns
from DataCorrectness.DataCleaning.EmptyValueCheck import empty_check
from DataCorrectness.DataCleaning.RemoveValues import remove_rows
from DataCorrectness.ModifyData.ChangePercentageToDecimal import percentage_to_decimal
from DataCorrectness.ModifyData.ModifyTermToInt import modify_term_to_int
from DataCorrectness.ModifyData.GradeConverter import grade_converter, subgrade_converter
from DataCorrectness.ModifyData.ChangeDateToYear import change_dtype_to_datetime


class DataClean:
    def __init__(self, model: ModelParameters, check_empty_values: bool = True, change_irregular_dtypes: bool = True):
        """

        :param model: ModelParameters class, containing the data and parameters of interest
        :param check_empty_values: bool. If True, then it will check and remove empty columns and/or values.
        :param change_irregular_dtypes: bool. If True, then it'll change percentages to floats (decimals), grades and
        subgrades to ordinal data, dates to datetime class and change the term (which is given as a string) to int.
        """
        self.model = model
        self.empty_check = check_empty_values
        self.change_irregular_dtypes = change_irregular_dtypes

    def __str__(self):
        return f"{self.model}\n Check Empty Values: {self.empty_check}"

    def complete_data_clean(self):
        self.relevant_data()
        if self.empty_check:
            self.remove_empty_data()
        if self.change_irregular_dtypes:
            self.convert_irregular_dtype()

    def relevant_data(self):
        relevant_data = get_specific_columns(self.model.data, self.model.get_all_variables())
        self.model = ModelParameters(relevant_data, self.model.parameters, self.model.response_variable)

    def remove_empty_data(self):
        for col_name in self.model.parameters:
            empty_ser = empty_check(self.model.data[col_name])
            if empty_ser.all() or empty_ser.sum() / len(empty_ser) > 0.1:

                editted_df = drop_specific_columns(self.model.data, [col_name])
                editted_parameters = self.model.parameters
                editted_parameters.remove(col_name)
                self.model = ModelParameters(editted_df, editted_parameters, self.model.response_variable)
            else:
                editted_df = remove_rows(empty_ser, self.model.data)
                self.model = ModelParameters(editted_df, self.model.parameters, self.model.response_variable)

    def check_type(self):
        return

    def check_outlier(self):
        return

    def convert_irregular_dtype(self):

        factors = ['int_rate', 'revol_util', 'term', 'grade', 'sub_grade', 'issue_d', 'earliest_cr_line',
                   'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
        funct_lst = [percentage_to_decimal, percentage_to_decimal, modify_term_to_int, grade_converter,
                     subgrade_converter] + [change_dtype_to_datetime] * 5

        dtype_dct = dict(zip(factors, funct_lst))

        for factor in self.model.parameters:
            if factor in factors:
                self.model.data[factor] = dtype_dct[factor](self.model.data[factor])