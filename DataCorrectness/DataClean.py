from typing import List

import numpy as np
import pandas as pd

from DataCorrectness.DataCleaning.EmptyValueCheck import empty_check
from DataCorrectness.DataCleaning.RemoveValues import remove_rows
from DataCorrectness.DataCleaning.RobustZScore import robust_zscore
from DataCorrectness.ModelParameters import ModelParameters
from DataCorrectness.ModifyData.ChangeDateToYear import change_dtype_to_datetime
from DataCorrectness.ModifyData.ChangePercentageToDecimal import percentage_to_decimal
from DataCorrectness.ModifyData.EditDataFrame import get_specific_columns, drop_specific_columns
from DataCorrectness.ModifyData.GradeConverter import grade_converter, subgrade_converter
from DataCorrectness.ModifyData.ModifyLoanStatus import modify_loan_status
from DataCorrectness.ModifyData.ModifyTermToInt import modify_term_to_int
from DataCorrectness.ModifyData.StringsToBinary import initial_list_status_to_binary, yesno_to_binary, \
    application_type_to_binary, disbursement_method_to_binary


class DataClean:
    def __init__(self, model: ModelParameters, factors_to_check_outliers: List[str] = None,
                 check_empty_values: bool = True, change_irregular_dtypes: bool = True, outlier_check: bool = True,
                 max_zscore_tol: float = 10, check_types: bool = True):
        """

        :param model: ModelParameters class, containing the data and parameters of interest
        :param check_empty_values: bool. If True, then it will check and remove empty columns and/or values.
        :param change_irregular_dtypes: bool. If True, then it'll change percentages to floats (decimals), grades and
        subgrades to ordinal data, dates to datetime class and change the term (which is given as a string) to int.
        """
        self.model = model
        self.outlier_factors = factors_to_check_outliers
        self.empty_check = check_empty_values
        self.change_irregular_dtypes = change_irregular_dtypes
        self.outlier_check = outlier_check
        self.max_zscore_tolerance = max_zscore_tol
        self.check_types = check_types
        self.outlier_df = pd.DataFrame({})

    def __str__(self):
        return f"{self.model}\n Check Empty Values: {self.empty_check}\n Check Irregular Dtypes: {self.change_irregular_dtypes}"

    def complete_data_clean(self):
        self.relevant_data()
        self.model.data = modify_loan_status(self.model.data)
        if self.empty_check:
            self.remove_empty_data()
        if self.change_irregular_dtypes:
            self.convert_irregular_dtype()
        if self.outlier_check:
            self.check_outlier(max_zscore_tol=self.max_zscore_tolerance)
        if self.check_types:
            # self.check_type()
            pass

        return ModelParameters(self.model.data, self.model.parameters, self.model.response_variable)

    def relevant_data(self):
        relevant_data = get_specific_columns(self.model.data, self.model.get_all_variables())
        self.model = ModelParameters(relevant_data, self.model.parameters, self.model.response_variable)

    def int_and_float_data(self):
        """should be used postclean"""
        newdf = self.model.data
        for i in newdf.columns:
            dtypecheck = True
            dt = newdf[i].dtype

            if dt != np.int64 and dt != np.float64:
                dtypecheck = False

            if dtypecheck is False:
                newdf.pop(i)

        return ModelParameters(newdf, self.model.parameters, self.model.response_variable)

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

    # def check_type(self, remove_incorrect_type=False):
    #     expected_dtype = ExpectedDTypes().expected_dtype_dict
    #     for factor in self.model.parameters:
    #         if factor in expected_dtype.keys():
    #             dtype_check = data_type_check(self.model.data[factor], expected_dtype[factor])
    #             if remove_incorrect_type:
    #                 self.model.data = remove_rows(~dtype_check, self.model.data)
    #             if dtype_check.all() is False:
    #                 print(f"{100 * dtype_check.sum() / dtype_check.count()}% of Factor {factor} is the wrong type")

    def check_outlier(self, max_zscore_tol: float = 10, relevant_data_checked=True):
        if relevant_data_checked is False:
            self.relevant_data()
        if self.outlier_factors is None:
            return None

        for factor in self.outlier_factors:
            cond1 = (self.model.data[factor].dtype == np.dtype(float))
            cond2 = (self.model.data[factor].dtype == np.dtype(int))
            if isinstance(cond1, pd.Series) or isinstance(cond2, pd.Series):
                cond1 = (self.model.data[factor].dtype == np.dtype(float)).all()
                cond2 = (self.model.data[factor].dtype == np.dtype(int)).all()

            if cond1 or cond2:
                zscore_ser = robust_zscore(self.model.data[factor]).abs()
                removal_bool_ser = zscore_ser > max_zscore_tol
                self.outlier_df = pd.concat([self.outlier_df, self.model.data.loc[removal_bool_ser, :]])
                self.model.data = remove_rows(removal_bool_ser, self.model.data)

    def convert_irregular_dtype(self):

        factors = ['int_rate', 'revol_util', 'term', 'grade', 'sub_grade', 'issue_d', 'earliest_cr_line',
                   'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'initial_list_status', 'pymnt_plan',
                   'hardship_flag', 'application_type', 'disbursement_method']
        funct_lst = [percentage_to_decimal, percentage_to_decimal, modify_term_to_int, grade_converter,
                     subgrade_converter] + [change_dtype_to_datetime] * 5 + [initial_list_status_to_binary,
                                                                             yesno_to_binary, yesno_to_binary,
                                                                             application_type_to_binary,
                                                                             disbursement_method_to_binary]

        dtype_dct = dict(zip(factors, funct_lst))

        for factor in self.model.parameters:
            if factor in factors:
                self.model.data[factor] = dtype_dct[factor](self.model.data[factor])
