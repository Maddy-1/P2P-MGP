from DataCorrectness.ModelParameters import ModelParameters
from DataCorrectness.ModifyData.EditDataFrame import get_specific_columns, drop_specific_columns
from DataCorrectness.DataCleaning.EmptyValueCheck import empty_check
from DataCorrectness.DataCleaning.RemoveValues import remove_rows


class DataClean:
    def __init__(self, model: ModelParameters, check_empty_values: bool = True):
        """

        :param model: ModelParameters class, containing the data and parameters of interest
        :param check_empty_values: bool. If True, then it will check and remove empty columns and/or values.
        """
        self.model = model
        self.empty_check = check_empty_values

    def __str__(self):
        return f"{self.model}\n Check Empty Values: {self.empty_check}"

    def complete_data_clean(self):
        self.relevant_data()
        self.remove_empty_data()

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

    def convert_dtype(self):
        return