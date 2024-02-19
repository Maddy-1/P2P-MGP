from typing import List

import pandas as pd


class ModelParameters:

    def __init__(self, data: pd.DataFrame, relevant_parameters: List[str], response_variable: str):
        self.data = data
        self.parameters = relevant_parameters
        self.response_variable = response_variable

    def __str__(self):
        return f' Y: {self.response_variable} \n X: {self.parameters}'

    def remove_parameter(self, removed_parameter: str):
        self.parameters = [param for param in self.parameters if param != removed_parameter]
        return ModelParameters(self.data, self.parameters, self.response_variable)

    def add_parameter(self, added_parameter: str):
        new_parameters = self.parameters + [added_parameter]
        return ModelParameters(self.data, new_parameters, self.response_variable)

    def update_data(self, new_data: pd.DataFrame):
        self.data = new_data
        return ModelParameters(self.data, self.parameters, self.response_variable)

    def get_all_variables(self):
        return self.parameters + [self.response_variable]
