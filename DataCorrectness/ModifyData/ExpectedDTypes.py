import pandas as pd


class ExpectedDTypes:
    def __init__(self):
        self.expected_dtype_dict = pd.read_csv("expected_dtypes.csv").iloc[:, 1:]

    def __str__(self):
        return str(self.expected_dtype_dict)
