import numpy as np
import pandas as pd


class ExpectedDTypes:
    def __init__(self):
        self.expected_dtype_dict = pd.read_csv("expected_dtypes.csv").iloc[:, 1:]
        self.integer_and_float_dtype_factors = self.int_and_float_df()

    def __str__(self):
        return str(self.expected_dtype_dict)

    def int_and_float_df(self):
        x = self.expected_dtype_dict
        fldf = x.loc[x['dtype'] == np.dtype('float64'), :]
        intdf = x.loc[x['dtype'] == np.dtype('int64'), :]
        df = pd.concat([fldf, intdf])
        return df.reset_index(drop=True)
