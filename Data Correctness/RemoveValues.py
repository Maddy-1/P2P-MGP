import numpy as np
import pandas as pd


def remove_rows(removal_index_ser: pd.Series, data: pd.DataFrame):
    """

    :param removal_index_ser: series of trues and falses. If an element is true, then that row will be removed.
    :param data: df of data
    :return: df of data, but the rows corresponding to "True" are removed. the index is reset.
    example:
    input = pd.DataFrame({'name': ['a', 'b', 'c', 'd'], 'age': [1, 2, np.nan, 7]})
    output = pd.DataFrame({'name': ['a', 'b', 'd'], 'age': [1, 2,  7]})
    """
    return data.loc[~removal_index_ser, :].reset_index(drop=True)




