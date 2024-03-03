import numpy as np
import pandas as pd


def modify_term_to_int(ser: pd.Series) -> pd.Series:
    """
    Series must not have nan values.
    examples:
    Input = pd.Series(['36 months', '60 months'])
    Output = pd.Series([36, 60])
    """

    if ser.dtype == np.int64:
        return ser
    return ser.str.extract('(\d+)').astype(int)

