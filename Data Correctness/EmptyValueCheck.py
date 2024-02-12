import pandas as pd


def empty_check(ser: pd.Series):
    return ser.isna()
