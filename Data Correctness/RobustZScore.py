import pandas as pd


def robust_zscore(ser: pd.Series):
    scaled_mad = 1.4826 * (ser - ser.median()).abs().median()
    z = (ser - ser.median()) / scaled_mad
    return z
