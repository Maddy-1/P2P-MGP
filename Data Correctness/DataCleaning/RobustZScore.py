import pandas as pd


def robust_zscore(ser: pd.Series):
    """1.4826 scaling factor is used to make it a consistent estimator for normal data.
    This is derived from 1/inversenormalcdf(3/4).
    The assumption of normality is helpful, but even for non-normal data, this is helpful to give an indication of how
     extreme a value is, in relation to the other data.
     A common use of this is if abs(z) > 10, then that point is an outlier."""
    scaled_mad = 1.4826 * (ser - ser.median()).abs().median()
    median_centered_data = ser - ser.median()
    zero_check = median_centered_data == 0

    z = median_centered_data / scaled_mad
    z = z.mask(zero_check==True, (ser - ser.median())/ser.std())
    return z
