import pandas as pd


def data_type_check(ser: pd.Series, datatype):
    """

    :param ser: series of data
    :param datatype: dtype, e.g int, float, str
    :return: bool series
    e.g x = pd.Series([4, "hi", np.nan, 7])
    data_type_check(x, int) = pd.Series([True, False, False, True])
    """
    return ser.apply(type) == datatype
