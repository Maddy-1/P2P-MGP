import pandas as pd

def percentage_to_decimal(ser: pd.Series) -> pd.Series:
    """
    dtype of series assumed to be string.
    example:
    input = pd.Series(['1.5%', ' 2%', '12%'])
    output = pd.Series([0.015, 0.02, 0.12])
    """
    newser = ser.str.slice(stop=-1).astype(float)/100
    return newser
