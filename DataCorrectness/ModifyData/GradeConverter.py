import numpy as np
import pandas as pd


def _subgrade():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    numbers = ['1', '2', '3', '4', '5']
    lst = []
    for l in letters:
        for n in numbers:
            lst.append(l + n)

    return np.array(lst)


def grade_converter(ser: pd.Series) -> pd.Series:
    dct = dict(zip(np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G']), np.arange(1, 8)))
    return ser.map(dct)


def subgrade_converter(ser: pd.Series) -> pd.Series:
    dct = dict(zip(_subgrade(), np.arange(1, len(_subgrade()) + 1)))
    return ser.map(dct)
