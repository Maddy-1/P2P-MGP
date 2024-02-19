import pandas as pd
from typing import List
def get_specific_columns(df: pd.DataFrame, col_names: List[str]):
    new_df = df.loc[:, col_names]
    return new_df

def drop_specific_columns(df: pd.DataFrame, col_names: List[str]):
    new_df = df.drop(columns = col_names)
    return new_df