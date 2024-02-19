import pandas as pd


def change_issue_date_to_year(df: pd.DataFrame) -> pd.DataFrame:
    newdf = df.copy()
    newdf['issue_year'] = pd.to_datetime(newdf['issue_d']).dt.year
    return newdf


def change_dtype_to_datetime(ser: pd.Series) -> pd.Series:
    newser = pd.to_datetime(ser).dt.strftime('%m-%Y')
    return newser

x = pd.Series(
    ['SEP-2014', 'NOV-2002']
)

