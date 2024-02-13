import pandas as pd


def modify_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    co = df.loc[df['loan_status'] == 'Charged Off', :]
    fp = df.loc[df['loan_status'] == 'Fully Paid', :]
    newdf = pd.concat([co, fp]).reset_index(drop=True)
    return newdf.replace({'Fully Paid': 0, 'Charged Off': 1})
