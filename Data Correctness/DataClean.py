from typing import List
import pandas as pd


class DataClean:
    def __init__(self, data: pd.DataFrame, relevant_parameters: List[str]):
        self.data = data
        self.parameters = relevant_parameters

