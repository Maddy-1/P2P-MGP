import AssertDataType
import EditDataFrame
import EmptyValueCheck
import ModifyLoanStatus
import RemoveValues
import RobustZScore
from typing import List
import pandas as pd
import numpy as np
class DataClean:
    def __init__(self, data: pd.DataFrame, relevant_parameters: List[str]):
        self.data = data
        self.parameters = relevant_parameters

