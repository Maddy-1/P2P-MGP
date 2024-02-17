import pytest

import pandas as pd


@pytest.fixture
def empty_dataframe():
    return pd.DataFrame({})
