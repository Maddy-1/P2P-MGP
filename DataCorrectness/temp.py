from DataCorrectness.DataClean import DataClean
from DataCorrectness.ModelParameters import ModelParameters

import numpy as np
import pandas as pd

def missing_data():
    data = pd.DataFrame(
        {
            'Factor 1': [np.nan] * 10 + [20],
            'Factor 2': np.arange(0, 11),
            'Factor 3': [np.nan]*11,
            'Factor 4': list(range(0, 10)) + [np.nan],
            'Response': [1] * 11
        }
    )
    return data

mp = ModelParameters(missing_data(), ['Factor 1', 'Factor 2', 'Factor 4'], 'Response')
dc = DataClean(mp)
dc.relevant_data()
dc.remove_empty_data()
expected_output = pd.DataFrame(
    {
        'Factor 2': np.arange(0, 11),
        'Response': [1] * 11
    }
)
print(dc.model.data)
#pd.testing.assert_frame_equal(dc.model.data, expected_output)
ser = pd.Series([1,2,3,4])
print(pd.Series(np.zeros(ser.shape)))
