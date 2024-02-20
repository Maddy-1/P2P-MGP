import pytest
import numpy as np
import pandas as pd
from DataCorrectness.ModelParameters import ModelParameters
from DataCorrectness.DataClean import DataClean


def missing_data():
    data = pd.DataFrame({'Factor 1': [np.nan] * 10 + [20], 'Factor 2': np.arange(0, 11), 'Factor 3': [np.nan] * 11,
                         'Factor 4': list(range(0, 10)) + [np.nan], 'Response': [1] * 11})
    return data


@pytest.fixture
def model_with_majority_missing_data():
    return ModelParameters(missing_data(), ['Factor 1', 'Factor 2'], 'Response')


@pytest.fixture
def model_with_missing_data():
    return ModelParameters(missing_data(), ['Factor 2', 'Factor 3'], 'Response')


@pytest.fixture
def model_with_minority_missing_data():
    return ModelParameters(missing_data(), ['Factor 2', 'Factor 4'], 'Response')


def test_relevant_data_method(model_with_majority_missing_data):
    expected_output = pd.DataFrame(
        {'Factor 1': [np.nan] * 10 + [20], 'Factor 2': np.arange(0, 11), 'Response': [1] * 11})
    dc = DataClean(model_with_majority_missing_data)
    dc.relevant_data()
    pd.testing.assert_frame_equal(dc.model.data, expected_output)


def test_removal_of_majority_missing_data(model_with_majority_missing_data):
    dc = DataClean(model_with_majority_missing_data)
    dc.relevant_data()
    dc.remove_empty_data()
    expected_output = pd.DataFrame({'Factor 2': np.arange(0, 11), 'Response': [1] * 11})

    pd.testing.assert_frame_equal(dc.model.data, expected_output)


def test_removal_of_missing_data(model_with_missing_data):
    dc = DataClean(model_with_missing_data)
    dc.relevant_data()
    dc.remove_empty_data()
    expected_output = pd.DataFrame({'Factor 2': np.arange(0, 11), 'Response': [1] * 11})

    pd.testing.assert_frame_equal(dc.model.data, expected_output)


def test_removal_of_minority_missing_data(model_with_minority_missing_data):
    dc = DataClean(model_with_minority_missing_data)
    dc.relevant_data()
    dc.remove_empty_data()
    expected_output = pd.DataFrame(
        {'Factor 2': np.arange(0, 10), 'Factor 4': [float(i) for i in range(0, 10)], 'Response': [1] * 10})

    pd.testing.assert_frame_equal(dc.model.data, expected_output)
