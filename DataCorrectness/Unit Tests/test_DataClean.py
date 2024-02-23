import numpy as np
import pandas as pd
import pytest

from DataCorrectness.DataClean import DataClean
from DataCorrectness.ModelParameters import ModelParameters


def missing_data():
    data = pd.DataFrame({'Factor 1': [np.nan] * 10 + [20], 'Factor 2': np.arange(0, 11), 'Factor 3': [np.nan] * 11,
                         'Factor 4': list(range(0, 10)) + [np.nan], 'Response': [1] * 11})
    return data


def outlier_data():
    data = pd.DataFrame({'Factor 1': np.arange(0, 6), 'Factor 2': [0.9, 1, 100, 1.1, 1, 0.9], 'Response': [1] * 6})
    return data


@pytest.fixture
def model_with_outlier_data():
    return ModelParameters(outlier_data(), ['Factor 1', 'Factor 2'], 'Response')


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


def test_outlier_detection_outlier_df(model_with_outlier_data):
    dc = DataClean(model_with_outlier_data, factors_to_check_outliers=['Factor 1', 'Factor 2'])
    dc.check_outlier()
    expected_output = pd.DataFrame(
        {'Factor 1': np.array([0, 1, 3, 4, 5]), 'Factor 2': [0.9, 1, 1.1, 1, 0.9], 'Response': [1] * 5})
    pd.testing.assert_frame_equal(dc.model.data, expected_output)


def test_outlier_detection_data(model_with_outlier_data):
    dc = DataClean(model_with_outlier_data, factors_to_check_outliers=['Factor 1', 'Factor 2'])
    dc.check_outlier()
    expected_output = pd.DataFrame({'Factor 1': np.array([2]), 'Factor 2': np.array([100], dtype=float), 'Response': [1]})
    expected_output.index = [2]
    pd.testing.assert_frame_equal(dc.outlier_df, expected_output)
