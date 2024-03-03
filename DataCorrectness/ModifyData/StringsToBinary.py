import pandas as pd


def initial_list_status_to_binary(ser: pd.Series):
    """w = whole, f = fractional"""
    newser = ser.replace({'w': 1, 'f': 0})
    return newser


def yesno_to_binary(ser: pd.Series):
    newser = ser.replace({'Y': 1, 'N': 0, 'y': 1, 'n': 0})
    return newser


def application_type_to_binary(ser: pd.Series):
    newser = ser.replace({'Individual': 1, 'Joint App': 0})
    return newser


def disbursement_method_to_binary(ser: pd.Series):
    newser = ser.replace({'Cash': 1, 'DirectPay': 0})
    return newser
