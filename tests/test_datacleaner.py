
import pytest
import pandas as pd
import numpy as np
from DataForge.datacleaner import DataCleaner

def test_fill_missing_values():
    # Test filling missing values in a DataFrame
    data = {'A': [1, 2, np.nan, 4], 'B': [np.nan, 6, 7, 8]}
    df = pd.DataFrame(data)

    # Fill missing values using the mean strategy
    filled_df = DataCleaner.fill_missing_values(df, strategy='mean')

    # Check that missing values are filled correctly
    assert round(filled_df['A'].iloc[2], 2) == 2.33  # Mean of [1, 2, 4] for column 'A'
    assert filled_df['B'].iloc[0] == 7.0   # Mean of [6, 7, 8] for column 'B'

def test_remove_outliers_iqr():
    # Test removing outliers using the IQR method
    data = {'A': [1, 2, 3, 100, 5], 'B': [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)

    # Remove outliers using IQR with a factor of 1.5
    cleaned_df = DataCleaner.remove_outliers(df, method='IQR', factor=1.5)

    # Expect only the row with A=100 to be removed
    expected_data = {'A': [1, 2, 3, 5], 'B': [10, 20, 30, 50]}
    expected_df = pd.DataFrame(expected_data)

    assert cleaned_df.equals(expected_df)

def test_encode_categorical():
    # Test encoding categorical variables
    data = {'Category': ['A', 'B', 'A', 'C'], 'Values': [10, 15, 10, 20]}
    df = pd.DataFrame(data)

    # Encode categorical variables using label encoding
    encoded_df = DataCleaner.encode_categorical(df, method='label')

    # Check that the categories are encoded correctly
    expected_data = {'Category': [0, 1, 0, 2], 'Values': [10, 15, 10, 20]}
    expected_df = pd.DataFrame(expected_data)

    assert encoded_df.equals(expected_df)
