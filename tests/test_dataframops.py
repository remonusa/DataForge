
import pytest
import pandas as pd
from DataForge.dataframops import DataFrameOps

def test_save_and_load_csv():
    # Test saving and loading a DataFrame to CSV
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)

    # Save to CSV
    DataFrameOps.save_df_to_csv(df, "test_output.csv")

    # Load from CSV
    loaded_df = DataFrameOps.csv_to_df("test_output.csv")

    # Check if the loaded DataFrame is the same as the original
    assert df.equals(loaded_df)

def test_filter_by_value_range():
    # Test filtering DataFrame rows by a range of values
    data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)

    # Filter rows where column 'A' is between 2 and 4
    filtered_df = DataFrameOps.filter_by_value_range(df, column='A', min_val=2, max_val=4)

    # Check that only rows 2, 3, and 4 are in the filtered DataFrame
    expected_data = {'A': [2, 3, 4], 'B': [20, 30, 40]}
    expected_df = pd.DataFrame(expected_data)

    assert filtered_df.equals(expected_df)

def test_filter_by_membership():
    # Test filtering DataFrame rows by membership
    data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)

    # Filter rows where 'A' is in the list [2, 4]
    filtered_df = DataFrameOps.filter_by_membership(df, column='A', values_list=[2, 4])

    # Check that only rows with A=2 and A=4 are in the filtered DataFrame
    expected_data = {'A': [2, 4], 'B': [20, 40]}
    expected_df = pd.DataFrame(expected_data)

    assert filtered_df.equals(expected_df)
