import pandas as pd
from DataForge.dataframe_ops import DataFrameOps  # Adjust based on actual module path

def test_save_to_csv_and_load_csv():
    print("Starting test for save_df_to_csv and csv_to_df methods...")

    # Step 1: Create a sample DataFrame
    print("1. Creating a sample DataFrame.")
    df_original = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['x', 'y', 'z']
    })

    # File name for testing
    test_file_name = 'test_dataframe.csv'

    # Step 2: Save the DataFrame to a CSV file
    print("2. Saving the DataFrame to a CSV file.")
    DataFrameOps.save_df_to_csv(df_original, test_file_name)
    
    # Step 3: Load the CSV file back into a DataFrame
    print("3. Loading the CSV file back into a DataFrame.")
    df_loaded = DataFrameOps.csv_to_df(test_file_name)

    # Step 4: Verify the loaded DataFrame matches the original DataFrame
    print("4. Verifying the loaded DataFrame matches the original DataFrame.")
    pd.testing.assert_frame_equal(df_original, df_loaded, check_dtype=True)

    print("Test passed successfully: The DataFrame was saved to and loaded from a CSV file correctly.")

# Run the test function
if __name__ == "__main__":
    test_save_to_csv_and_load_csv()
