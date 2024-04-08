import pandas as pd
import unittest

class TestDataFrameOps(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'Name': ['John', 'Jane', 'Mike', 'Emily'],
            'Age': [25, 30, 35, 40],
            'City': ['New York', 'London', 'Paris', 'Tokyo']
        }
        self.df = pd.DataFrame(data)

    def test_save_df_to_csv(self):
        # Test saving DataFrame to CSV
        file_name = 'test.csv'
        DataFrameOps.save_df_to_csv(self.df, file_name)
        # Assert that the file exists
        self.assertTrue(os.path.exists(file_name))
        # Clean up the file
        os.remove(file_name)

    def test_csv_to_df(self):
        # Test loading DataFrame from CSV
        file_name = 'test.csv'
        self.df.to_csv(file_name, index=False)
        loaded_df = DataFrameOps.csv_to_df(file_name)
        # Assert that the loaded DataFrame is equal to the original DataFrame
        pd.testing.assert_frame_equal(self.df, loaded_df)
        # Clean up the file
        os.remove(file_name)

    def test_df_to_excel(self):
        # Test saving DataFrame to Excel
        file_name = 'test.xlsx'
        DataFrameOps.df_to_excel(self.df, file_name)
        # Assert that the file exists
        self.assertTrue(os.path.exists(file_name))
        # Clean up the file
        os.remove(file_name)

    def test_filter_by_word(self):
        # Test filtering DataFrame by word
        filtered_df = DataFrameOps.filter_by_word(self.df, 'Name', 'John')
        # Assert that the filtered DataFrame contains only the row with 'John'
        expected_df = pd.DataFrame({'Name': ['John'], 'Age': [25], 'City': ['New York']})
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_filter_by_value_range(self):
        # Test filtering DataFrame by value range
        filtered_df = DataFrameOps.filter_by_value_range(self.df, 'Age', 30, 40)
        # Assert that the filtered DataFrame contains only the rows with ages between 30 and 40 (inclusive)
        expected_df = pd.DataFrame({'Name': ['Jane', 'Mike', 'Emily'], 'Age': [30, 35, 40], 'City': ['London', 'Paris', 'Tokyo']})
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_filter_by_membership(self):
        # Test filtering DataFrame by membership
        filtered_df = DataFrameOps.filter_by_membership(self.df, 'City', ['London', 'Paris'])
        # Assert that the filtered DataFrame contains only the rows with cities 'London' and 'Paris'
        expected_df = pd.DataFrame({'Name': ['Jane', 'Mike'], 'Age': [30, 35], 'City': ['London', 'Paris']})
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_filter_by_date_range(self):
        # Test filtering DataFrame by date range
        data = {
            'Date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'],
            'Value': [10, 20, 30, 40]
        }
        df = pd.DataFrame(data)
        filtered_df = DataFrameOps.filter_by_date_range(df, 'Date', '2022-02-01', '2022-03-01')
        # Assert that the filtered DataFrame contains only the rows with dates between '2022-02-01' and '2022-03-01' (inclusive)
        expected_df = pd.DataFrame({'Date': ['2022-02-01', '2022-03-01'], 'Value': [20, 30]})
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_filter_conditions(self):
        # Test filtering DataFrame by conditions
        filtered_df = DataFrameOps.filter_conditions(self.df, ["Age > 30", "City == 'London'"])
        # Assert that the filtered DataFrame contains only the row with age > 30 and city 'London'
        expected_df = pd.DataFrame({'Name': ['Mike'], 'Age': [35], 'City': ['London']})
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_exclude_values(self):
        # Test excluding values from DataFrame
        excluded_df = DataFrameOps.exclude_values(self.df, 'Age', [25, 30])
        # Assert that the excluded DataFrame contains only the rows with ages other than 25 and 30
        expected_df = pd.DataFrame({'Name': ['Mike', 'Emily'], 'Age': [35, 40], 'City': ['Paris', 'Tokyo']})
        pd.testing.assert_frame_equal(excluded_df, expected_df)

    def test_filter_starts_with(self):
        # Test filtering DataFrame by values starting with a pattern
        filtered_df = DataFrameOps.filter_starts_with(self.df, 'Name', 'J')
        # Assert that the filtered DataFrame contains only the rows with names starting with 'J'
        expected_df = pd.DataFrame({'Name': ['John', 'Jane'], 'Age': [25, 30], 'City': ['New York', 'London']})
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_filter_ends_with(self):
        # Test filtering DataFrame by values ending with a pattern
        filtered_df = DataFrameOps.filter_ends_with(self.df, 'City', 'n')
        # Assert that the filtered DataFrame contains only the rows with cities ending with 'n'
        expected_df = pd.DataFrame({'Name': ['John', 'London'], 'Age': [25, 30], 'City': ['New York', 'London']})
        pd.testing.assert_frame_equal(filtered_df, expected_df)

if __name__ == '__main__':
    unittest.main()