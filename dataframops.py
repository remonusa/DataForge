import pandas as pd
import matplotlib.pyplot as plt


class DataFrameOps:
    """
    Class for various DataFrame operations, including saving, loading, and filtering DataFrames.
    """

    @staticmethod
    def _validate_dataframe(df):
        """
        Validate that the input is a non-empty pandas DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If the input is not a DataFrame or is empty.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input data should be a non-empty pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        @staticmethod
        def _validate_column_exists(df, column):
            """
            Validate if the specified column exists in the DataFrame.

            Parameters:
                df (pd.DataFrame): The DataFrame to check.
                column (str): The column name to validate.

            Raises:
                KeyError: If the column is not found in the DataFrame.
            """
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame.")
    # 1. Save DataFrame to CSV
    @staticmethod
    def save_df_to_csv(df, file_name, verbose=False):
        """
        Save a DataFrame to a CSV file.

        Parameters:
            df (pd.DataFrame): DataFrame to save.
            file_name (str): Name of the file to save the DataFrame to.
            verbose (bool): If True, prints a success message.

        Returns:
            None
        """
        DataFrameOps._validate_dataframe(df)

        try:
            df.to_csv(file_name, index=False)
            if verbose:
                print(f"DataFrame successfully saved to '{file_name}'")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")

    # 2. Load DataFrame from CSV
    @staticmethod
    def csv_to_df(file_name, verbose=False):
        """
        Load a DataFrame from a CSV file.

        Parameters:
            file_name (str): Name of the CSV file to load the DataFrame from.
            verbose (bool): If True, prints a success message.

        Returns:
            pd.DataFrame: Loaded DataFrame, or None if an error occurs.
        """
        try:
            df = pd.read_csv(file_name)
            if verbose:
                print(f"DataFrame successfully loaded from '{file_name}'")
            return df
        except FileNotFoundError:
            print(f"File '{file_name}' not found.")
        except Exception as e:
            print(f"Error loading CSV: {e}")
        return None

    # 3. Save DataFrame to Excel
    @staticmethod
    def df_to_excel(df, file_name, verbose=False):
        """
        Save a DataFrame to an Excel file.

        Parameters:
            df (pd.DataFrame): DataFrame to save.
            file_name (str): Name of the file to save the DataFrame to.
            verbose (bool): If True, prints a success message.

        Returns:
            None
        """
        DataFrameOps._validate_dataframe(df)

        try:
            df.to_excel(file_name, index=False)
            if verbose:
                print(f"DataFrame successfully saved to Excel file '{file_name}'")
        except Exception as e:
            print(f"Error saving DataFrame to Excel: {e}")

    # 4. Filter DataFrame by Word in a Column
    @staticmethod
    def filter_by_word(df, column, word, case_sensitive=False, exact_match=False, is_index=False, verbose=False):
        """
        Filter DataFrame rows by a word in a specified column.

        Parameters:
            df (pd.DataFrame): DataFrame to filter.
            column (str): Column to search for the word.
            word (str): Word to filter by.
            case_sensitive (bool): Whether the search is case sensitive.
            exact_match (bool): Whether to match the exact word.
            is_index (bool): Whether to return the index of matching rows.
            verbose (bool): If True, prints the number of rows that match the filter.

        Returns:
            pd.DataFrame or pd.Index: Filtered DataFrame or index of matching rows.
        """
        # Validate inputs
        DataFrameOps._validate_dataframe(df)
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        # Perform filtering
        df = df.copy()
        if not case_sensitive:
            df[column] = df[column].str.lower()
            word = word.lower()

        if exact_match:
            result = df[df[column] == word]
        else:
            result = df[df[column].str.contains(word, na=False)]

        if verbose:
            print(f"Filtered DataFrame with word '{word}' in column '{column}' - Rows matched: {len(result)}")

        return result.index if is_index else result

    # 5. Filter DataFrame by Value Range
    @staticmethod
    def filter_by_value_range(df, column, min_val, max_val, inclusive=True, verbose=False):
        """
        Filter DataFrame rows by a range of values in a specified column.

        Parameters:
            df (pd.DataFrame): DataFrame to filter.
            column (str): Column to filter by value range.
            min_val (int/float): Minimum value of the range.
            max_val (int/float): Maximum value of the range.
            inclusive (bool): Whether the range is inclusive.
            verbose (bool): If True, prints the number of rows that match the filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        # Validate inputs
        DataFrameOps._validate_dataframe(df)
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        # Apply the range filter
        if inclusive:
            result = df[(df[column] >= min_val) & (df[column] <= max_val)]
        else:
            result = df[(df[column] > min_val) & (df[column] < max_val)]

        if verbose:
            print(f"Filtered DataFrame by range {min_val} to {max_val} in column '{column}' - Rows matched: {len(result)}")

        return result

     # 6. Filter DataFrame by Membership in a List
    @staticmethod
    def filter_by_membership(df, column, values_list, verbose=False):
        """
        Filter DataFrame rows by membership in a list of values in a specified column.

        Parameters:
            df (pd.DataFrame): DataFrame to filter.
            column (str): Column to filter by membership.
            values_list (list): List of values to filter by.
            verbose (bool): If True, prints the number of rows that match the filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        # Validate inputs
        DataFrameOps._validate_dataframe(df)
        DataFrameOps._validate_column_exists(df, column)

        result = df[df[column].isin(values_list)]

        if verbose:
            print(f"6. Filter by Membership: Rows matching values {values_list} in column '{column}': {len(result)}")

        return result

    # 7. Filter DataFrame by Date Range
    @staticmethod
    def filter_by_date_range(df, date_column, start_date, end_date, verbose=False):
        """
        Filter DataFrame rows by a range of dates in a specified column.

        Parameters:
            df (pd.DataFrame): DataFrame to filter.
            date_column (str): Column to filter by date range.
            start_date (str/datetime): Start date of the range.
            end_date (str/datetime): End date of the range.
            verbose (bool): If True, prints the number of rows that match the filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        # Validate inputs
        DataFrameOps._validate_dataframe(df)
        DataFrameOps._validate_column_exists(df, date_column)

        mask = (df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))
        result = df.loc[mask]

        if verbose:
            print(f"7. Filter by Date Range: Rows within {start_date} to {end_date} in column '{date_column}': {len(result)}")

        return result

    # 8. Filter DataFrame by Multiple Conditions
    @staticmethod
    def filter_conditions(df, conditions, is_index=False, verbose=False):
        """
        Filter DataFrame rows based on multiple conditions.

        Parameters:
            df (pd.DataFrame): DataFrame to filter.
            conditions (str): Conditions to filter by (string format as in pandas `.query`).
            is_index (bool): Whether to return the index of matching rows.
            verbose (bool): If True, prints the number of rows that match the conditions.

        Returns:
            pd.DataFrame or pd.Index: Filtered DataFrame or index of matching rows.
        """
        # Validate inputs
        DataFrameOps._validate_dataframe(df)

        try:
            result = df.query(conditions)
            if verbose:
                print(f"8. Filter by Conditions: Rows matching '{conditions}': {len(result)}")
            return result.index if is_index else result
        except Exception as e:
            raise ValueError(f"Invalid condition provided: {e}")

    # 9. Exclude Rows with Specified Values
    @staticmethod
    def exclude_values(df, column, values_to_exclude, is_index=False, verbose=False):
        """
        Exclude rows from a DataFrame based on values in a specified column.

        Parameters:
            df (pd.DataFrame): DataFrame to filter.
            column (str): Column to exclude values from.
            values_to_exclude (list): Values to exclude.
            is_index (bool): Whether to return the index of remaining rows.
            verbose (bool): If True, prints the number of rows remaining after exclusion.

        Returns:
            pd.DataFrame or pd.Index: Filtered DataFrame or index of remaining rows.
        """
        # Validate inputs
        DataFrameOps._validate_dataframe(df)
        DataFrameOps._validate_column_exists(df, column)

        result = df[~df[column].isin(values_to_exclude)]

        if verbose:
            print(f"9. Exclude Values: Rows after excluding {values_to_exclude} in column '{column}': {len(result)}")

        return result.index if is_index else result

    # 10. Filter DataFrame by Starts With Pattern
    @staticmethod
    def filter_starts_with(df, column, pattern, is_index=False, case_sensitive=True, verbose=False):
        """
        Filter DataFrame rows where a specified column starts with a given pattern.

        Parameters:
            df (pd.DataFrame): DataFrame to filter.
            column (str): Column to filter by pattern.
            pattern (str): Pattern to filter by.
            is_index (bool): Whether to return the index of matching rows.
            case_sensitive (bool): Whether the search is case sensitive.
            verbose (bool): If True, prints the number of rows that match the filter.

        Returns:
            pd.DataFrame or pd.Index: Filtered DataFrame or index of matching rows.
        """
        # Validate inputs
        DataFrameOps._validate_dataframe(df)
        DataFrameOps._validate_column_exists(df, column)

        # Perform filtering
        if not case_sensitive:
            df[column] = df[column].str.lower()
            pattern = pattern.lower()

        result = df[df[column].str.startswith(pattern, na=False)]

        if verbose:
            print(f"10. Filter Starts With: Rows starting with '{pattern}' in column '{column}': {len(result)}")

        return result.index if is_index else result