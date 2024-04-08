import pandas as pd

class DataFrameOps:
    @staticmethod
    def save_df_to_csv(df, file_name):
        df.to_csv(file_name, index=False)
        print(f"DataFrame saved to {file_name}")

    @staticmethod
    def csv_to_df(file_name):
        return pd.read_csv(file_name)

    @staticmethod
    def df_to_excel(df, file_name):
        df.to_excel(file_name, index=False)
        print(f"DataFrame saved to {file_name}")
    

    
    @staticmethod
    def filter_by_word(df, column, word, case_sensitive=False, exact_match=False, is_index=False):
        """
        Filters the DataFrame based on a word match in a specified column. The column can be identified by name or index,
        with explicit indication.

        Parameters:
        - df: pandas.DataFrame to filter.
        - column: The column name (str) or index (int) to apply the filter on.
        - word: The word to match.
        - case_sensitive: Whether the match should be case-sensitive. Defaults to False.
        - exact_match: If True, filters for exact word matches. Otherwise, includes partial matches. Defaults to False.
        - is_index: Specifies if 'column' is a column index. Defaults to False.

        Returns:
        - A new DataFrame with rows matching the specified word.
        """
        # Convert column index to name if necessary
        if is_index:
            column = df.columns[column]

        if not case_sensitive:
            word = word.lower()
            df_temp = df.copy()
            df_temp[column] = df_temp[column].astype(str).str.lower()
        else:
            df_temp = df

        if exact_match:
            filtered_df = df_temp[df_temp[column] == word]
        else:
            filtered_df = df_temp[df_temp[column].str.contains(word, na=False)]

        return filtered_df

# Example usage:
# df_filtere    d 
    
# DataFrameOps.filter_by_word(df, 'columnName', 'searchWord', is_index=False)
# df_filtered = DataFrameOps.filter_by_word(df, 0, 'searchWord', is_index=True)

    @staticmethod
    def filter_by_value_range(df, column, min_val, max_val, inclusive=True):
        """
        Filters the DataFrame to include rows where the specified column's values are within the given range.

        Parameters:
        - df: pandas.DataFrame to filter.
        - column: The column name to apply the filter on.
        - min_val: The minimum value of the range.
        - max_val: The maximum value of the range.
        - inclusive: Whether the range includes the boundary values. Defaults to True.

        Returns:
        - A new DataFrame with rows matching the specified range.
        """
        if inclusive:
            return df[(df[column] >= min_val) & (df[column] <= max_val)]
        else:
            return df[(df[column] > min_val) & (df[column] < max_val)]

    
    @staticmethod
    def filter_by_membership(df, column, values_list):
        """
        Filters the DataFrame to include rows where the specified column's value is in the provided list of values.

        Parameters:
        - df: pandas.DataFrame to filter.
        - column: The column name to apply the filter on.
        - values_list: A list of values to include.

        Returns:
        - A new DataFrame with rows where the column's value is in the values_list.
        """
        return df[df[column].isin(values_list)]



    
    @staticmethod
    def filter_by_date_range(df, date_column, start_date, end_date):
        """
        Filters the DataFrame to include rows where the date in the specified column is within the given date range.

        Parameters:
        - df: pandas.DataFrame to filter.
        - date_column: The name of the column with datetime values.
        - start_date: The start date of the range (inclusive).
        - end_date: The end date of the range (inclusive).

        Returns:
        - A new DataFrame with rows within the specified date range.
        """
        # Ensure the date_column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Filter the DataFrame
        return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    
    @staticmethod
    def filter_conditions(df, conditions, is_index=False):
        """
        Filters the DataFrame based on multiple conditions, with support for both column names and indices.

        Parameters:
        - df: pandas.DataFrame to filter.
        - conditions: A condition or list of conditions as strings.
        - is_index: Boolean indicating if conditions refer to column indices (True) or names (False).

        Returns:
        - A new DataFrame filtered based on the specified conditions.
        """
        if isinstance(conditions, str):
            conditions = [conditions]

        mask = pd.Series(True, index=df.index)
        for condition in conditions:
            try:
                column, operator, value = DataFrameOps._parse_condition(condition, df, is_index)
                
                if operator == '==':
                    mask &= (df[column] == value)
                elif operator == '>':
                    mask &= (df[column] > value)
                elif operator == '<':
                    mask &= (df[column] < value)
                elif operator == 'like':
                    mask &= df[column].str.contains(value.replace('%', ''), regex=True)
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
            except Exception as e:
                print(f"Error processing condition '{condition}': {e}")
                return pd.DataFrame()  # Return empty DataFrame on error

        return df[mask]

    @staticmethod
    def _parse_condition(condition, df, is_index):
        """
        Parses a condition string into column, operator, and value.
        Adjusts column identifier based on is_index.
        """
        matches = re.match(r"(.*?)\s*(==|>|<|like)\s*(.*)", condition)
        if not matches:
            raise ValueError("Condition format is invalid")

        column, operator, value = matches.groups()

        # Adjust column if is_index is True
        if is_index:
            try:
                column_index = int(column)
                column = df.columns[column_index]
            except (ValueError, IndexError):
                raise ValueError(f"Column index {column} is out of bounds")

        # Attempt to convert value to numeric or bool, fallback to string
        try:
            value = eval(value)
        except:
            pass

        return column, operator, value
    

    @staticmethod
    def exclude_values(df, column, values_to_exclude, is_index=False):
        """
        Excludes rows based on specific values in a given column.

        Parameters:
        - df: pandas.DataFrame to filter.
        - column: The column name (str) or index (int) to apply the exclusion on.
        - values_to_exclude: A value or list of values to be excluded.
        - is_index: Boolean indicating if 'column' is a column index. Defaults to False.

        Returns:
        - A new DataFrame with rows containing the specified values in the given column excluded.

        Example Usage:
        --------------
        # Excluding rows where 'age' column is 25
        df_filtered = DataFrameOps.exclude_values(df, 'age', 25)
        
        # Excluding rows where first column contains either 100 or 200
        df_filtered = DataFrameOps.exclude_values(df, 0, [100, 200], is_index=True)
        """
        # Adjust column identifier if it's an index
        if is_index:
            try:
                column = df.columns[column]
            except IndexError:
                raise ValueError("Column index is out of bounds.")

        # Ensure values_to_exclude is a list to simplify processing
        if not isinstance(values_to_exclude, list):
            values_to_exclude = [values_to_exclude]

        # Exclude specified values
        mask = ~df[column].isin(values_to_exclude)
        return df[mask]
    
    @staticmethod
    def _get_column_name(df, column, is_index):
        """
        Utility method to get column name from index if necessary.
        """
        if is_index:
            try:
                return df.columns[column]
            except IndexError:
                raise ValueError("Column index is out of bounds.")
        return column
    @staticmethod
    def filter_starts_with(df, column, pattern, is_index=False, case_sensitive=True):
        """
        Filters rows where values in the specified column start with the given pattern.

        Parameters are similar to the filter_starts_with method.
        """
        column = DataFrameOps._get_column_name(df, column, is_index)
        if case_sensitive:
            return df[df[column].str.startswith(pattern, na=False)]
        else:
            return df[df[column].str.lower().str.startswith(pattern.lower(), na=False)]

    @staticmethod
    def filter_ends_with(df, column, pattern, is_index=False, case_sensitive=True):
        """
        Filters rows where values in the specified column end with the given pattern.

        Parameters are similar to the filter_ends_with method.
        """
        column = DataFrameOps._get_column_name(df, column, is_index)
        if case_sensitive:
            return df[df[column].str.endswith(pattern, na=False)]
        else:
            return df[df[column].str.lower().str.endswith(pattern.lower(), na=False)]

    @staticmethod
    def filter_top_n(df, column, n, is_index=False, ascending=False):
        """
        Returns a DataFrame with the top N items sorted by the specified column.

        Parameters:
        - df: pandas.DataFrame to filter.
        - column: The column name (str) or index (int) used for sorting to select top N items.
        - n: Number of top items to keep.
        - is_index: Boolean indicating if 'column' is a column index. Defaults to False.
        - ascending: Sort ascending vs. descending. Defaults to False for top N items.

        Returns:
        - A new DataFrame with the top N items.

        Example Usage:
        --------------
        # Get top 5 items by 'sales' column
        top_sales = DataFrameOps.filter_top_n(df_sales, 'sales', 5)
        
        # Get top 3 items by the first column, in ascending order
        top_items = DataFrameOps.filter_top_n(df_items, 0, 3, is_index=True, ascending=True)
        """
        if is_index:
            column = df.columns[column]

        return df.sort_values(by=column, ascending=ascending).head(n)
    

    @staticmethod
    def match_by_column(df1, df2, column_name):
        """
        Matches and returns rows from two DataFrames where the values in the specified column are the same.

        Parameters:
        - df1: The first DataFrame.
        - df2: The second DataFrame.
        - column_name: The name of the column to match on.

        Returns:
        - A DataFrame containing matching rows from both input DataFrames based on the specified column.
        """
        # Performing an inner join on the specified column
        matched_df = pd.merge(df1, df2, how='inner', on=column_name)
        return matched_df
    
    @staticmethod
    def match_and_return_from(df1, df2, column_name, return_from):
        """
        Matches rows between two DataFrames based on a common column and returns only the 
        matching rows from one of the specified DataFrames as chosen by the user.

        Parameters:
        - df1: The first DataFrame.
        - df2: The second DataFrame.
        - column_name: The name of the column to match on.
        - return_from: A string specifying which DataFrame ('df1' or 'df2') to return rows from.

        Returns:
        - A DataFrame containing matching rows from the specified DataFrame based on the common column.
        """
        if return_from not in ['df1', 'df2']:
            raise ValueError("The 'return_from' parameter must be 'df1' or 'df2'.")

        # Perform an inner join to get matching rows
        matched_df = pd.merge(df1, df2, how='inner', on=column_name)

        # Return only columns from the specified DataFrame
        if return_from == 'df1':
            columns_to_keep = [col for col in matched_df.columns if col in df1.columns or col == column_name]
        else:  # return_from == 'df2'
            columns_to_keep = [col for col in matched_df.columns if col in df2.columns or col == column_name]

        return matched_df[columns_to_keep]
    

    @staticmethod
    def non_matching_rows(df1, df2, column_name):
        """
        Finds rows in both DataFrames that do not have a match based on the specified column.

        Returns two new DataFrames with non-matching rows from df1 and df2, respectively.
        """
        df1_ids = set(df1[column_name])
        df2_ids = set(df2[column_name])

        non_matching_df1 = df1[~df1[column_name].isin(df2_ids)]
        non_matching_df2 = df2[~df2[column_name].isin(df1_ids)]

        return non_matching_df1, non_matching_df2

    @staticmethod
    def non_matching_from(df1, df2, column_name, return_from):
        """
        Finds rows in the specified DataFrame (df1 or df2) that do not have a match in the other DataFrame.

        Returns a new DataFrame with non-matching rows from the specified DataFrame.
        """
        if return_from == 'df1':
            non_matching_ids = set(df1[column_name]) - set(df2[column_name])
            return df1[df1[column_name].isin(non_matching_ids)]
        elif return_from == 'df2':
            non_matching_ids = set(df2[column_name]) - set(df1[column_name])
            return df2[df2[column_name].isin(non_matching_ids)]
        else:
            raise ValueError("Invalid 'return_from' value. Choose 'df1' or 'df2'.")
        


    @staticmethod
    def match_by_conditions(df1, df2, match_column, additional_conditions=[], is_index=False):
        """
        Matches rows between two DataFrames based on a common column and additional conditions.
        """
        # Convert column index to name if necessary
        match_column = DataFrameOps._get_column_name(df1 if not is_index else df2, match_column, is_index)

        # Perform an inner join on the match column
        matched_df = pd.merge(df1, df2, how='inner', on=match_column)

        # Apply additional conditions using the logic similar to `filter_conditions`
        for condition in additional_conditions:
            column, operator, value = condition
            matched_df = DataFrameOps._apply_condition(matched_df, column, operator, value, is_index=False)

        return matched_df

    @staticmethod
    def match_and_return_from_with_conditions(df1, df2, match_column, return_from, additional_conditions=[], is_index=False):
        """
        Matches rows between two DataFrames based on a common column and additional conditions, but returns only the matching rows from one of the specified DataFrames.
        """
        matched_df = DataFrameOps.match_by_conditions(df1, df2, match_column, additional_conditions, is_index)

        # Filter to return only columns from the specified DataFrame after applying conditions
        if return_from == 'df1':
            columns_to_keep = [col for col in matched_df.columns if col in df1.columns or col == match_column]
        else:  # return_from == 'df2'
            columns_to_keep = [col for col in matched_df.columns if col in df2.columns or col == match_column]

        return matched_df[columns_to_keep]

    @staticmethod
    def _apply_condition(df, column, operator, value, is_index):
        """
        Applies a single condition to the DataFrame, filtering rows based on the condition. 
        This is a helper function that encapsulates the logic for applying conditions.
        """
        # This function needs to implement the logic for applying the condition based on the operator and value
        # Similar to the implementation details in `filter_conditions`
        # For simplicity, this function is described conceptually
        return df  # Placeholder for the actual implementation

    @staticmethod
    def _get_column_name(df, column, is_index):
        """
        Utility method to get column name from index if necessary.
        """
        if is_index:
            try:
                return df.columns[column]
            except IndexError:
                raise ValueError("Column index is out of bounds.")
        return column     