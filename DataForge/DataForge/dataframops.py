import pandas as pd
import matplotlib.pyplot as plt

class DataFrameOps:
    """
    Class for various DataFrame operations.
    """

    @staticmethod
    def save_df_to_csv(df, file_name):
        """
        Save a DataFrame to a CSV file.

        :param df: DataFrame to save.
        :param file_name: Name of the file to save the DataFrame to.
        """
        try:
            df.to_csv(file_name, index=False)
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")

    @staticmethod
    def csv_to_df(file_name):
        """
        Load a DataFrame from a CSV file.

        :param file_name: Name of the CSV file to load the DataFrame from.
        :return: Loaded DataFrame.
        """
        try:
            return pd.read_csv(file_name)
        except FileNotFoundError:
            print(f"File {file_name} not found.")
        except Exception as e:
            print(f"Error loading CSV: {e}")

    @staticmethod
    def df_to_excel(df, file_name):
        """
        Save a DataFrame to an Excel file.

        :param df: DataFrame to save.
        :param file_name: Name of the file to save the DataFrame to.
        """
        try:
            df.to_excel(file_name, index=False)
        except Exception as e:
            print(f"Error saving DataFrame to Excel: {e}")

    @staticmethod
    def filter_by_word(df, column, word, case_sensitive=False, exact_match=False, is_index=False):
        """
        Filter DataFrame rows by a word in a specified column.

        :param df: DataFrame to filter.
        :param column: Column to search for the word.
        :param word: Word to filter by.
        :param case_sensitive: Whether the search is case sensitive.
        :param exact_match: Whether to match the exact word.
        :param is_index: Whether to return the index of matching rows.
        :return: Filtered DataFrame or index of matching rows.
        """
        if not case_sensitive:
            df[column] = df[column].str.lower()
            word = word.lower()
        if exact_match:
            result = df[df[column] == word]
        else:
            result = df[df[column].str.contains(word, na=False)]
        return result.index if is_index else result

    @staticmethod
    def filter_by_value_range(df, column, min_val, max_val, inclusive=True):
        """
        Filter DataFrame rows by a range of values in a specified column.

        :param df: DataFrame to filter.
        :param column: Column to filter by value range.
        :param min_val: Minimum value of the range.
        :param max_val: Maximum value of the range.
        :param inclusive: Whether the range is inclusive.
        :return: Filtered DataFrame.
        """
        if inclusive:
            return df[(df[column] >= min_val) & (df[column] <= max_val)]
        else:
            return df[(df[column] > min_val) & (df[column] < max_val)]

    @staticmethod
    def filter_by_membership(df, column, values_list):
        """
        Filter DataFrame rows by membership in a list of values in a specified column.

        :param df: DataFrame to filter.
        :param column: Column to filter by membership.
        :param values_list: List of values to filter by.
        :return: Filtered DataFrame.
        """
        return df[df[column].isin(values_list)]

    @staticmethod
    def filter_by_date_range(df, date_column, start_date, end_date):
        """
        Filter DataFrame rows by a range of dates in a specified column.

        :param df: DataFrame to filter.
        :param date_column: Column to filter by date range.
        :param start_date: Start date of the range.
        :param end_date: End date of the range.
        :return: Filtered DataFrame.
        """
        mask = (df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))
        return df.loc[mask]

    @staticmethod
    def filter_conditions(df, conditions, is_index=False):
        """
        Filter DataFrame rows based on multiple conditions.

        :param df: DataFrame to filter.
        :param conditions: Conditions to filter by.
        :param is_index: Whether to return the index of matching rows.
        :return: Filtered DataFrame or index of matching rows.
        """
        result = df.query(conditions)
        return result.index if is_index else result

    @staticmethod
    def exclude_values(df, column, values_to_exclude, is_index=False):
        """
        Exclude rows from a DataFrame based on values in a specified column.

        :param df: DataFrame to filter.
        :param column: Column to exclude values from.
        :param values_to_exclude: Values to exclude.
        :param is_index: Whether to return the index of remaining rows.
        :return: Filtered DataFrame or index of remaining rows.
        """
        result = df[~df[column].isin(values_to_exclude)]
        return result.index if is_index else result

    @staticmethod
    def filter_starts_with(df, column, pattern, is_index=False, case_sensitive=True):
        """
        Filter DataFrame rows where a specified column starts with a given pattern.

        :param df: DataFrame to filter.
        :param column: Column to filter by pattern.
        :param pattern: Pattern to filter by.
        :param is_index: Whether to return the index of matching rows.
        :param case_sensitive: Whether the search is case sensitive.
        :return: Filtered DataFrame or index of matching rows.
        """
        if not case_sensitive:
            df[column] = df[column].str.lower()
            pattern = pattern.lower()
        result = df[df[column].str.startswith(pattern, na=False)]
        return result.index if is_index else result

    @staticmethod
    def filter_ends_with(df, column, pattern, is_index=False, case_sensitive=True):
        """
        Filter DataFrame rows where a specified column ends with a given pattern.

        :param df: DataFrame to filter.
        :param column: Column to filter by pattern.
        :param pattern: Pattern to filter by.
        :param is_index: Whether to return the index of matching rows.
        :param case_sensitive: Whether the search is case sensitive.
        :return: Filtered DataFrame or index of matching rows.
        """
        if not case_sensitive:
            df[column] = df[column].str.lower()
            pattern = pattern.lower()
        result = df[df[column].str.endswith(pattern, na=False)]
        return result.index if is_index else result

    @staticmethod
    def filter_top_n(df, column, n, is_index=False, ascending=False):
        """
        Filter the top N rows of a DataFrame based on a specified column.

        :param df: DataFrame to filter.
        :param column: Column to filter by.
        :param n: Number of rows to return.
        :param is_index: Whether to return the index of matching rows.
        :param ascending: Whether to sort in ascending order.
        :return: Filtered DataFrame or index of matching rows.
        """
        result = df.nsmallest(n, column) if ascending else df.nlargest(n, column)
        return result.index if is_index else result

    @staticmethod
    def match_by_conditions(df1, df2, match_column, additional_conditions=[], is_index=False):
        """
        Match rows from two DataFrames based on a column and additional conditions.

        :param df1: First DataFrame.
        :param df2: Second DataFrame.
        :param match_column: Column to match on.
        :param additional_conditions: List of additional conditions for matching.
        :param is_index: Whether to return the index of matching rows.
        :return: Matching rows DataFrame or index of matching rows.
        """
        condition = df1[match_column].isin(df2[match_column])
        if additional_conditions:
            condition &= df1.eval(' & '.join(additional_conditions))
        result = df1[condition]
        return result.index if is_index else result

    @staticmethod
    def match_and_return_from_with_conditions(df1, df2, match_column, return_from, additional_conditions=[], is_index=False):
        """
        Match rows from two DataFrames based on a column and additional conditions, returning specified columns.

        :param df1: First DataFrame.
        :param df2: Second DataFrame.
        :param match_column: Column to match on.
        :param return_from: Columns to return from the matched rows.
        :param additional_conditions: List of additional conditions for matching.
        :param is_index: Whether to return the index of matching rows.
        :return: DataFrame with specified columns from matching rows or index of matching rows.
        """
        condition = df1[match_column].isin(df2[match_column])
        if additional_conditions:
            condition &= df1.eval(' & '.join(additional_conditions))
        result = df1.loc[condition, return_from]
        return result.index if is_index else result

    @staticmethod
    def non_matching_rows_with_conditions(df1, df2, match_column, additional_conditions=[]):
        """
        Get non-matching rows from the first DataFrame based on a column and additional conditions.

        :param df1: First DataFrame.
        :param df2: Second DataFrame.
        :param match_column: Column to match on.
        :param additional_conditions: List of additional conditions for matching.
        :return: DataFrame with non-matching rows.
        """
        condition = ~df1[match_column].isin(df2[match_column])
        if additional_conditions:
            condition &= df1.eval(' & '.join(additional_conditions))
        return df1[condition]

    @staticmethod
    def non_matching_from_with_conditions(df1, df2, match_column, return_from, additional_conditions=[]):
        """
        Get non-matching rows from the first DataFrame based on a column and additional conditions, returning specified columns.

        :param df1: First DataFrame.
        :param df2: Second DataFrame.
        :param match_column: Column to match on.
        :param return_from: Columns to return from the non-matching rows.
        :param additional_conditions: List of additional conditions for matching.
        :return: DataFrame with specified columns from non-matching rows.
        """
        condition = ~df1[match_column].isin(df2[match_column])
        if additional_conditions:
            condition &= df1.eval(' & '.join(additional_conditions))
        return df1.loc[condition, return_from]

    # Additional methods for data cleaning, transformation, and visualization

    @staticmethod
    def drop_na(df, columns):
        """
        Drop rows with missing values in specified columns.

        :param df: DataFrame to clean.
        :param columns: Columns to check for missing values.
        :return: Cleaned DataFrame.
        """
        return df.dropna(subset=columns)

    @staticmethod
    def fill_na(df, column, value):
        """
        Fill missing values in a specified column with a given value.

        :param df: DataFrame to clean.
        :param column: Column to fill missing values in.
        :param value: Value to fill missing values with.
        :return: Cleaned DataFrame.
        """
        df[column].fillna(value, inplace=True)
        return df

    @staticmethod
    def remove_duplicates(df, subset):
        """
        Remove duplicate rows based on a subset of columns.

        :param df: DataFrame to clean.
        :param subset: Columns to check for duplicates.
        :return: Cleaned DataFrame.
        """
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def add_column(df, column_name, default_value):
        """
        Add a new column with a default value.

        :param df: DataFrame to modify.
        :param column_name: Name of the new column.
        :param default_value: Default value for the new column.
        :return: Modified DataFrame.
        """
        df[column_name] = default_value
        return df

    @staticmethod
    def rename_column(df, old_name, new_name):
        """
        Rename a column.

        :param df: DataFrame to modify.
        :param old_name: Old column name.
        :param new_name: New column name.
        :return: Modified DataFrame.
        """
        df.rename(columns={old_name: new_name}, inplace=True)
        return df

    @staticmethod
    def apply_function(df, column, func):
        """
        Apply a function to a column.

        :param df: DataFrame to modify.
        :param column: Column to apply the function to.
        :param func: Function to apply.
        :return: Modified DataFrame.
        """
        df[column] = df[column].apply(func)
        return df

    @staticmethod
    def group_and_aggregate(df, group_by_column, agg_column, agg_func):
        """
        Group by a column and aggregate another column with a given function.

        :param df: DataFrame to group and aggregate.
        :param group_by_column: Column to group by.
        :param agg_column: Column to aggregate.
        :param agg_func: Aggregation function (e.g., sum, mean).
        :return: Aggregated DataFrame.
        """
        return df.groupby(group_by_column)[agg_column].agg(agg_func)

    @staticmethod
    def plot_histogram(df, column):
        """
        Plot a histogram of a column.

        :param df: DataFrame to plot.
        :param column: Column to plot a histogram of.
        """
        df[column].plot(kind='hist')
        plt.show()

    @staticmethod
    def plot_scatter(df, x_column, y_column):
        """
        Plot a scatter plot of two columns.

        :param df: DataFrame to plot.
        :param x_column: Column for x-axis.
        :param y_column: Column for y-axis.
        """
        df.plot(kind='scatter', x=x_column, y=y_column)
        plt.show()
