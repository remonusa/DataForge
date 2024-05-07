DataForge
This is a Python library for data manipulation and file operations.

save_df_to_csv(df, file_name)
csv_to_df(file_name)
df_to_excel(df, file_name)
filter_by_word(df, column, word, case_sensitive=False, exact_match=False, is_index=False)
filter_by_value_range(df, column, min_val, max_val, inclusive=True)
filter_by_membership(df, column, values_list)
filter_by_date_range(df, date_column, start_date, end_date)
filter_conditions(df, conditions, is_index=False)
exclude_values(df, column, values_to_exclude, is_index=False)
filter_starts_with(df, column, pattern, is_index=False, case_sensitive=True)
filter_ends_with(df, column, pattern, is_index=False, case_sensitive=True)
filter_top_n(df, column, n, is_index=False, ascending=False)
match_by_conditions(df1, df2, match_column, additional_conditions=[], is_index=False)
match_and_return_from_with_conditions(df1, df2, match_column, return_from, additional_conditions=[], is_index=False)
non_matching_rows_with_conditions(df1, df2, match_column, additional_conditions=[])
non_matching_from_with_conditions(df1, df2, match_column, return_from, additional_conditions=[])
Below is the detailed documentation for the first 8 methods of your DataFrameOps class, provided in Markdown format suitable for a README file. This document explains each method's purpose, parameters, and includes usage examples.

# DataFrameOps Class Documentation

`DataFrameOps` is a utility class offering a suite of static methods to simplify common operations with pandas DataFrames, including data filtering, file operations, and advanced matching techniques.

## Methods Overview

### 1. Saving and Loading Data

#### `save_df_to_csv(df, file_name)`

Saves the DataFrame to a CSV file, excluding the index.

- **Parameters**:
  - `df`: DataFrame to save.
  - `file_name`: Name of the file to save the DataFrame to.

- **Example**:
  ```python
  DataFrameOps.save_df_to_csv(df, 'data.csv')
csv_to_df(file_name)
Loads a DataFrame from a CSV file.

Parameters:

file_name: Name of the CSV file to load.
Example:

df = DataFrameOps.csv_to_df('data.csv')
df_to_excel(df, file_name)
Saves the DataFrame to an Excel file, excluding the index.

Parameters:

df: DataFrame to save.
file_name: Name of the Excel file to save the DataFrame to.
Example:

DataFrameOps.df_to_excel(df, 'data.xlsx')
2. Filtering Data
filter_by_word(df, column, word, case_sensitive=False, exact_match=False, is_index=False)
Filters the DataFrame based on a word match in a specified column.

Parameters:

df: DataFrame to filter.
column: Column name or index to apply the filter on.
word: Word to match.
case_sensitive: If True, the match is case sensitive. Defaults to False.
exact_match: If True, only exact matches are returned. Defaults to False.
is_index: If True, column is treated as an index. Defaults to False.
Example:

filtered_df = DataFrameOps.filter_by_word(df, 'name', 'Alice', case_sensitive=True)
filter_by_value_range(df, column, min_val, max_val, inclusive=True)
Filters the DataFrame to include rows where the specified column's values are within a given range.

Parameters:

df: DataFrame to filter.
column: Column name to apply the filter on.
min_val: Minimum value of the range.
max_val: Maximum value of the range.
inclusive: If True, the range is inclusive of min_val and max_val.
Example:

filtered_df = DataFrameOps.filter_by_value_range(df, 'age', 20, 30)
filter_by_membership(df, column, values_list)
Filters the DataFrame to include rows where the specified column's value is in the provided list of values.

Parameters:

df: DataFrame to filter.
column: Column name to apply the filter on.
values_list: List of values to include.
Example:

filtered_df = DataFrameOps.filter_by_membership(df, 'department', ['HR', 'Sales'])
filter_by_date_range(df, date_column, start_date, end_date)
Filters the DataFrame to include rows where the date in the specified column is within the given date range.

Parameters:

df: DataFrame to filter.
date_column: Column name with datetime values.
start_date: Start date of the range (inclusive).
end_date: End date of the range (inclusive).
Example:

filtered_df = DataFrameOps.filter_by_date_range(df, 'hire_date', '2020-01-01', '2020-12-31')
(Continue in the next message with the rest of the methods...)


This part covers detailed explanations and examples for the first 8 methods of the `DataFrameOps` class. The format provides clarity on what each method does, how it can be called, and what parameters it expects, helping users to effectively utilize the class's functionalities.



Continuing the documentation for the remaining methods of your `DataFrameOps` class:

```markdown
### 3. Advanced Filtering and Matching

#### `filter_conditions(df, conditions, is_index=False)`

Filters the DataFrame based on multiple specified conditions.

- **Parameters**:
  - `df`: DataFrame to filter.
  - `conditions`: A list of conditions, each represented as a tuple (`column`, `operator`, `value`).
  - `is_index`: If `True`, `column` in conditions is treated as an index.

- **Example**:
  ```python
  conditions = [('age', '>', 30), ('department', '==', 'Sales')]
  filtered_df = DataFrameOps.filter_conditions(df, conditions)
exclude_values(df, column, values_to_exclude, is_index=False)
Excludes rows from the DataFrame based on specific values in a given column.

Parameters:

df: DataFrame to filter.
column: Column name or index to apply the exclusion on.
values_to_exclude: A value or list of values to be excluded.
is_index: If True, column is treated as an index.
Example:

excluded_df = DataFrameOps.exclude_values(df, 'status', ['Inactive', 'Suspended'])
filter_starts_with(df, column, pattern, is_index=False, case_sensitive=True)
Filters rows where values in the specified column start with the given pattern.

Parameters:

Similar to exclude_values, with pattern specifying the start string.
Example:

starts_with_df = DataFrameOps.filter_starts_with(df, 'name', 'Jo', case_sensitive=False)
filter_ends_with(df, column, pattern, is_index=False, case_sensitive=True)
Filters rows where values in the specified column end with the given pattern.

Parameters:

Similar to filter_starts_with, focusing on the end of the string.
Example:

ends_with_df = DataFrameOps.filter_ends_with(df, 'email', '@example.com')
filter_top_n(df, column, n, is_index=False, ascending=False)
Returns a DataFrame with the top N items sorted by the specified column.

Parameters:

df: DataFrame to filter.
column: Column name or index used for sorting to select top N items.
n: Number of top items to keep.
is_index: If True, column is treated as an index.
ascending: Sort ascending vs. descending.
Example:

top_n_df = DataFrameOps.filter_top_n(df, 'sales', 5)
match_by_conditions(df1, df2, match_column, additional_conditions=[], is_index=False)
Matches rows between two DataFrames based on a common column and additional conditions.

Parameters:

df1, df2: DataFrames to match.
match_column: The common column to match on.
additional_conditions: Additional conditions for matching.
is_index: If True, match_column is an index.
Example:

matched_df = DataFrameOps.match_by_conditions(df1, df2, 'employee_id', [('department', '==', 'Sales')])
match_and_return_from_with_conditions(df1, df2, match_column, return_from, additional_conditions=[], is_index=False)
Matches rows between two DataFrames based on a common column and additional conditions but returns only the matching rows from one of the specified DataFrames.

Parameters:

Similar to match_by_conditions, with return_from specifying the DataFrame to return rows from (df1 or df2).
Example:

matched_from_df1 = DataFrameOps.match_and_return_from_with_conditions(df1, df2, 'employee_id', 'df1', [('age', '>', 30)])
non_matching_rows_with_conditions(df1, df2, match_column, additional_conditions=[])
Finds rows in both DataFrames that do not have a match based on the specified column and additional conditions.

Parameters:

Similar to match_by_conditions, but focuses on non-matching rows.
Example:

non_matching_df1, non_matching_df2 = DataFrameOps.non_matching_rows_with_conditions(df1, df2, 'employee_id', additional_conditions=[('status', '==', 'Active')])
non_matching_from_with_conditions(df1, df2, match_column, return_from, additional_conditions=[])
Finds non-matching rows


