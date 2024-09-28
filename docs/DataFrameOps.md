
# DataFrameOps Class Documentation

## Overview
The `DataFrameOps` class provides a set of utility functions for common operations on pandas DataFrames. This class includes methods for file operations, filtering, matching, grouping, and basic visualization. It is designed to streamline repetitive tasks when working with pandas.

### Key Features
- Save and load DataFrames from CSV and Excel files.
- Filter rows based on value ranges, membership, or patterns.
- Perform advanced filtering with conditions and exclusions.
- Match rows between two DataFrames based on one or more conditions.
- Create simple visualizations like histograms and scatter plots.

## Class Methods

### File Operations
1. **`save_df_to_csv(df, file_name, verbose=False)`**: Save a DataFrame to a CSV file.
   - **Parameters**:
     - `df (pd.DataFrame)`: DataFrame to save.
     - `file_name (str)`: File path to save the CSV.
     - `verbose (bool)`: Print a success message if True.
   - **Returns**: None.

2. **`csv_to_df(file_name, verbose=False)`**: Load a DataFrame from a CSV file.
   - **Parameters**:
     - `file_name (str)`: File path to load the CSV from.
     - `verbose (bool)`: Print a success message if True.
   - **Returns**: Loaded `pd.DataFrame` or None if an error occurs.

3. **`df_to_excel(df, file_name, verbose=False)`**: Save a DataFrame to an Excel file.
   - **Parameters**:
     - `df (pd.DataFrame)`: DataFrame to save.
     - `file_name (str)`: File path to save the Excel file.
     - `verbose (bool)`: Print a success message if True.
   - **Returns**: None.

### Filtering Methods
1. **`filter_by_value_range(df, column, min_val, max_val, inclusive=True, verbose=False)`**: Filter rows by a range of values in a specified column.
2. **`filter_by_membership(df, column, values_list, verbose=False)`**: Filter rows by membership in a list of values in a specified column.
3. **`filter_by_date_range(df, date_column, start_date, end_date, verbose=False)`**: Filter rows by a date range.
4. **`filter_conditions(df, conditions, is_index=False, verbose=False)`**: Filter rows based on multiple conditions.
5. **`exclude_values(df, column, values_to_exclude, verbose=False)`**: Exclude rows from a DataFrame based on values.
6. **`filter_starts_with(df, column, pattern, verbose=False)`**: Filter rows where a column starts with a given pattern.
7. **`filter_ends_with(df, column, pattern, verbose=False)`**: Filter rows where a column ends with a given pattern.

### Matching and Merging
1. **`match_by_conditions(df1, df2, match_column, additional_conditions=[], verbose=False)`**: Match rows from two DataFrames based on a column and additional conditions.
2. **`non_matching_rows_with_conditions(df1, df2, match_column, additional_conditions=[], verbose=False)`**: Get non-matching rows from the first DataFrame based on a column and additional conditions.

### Visualization Methods
1. **`plot_histogram(df, column)`**: Plot a histogram of a column.
2. **`plot_scatter(df, x_column, y_column)`**: Plot a scatter plot of two columns.

## Usage Example
```python
import pandas as pd
from DataForge.dataframops import DataFrameOps

# Sample DataFrame
data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
df = pd.DataFrame(data)

# Save DataFrame to CSV
DataFrameOps.save_df_to_csv(df, "sample.csv", verbose=True)

# Load DataFrame from CSV
loaded_df = DataFrameOps.csv_to_df("sample.csv", verbose=True)
print("Loaded DataFrame:", loaded_df)

# Filter rows by a range of values
filtered_df = DataFrameOps.filter_by_value_range(df, column='A', min_val=1, max_val=3, verbose=True)
print("Filtered DataFrame:", filtered_df)
```

## Additional Notes
- Ensure that columns specified in filtering methods exist in the DataFrame.
- Use the `verbose` flag to get feedback on the operations being performed.
