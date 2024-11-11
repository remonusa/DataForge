
# DataForge

## Overview
**DataForge** is a powerful data manipulation and preprocessing library designed for data engineers, data scientists, and machine learning practitioners. It provides a variety of data cleaning and transformation methods, making it a one-stop solution for data preparation tasks. With easy-to-use classes and methods, `DataForge` enables seamless data handling, from basic DataFrame operations to advanced data cleaning.

### Key Features
- **DataFrame Operations**: Merge, filter, and manipulate DataFrames with ease.
- **Data Cleaning**: Handle missing values, remove outliers, encode categorical variables, and more.

## Installation
### Local Installation (Manual Setup)
Since the library is not available on PyPI, you can install it manually using the following steps:

1. **Clone the repository:**
```bash
git clone https://github.com/remonusa/DataForge.git
```

2. **Navigate to the DataForge directory:**
```bash
cd DataForge
```

3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

4. **Import the library into your scripts by referencing the local path:**
```python
from DataForge.dataframops import DataFrameOps
from DataForge.datacleaner import DataCleaner
```

### Dependencies
Ensure you have the following libraries installed:
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`

If any dependency is missing, you can install it using:
```bash
pip install <library_name>
```

## Library Structure
DataForge currently consists of two main classes:

1. **`DataFrameOps`**: Provides utility functions for handling common DataFrame operations, such as merging, filtering, and aggregation.
2. **`DataCleaner`**: Includes a variety of methods for cleaning and preprocessing data, such as missing value imputation, outlier detection, and categorical encoding.

## Quickstart Guide

### 1. Using `DataFrameOps`
The `DataFrameOps` class provides utilities for common operations on DataFrames. Below is an example of how to use some of its methods:

```python
from DataForge.dataframops import DataFrameOps
import pandas as pd

# Sample DataFrame
data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
df = pd.DataFrame(data)

# Example: Saving and Loading DataFrames
DataFrameOps.save_df_to_csv(df, "sample.csv", verbose=True)
loaded_df = DataFrameOps.csv_to_df("sample.csv", verbose=True)

# Example: Filtering DataFrames by Value Range
filtered_df = DataFrameOps.filter_by_value_range(df, column='A', min_val=1, max_val=3, verbose=True)
print("Filtered DataFrame:", filtered_df)
```

### 2. Using `DataCleaner`
The `DataCleaner` class provides a comprehensive set of methods for cleaning data. Here’s how to use some of its key methods:

#### Handling Missing Values
```python
from DataForge.datacleaner import DataCleaner
import pandas as pd
import numpy as np

# Create a sample DataFrame with missing values
data = {'A': [1, 2, np.nan, 4], 'B': [10, np.nan, 30, 40]}
df = pd.DataFrame(data)

# Fill missing values using the mean strategy
filled_df = DataCleaner.fill_missing_values(df, strategy='mean', verbose=True)
print("Filled DataFrame:
", filled_df)
```

#### Removing Outliers
```python
# Remove outliers using the IQR method
cleaned_df = DataCleaner.remove_outliers(filled_df, method='IQR', factor=1.5, verbose=True)
print("DataFrame after removing outliers:
", cleaned_df)
```

## Detailed API Reference

### DataFrameOps Class
| Method Name                                   | Description                                            |
|-----------------------------------------------|--------------------------------------------------------|
| `save_df_to_csv(df, file_name, verbose)`       | Save a DataFrame to a CSV file.                         |
| `csv_to_df(file_name, verbose)`                | Load a DataFrame from a CSV file.                       |
| `df_to_excel(df, file_name, verbose)`          | Save a DataFrame to an Excel file.                      |
| `filter_by_word(df, column, word, ...)`        | Filter DataFrame rows by a word in a column.            |
| `filter_by_value_range(df, column, ...)`       | Filter DataFrame rows by a range of values.             |
| `filter_by_membership(df, column, ...)`        | Filter DataFrame rows by membership in a list.          |
| `filter_by_date_range(df, column, ...)`        | Filter DataFrame rows by a range of dates.              |
| `filter_conditions(df, conditions, ...)`       | Filter DataFrame rows based on multiple conditions.     |
| `exclude_values(df, column, values, ...)`      | Exclude rows from a DataFrame based on column values.   |
| `filter_starts_with(df, column, ...)`          | Filter rows where a column starts with a pattern.       |
| `filter_ends_with(df, column, ...)`            | Filter rows where a column ends with a pattern.         |
| `filter_top_n(df, column, n, ...)`             | Filter the top N rows of a DataFrame.                   |
| `match_by_conditions(df1, df2, ...)`           | Match rows from two DataFrames based on conditions.     |
| `non_matching_rows_with_conditions(df1, ...)`  | Get non-matching rows from the first DataFrame.         |

### DataCleaner Class
| Method Name                                 | Description                                            |
|---------------------------------------------|--------------------------------------------------------|
| `fill_missing_values(df, strategy, columns)` | Fill missing values using various strategies.          |
| `remove_outliers(df, method, factor)`        | Remove outliers based on IQR or Z-score.               |
| `encode_categorical(df, method)`             | Encode categorical variables using one-hot or label.   |
| `normalize_data(df, method)`                 | Normalize data using min-max or z-score normalization. |
| `knn_impute(df, n_neighbors)`                | Impute missing values using KNN.                       |
| `convert_to_datetime(df, columns)`           | Convert specified columns to datetime format.          |
| `remove_duplicates(df, columns, keep)`       | Remove duplicate rows.                                 |
| `clean_text_data(df, columns)`               | Clean text data by trimming whitespace and removing special characters. |
| `robust_scale(df, columns)`                  | Scale data using RobustScaler.                         |
| `iterative_impute(df, max_iter)`             | Perform iterative imputation.                          |

## Coming Soon
We are actively working on adding more classes to `DataForge`:

1. **`FeatureEngineering`**: Advanced feature transformation and creation.
2. **`TimeSeriesOps`**: Comprehensive methods for time-series analysis.
3. **`TextProcessor`**: Advanced text preprocessing techniques.

Stay tuned for future updates!

## Contributing
We welcome contributions from the community! If you would like to contribute, please feel free to open issues or submit pull requests.


This code is proprietary and not open for redistribution, modification, or use without explicit permission from the author.

