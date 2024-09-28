
# DataCleaner Class Documentation

## Overview
The `DataCleaner` class provides a variety of methods for cleaning and preprocessing data in a pandas DataFrame. This includes handling missing values, removing outliers, encoding categorical variables, scaling, and text processing.

### Key Features
- Handle missing values using multiple strategies (mean, median, mode, KNN, etc.).
- Detect and remove outliers using different methods (IQR, Z-score, Isolation Forest).
- Normalize and scale data using robust and quantile scaling techniques.
- Clean and preprocess text data.
- Advanced imputation methods like iterative imputation.

## Class Methods

### Missing Value Handling
1. **`fill_missing_values(df, strategy='mean', columns=None, verbose=False)`**: Fill missing values using a specified strategy.
   - **Parameters**:
     - `df (pd.DataFrame)`: Input DataFrame.
     - `strategy (str)`: Strategy to use ('mean', 'median', 'mode', 'constant').
     - `columns (list)`: List of columns to apply imputation to.
     - `verbose (bool)`: Print summary if True.

2. **`knn_impute(df, n_neighbors=5, verbose=False)`**: Impute missing values using K-Nearest Neighbors.
3. **`iterative_impute(df, max_iter=10, verbose=False)`**: Perform iterative imputation to handle missing values.

### Outlier Detection
1. **`remove_outliers(df, method='IQR', factor=1.5, verbose=False)`**: Remove outliers based on the IQR method.
2. **`isolation_forest_outlier_detection(df, contamination=0.05, verbose=False)`**: Detect and remove outliers using the Isolation Forest method.

### Categorical Data Handling
1. **`encode_categorical(df, method='onehot', verbose=False)`**: Encode categorical variables using one-hot or label encoding.

### Text Data Cleaning
1. **`clean_text_data(df, columns, verbose=False)`**: Clean text data by trimming whitespace and removing special characters.
2. **`remove_stopwords(df, column, stop_words=None, verbose=False)`**: Remove stopwords from text data.
3. **`text_vectorization(df, columns, method='tfidf', verbose=False)`**: Convert text columns into vectors using TF-IDF or Count Vectorizer.

## Usage Example
```python
import pandas as pd
from DataForge.datacleaner import DataCleaner

# Create a sample DataFrame
data = {'Text': ['Hello, world!', 'Data cleaning is fun!']}
df = pd.DataFrame(data)

# Clean text data
cleaned_df = DataCleaner.clean_text_data(df, columns=['Text'], verbose=True)
print("Cleaned DataFrame:
", cleaned_df)

# Encode categorical data
encoded_df = DataCleaner.encode_categorical(cleaned_df, method='label', verbose=True)
print("Encoded DataFrame:
", encoded_df)
```

## Additional Notes
- Use appropriate missing value handling methods based on the data characteristics.
- Choose the correct outlier detection method depending on the distribution of data.
