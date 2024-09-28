# Standard library imports
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Statsmodels imports for time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose


class DataCleaner:
    """
    Class to clean and preprocess data, including missing values imputation,
    encoding categorical variables, outlier detection, normalization, and text processing.
    Includes methods for input validation to ensure correct data usage.
    """

    # 0.1 Validate if the input is a DataFrame and non-empty
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

    # 0.2 Validate if the specified columns exist in the DataFrame
    @staticmethod
    def _validate_columns(df, columns):
        """
        Validate that the specified columns exist in the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to check.
            columns (list): List of columns to validate.

        Raises:
            KeyError: If any of the specified columns are not found in the DataFrame.
        """
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Columns not found in DataFrame: {missing_columns}")

    # 1. Fill Missing Values
    @staticmethod
    def fill_missing_values(df, strategy='mean', columns=None, constant_value=0, verbose=False):
        """
        Fill missing values in DataFrame using a specified strategy.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            strategy (str): Strategy to use for imputation - 'mean', 'median', 'mode', 'constant'.
            columns (list or None): List of columns to apply imputation. If None, applies to all columns.
            constant_value (Any): Value to use when strategy='constant'. Default is 0.
            verbose (bool): If True, prints the number of missing values filled for each column.

        Returns:
            pd.DataFrame: DataFrame with filled missing values.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        columns = columns or df.columns
        DataCleaner._validate_columns(df, columns)

        df = df.copy()
        filled_counts = {}  # Track how many missing values were filled in each column

        for column in columns:
            missing_before = df[column].isna().sum()
            if strategy == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif strategy == 'mode':
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value[0])
            elif strategy == 'constant':
                df[column] = df[column].fillna(constant_value)

            filled_counts[column] = missing_before - df[column].isna().sum()

        if verbose:
            print("1. Missing Values Filled:")
            for col, count in filled_counts.items():
                print(f" - Column '{col}': {count} missing values filled using strategy '{strategy}'")

        return df

    # 2. Remove Outliers
    @staticmethod
    def remove_outliers(df, method='IQR', factor=1.5, columns=None, verbose=False):
        """
        Remove outliers from DataFrame based on the specified method.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            method (str): Method to detect outliers ('IQR' or 'Z-score').
            factor (float): The factor to use for detecting outliers.
            columns (list or None): List of columns to consider for outlier detection. If None, uses all numeric columns.
            verbose (bool): If True, prints the number of outliers removed.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        columns = columns or df.select_dtypes(include=[np.number]).columns
        DataCleaner._validate_columns(df, columns)

        df = df.copy()
        initial_count = len(df)  # Record initial number of rows

        if method == 'IQR':
            Q1 = df[columns].quantile(0.25)
            Q3 = df[columns].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df[columns] < (Q1 - factor * IQR)) | (df[columns] > (Q3 + factor * IQR))).any(axis=1)
        elif method == 'Z-score':
            z_scores = np.abs(stats.zscore(df[columns]))
            mask = (z_scores < factor).all(axis=1)
        else:
            raise ValueError(f"Unsupported method '{method}'. Use 'IQR' or 'Z-score'.")

        df = df[mask]
        outliers_removed = initial_count - len(df)  # Calculate the number of rows removed

        if verbose:
            print(f"2. Outliers Removed:")
            print(f" - Method: '{method}' with factor {factor}")
            print(f" - Columns: {columns}")
            print(f" - Outliers removed: {outliers_removed}")

        return df

    # 3. Encode Categorical Variables
    @staticmethod
    def encode_categorical(df, method='onehot', drop_first=True, verbose=False):
        """
        Encode categorical variables using specified method.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            method (str): Encoding method ('onehot' or 'label').
            drop_first (bool): Whether to drop the first level in one-hot encoding to avoid multicollinearity.
            verbose (bool): If True, prints a summary of categorical columns encoded.

        Returns:
            pd.DataFrame: DataFrame with encoded categorical features.
        """
        # Validation
        DataCleaner._validate_dataframe(df)

        df = df.copy()
        if method == 'onehot':
            df = pd.get_dummies(df, drop_first=drop_first)
        elif method == 'label':
            for column in df.select_dtypes(include=['object']).columns:
                df[column] = LabelEncoder().fit_transform(df[column].astype(str))
                if verbose:
                    print(f"3. - Column '{column}' encoded using Label Encoding")
        else:
            raise ValueError(f"Unsupported encoding method '{method}'. Use 'onehot' or 'label'.")

        if verbose and method == 'onehot':
            print(f"3. One-hot encoded DataFrame with drop_first={drop_first}. New columns: {list(df.columns)}")

        return df

    # 4. Normalize Data
    @staticmethod
    def normalize_data(df, method='min-max', verbose=False):
        """
        Normalize data in DataFrame using min-max or z-score normalization.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            method (str): Normalization method ('min-max' or 'z-score').
            verbose (bool): If True, prints a summary of normalization applied.

        Returns:
            pd.DataFrame: DataFrame with normalized data.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        if not np.issubdtype(df.select_dtypes(include=[np.number]).dtypes[0], np.number):
            raise TypeError(f"All columns must be numeric for normalization.")

        df = df.copy()
        columns = df.columns

        if method == 'min-max':
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == 'z-score':
            df[columns] = df[columns].apply(zscore)
        else:
            raise ValueError(f"Unsupported normalization method '{method}'. Use 'min-max' or 'z-score'.")

        if verbose:
            print(f"4. Data normalized using {method} normalization for columns: {list(columns)}")

        return df

    # 5. KNN Imputation for Missing Values
    @staticmethod
    def knn_impute(df, n_neighbors=5, verbose=False):
        """
        Impute missing values using K-Nearest Neighbors.

        Parameters:
            df (pd.DataFrame): DataFrame to process.
            n_neighbors (int): Number of neighbors to use for imputation.
            verbose (bool): If True, prints the number of missing values filled.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        # Validation
        DataCleaner._validate_dataframe(df)

        imputer = KNNImputer(n_neighbors=n_neighbors)
        missing_before = df.isna().sum().sum()  # Count missing values before imputation
        df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        missing_after = df_filled.isna().sum().sum()  # Count missing values after imputation

        if verbose:
            print(f"5. KNN Imputation completed with {n_neighbors} neighbors.")
            print(f" - Missing values before: {missing_before}, Missing values after: {missing_after}")

        return df_filled


    # 6. Convert to Datetime
    @staticmethod
    def convert_to_datetime(df, columns, verbose=False):
        """
        Convert columns in a DataFrame to datetime objects.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of column names to convert.
            verbose (bool): If True, prints the conversion details.
        
        Returns:
            pd.DataFrame: DataFrame with specified columns converted to datetime.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        DataCleaner._validate_columns(df, columns)

        df = df.copy()
        conversion_summary = {}  # Track conversion details

        for column in columns:
            initial_type = df[column].dtype  # Record the initial data type
            df[column] = pd.to_datetime(df[column], errors='coerce')
            num_converted = df[column].notna().sum()
            conversion_summary[column] = (initial_type, 'datetime64[ns]', num_converted)

        if verbose:
            print("6. Datetime Conversion Summary:")
            for col, (initial, new, count) in conversion_summary.items():
                print(f" - Column '{col}': Converted from {initial} to {new} with {count} successful conversions")

        return df

    # 7. Remove Duplicates
    @staticmethod
    def remove_duplicates(df, columns=None, keep='first', verbose=False):
        """
        Remove duplicate rows from DataFrame based on specified columns.

        Parameters:
            df (pd.DataFrame): DataFrame to process.
            columns (list or None): Columns to consider for identifying duplicates. If None, considers all columns.
            keep (str): Which duplicate to keep ('first', 'last', 'none').
            verbose (bool): If True, prints the number of duplicates removed.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        if columns:
            DataCleaner._validate_columns(df, columns)

        initial_count = len(df)  # Record initial number of rows
        df = df.drop_duplicates(subset=columns, keep=keep)
        duplicates_removed = initial_count - len(df)

        if verbose:
            print(f"7. Duplicates Removed:")
            print(f" - Columns considered: {columns if columns else 'All Columns'}")
            print(f" - Duplicates removed: {duplicates_removed}")

        return df

    # 8. Clean Text Data
    @staticmethod
    def clean_text_data(df, columns, verbose=False):
        """
        Clean text data in specified columns by trimming whitespace and removing special characters.

        Parameters:
            df (pd.DataFrame): DataFrame to process.
            columns (list): List of columns to clean.
            verbose (bool): If True, prints the columns that were cleaned.

        Returns:
            pd.DataFrame: DataFrame with clean text data.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        DataCleaner._validate_columns(df, columns)

        df = df.copy()
        for column in columns:
            df[column] = df[column].str.strip()
            df[column] = df[column].str.replace(r"[^\w\s]", '', regex=True)

        if verbose:
            print(f"8. Text Data Cleaned for columns: {columns}")

        return df

    # 9. Robust Scale
    @staticmethod
    def robust_scale(df, columns, verbose=False):
        """
        Scale data using RobustScaler, which is less sensitive to outliers.

        Parameters:
            df (pd.DataFrame): DataFrame to process.
            columns (list): List of columns to scale.
            verbose (bool): If True, prints the scaling details.

        Returns:
            pd.DataFrame: DataFrame with scaled data.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        DataCleaner._validate_columns(df, columns)

        df = df.copy()
        scaler = RobustScaler()
        df[columns] = scaler.fit_transform(df[columns])

        if verbose:
            print(f"9. Robust Scaling applied to columns: {columns}")

        return df

    # 10. Iterative Imputation for Missing Values
    @staticmethod
    def iterative_impute(df, max_iter=10, verbose=False):
        """
        Perform iterative imputation to handle missing values in the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame with missing values.
            max_iter (int): Maximum number of imputation iterations.
            verbose (bool): If True, prints the number of missing values filled.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed using IterativeImputer.
        """
        # Validation
        DataCleaner._validate_dataframe(df)

        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        missing_before = df.isna().sum().sum()  # Count missing values before imputation
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        missing_after = df_imputed.isna().sum().sum()  # Count missing values after imputation

        if verbose:
            print(f"10. Iterative Imputation completed with {max_iter} maximum iterations.")
            print(f" - Missing values before: {missing_before}, Missing values after: {missing_after}")

        return df_imputed
    

    # 11. Isolation Forest Outlier Detection
    @staticmethod
    def isolation_forest_outlier_detection(df, contamination=0.05, verbose=False):
        """
        Detect and remove outliers using the Isolation Forest method.

        Parameters:
            df (pd.DataFrame): Input DataFrame to detect outliers.
            contamination (float): The proportion of outliers in the data.
            verbose (bool): If True, prints the number of outliers removed.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        # Validation
        DataCleaner._validate_dataframe(df)

        iforest = IsolationForest(contamination=contamination, random_state=42)
        df['outlier'] = iforest.fit_predict(df)
        outliers_removed = len(df[df['outlier'] == -1])

        if verbose:
            print(f"11. Isolation Forest Outlier Detection:")
            print(f" - Outliers detected: {outliers_removed}")

        return df[df['outlier'] == 1].drop(columns=['outlier'])

    # 12. Seasonal Decomposition for Time-Series Analysis
    @staticmethod
    def time_series_decomposition(df, column, period=12, verbose=False):
        """
        Decompose a time-series column into trend, seasonal, and residual components.

        Parameters:
            df (DataFrame): Input DataFrame with time-series data.
            column (str): The name of the time-series column to decompose.
            period (int): The frequency of the time-series data.
            verbose (bool): If True, prints a summary of the decomposition.

        Returns:
            dict: A dictionary with 'trend', 'seasonal', and 'residual' components as separate DataFrames.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        DataCleaner._validate_columns(df, [column])

        decomposition = seasonal_decompose(df[column], period=period)

        if verbose:
            print(f"12. Time-Series Decomposition for column '{column}':")
            print(f" - Decomposed into trend, seasonal, and residual components.")

        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }

    # 13. Automatic Data Type Conversion
    @staticmethod
    def automatic_dtype_conversion(df, verbose=False):
        """
        Automatically detect and convert data types in the DataFrame.

        Parameters:
            df (DataFrame): Input DataFrame with columns to be converted.
            verbose (bool): If True, prints the conversion details.

        Returns:
            DataFrame: DataFrame with automatically detected and converted data types.
        """
        # Validation
        DataCleaner._validate_dataframe(df)

        conversion_summary = {}
        for column in df.columns:
            initial_type = df[column].dtype
            try:
                df[column] = pd.to_datetime(df[column], errors='coerce')
                if df[column].notna().sum() == 0:  # If all values are NaT after conversion, try numeric
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                conversion_summary[column] = (initial_type, df[column].dtype)
            except Exception as e:
                print(f"Column {column}: Error converting - {e}")
                continue

        if verbose:
            print("13. Automatic Data Type Conversion Summary:")
            for col, (initial, new) in conversion_summary.items():
                print(f" - Column '{col}': Converted from {initial} to {new}")

        return df

    # 14. Quantile Transformation for Normalization
    @staticmethod
    def quantile_transform(df, columns, output_distribution='normal', verbose=False):
        """
        Transform columns to follow a normal or uniform distribution using QuantileTransformer.

        Parameters:
            df (DataFrame): The input DataFrame.
            columns (list): List of columns to transform.
            output_distribution (str): Target distribution for transformation ('normal' or 'uniform').
            verbose (bool): If True, prints the transformation details.

        Returns:
            DataFrame: Transformed DataFrame.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        DataCleaner._validate_columns(df, columns)

        transformer = QuantileTransformer(output_distribution=output_distribution)
        df[columns] = transformer.fit_transform(df[columns])

        if verbose:
            print(f"14. Quantile Transformation applied to columns: {columns} with distribution '{output_distribution}'")

        return df

    # 15. Logarithmic Transformation
    @staticmethod
    def log_transform(df, columns, verbose=False):
        """
        Apply logarithmic transformation to specified columns to handle skewness.

        Parameters:
            df (DataFrame): Input DataFrame.
            columns (list): List of column names to transform.
            verbose (bool): If True, prints the transformation details.

        Returns:
            DataFrame: Transformed DataFrame with log-transformed columns.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        DataCleaner._validate_columns(df, columns)

        df = df.copy()
        for col in columns:
            df[col] = df[col].apply(lambda x: np.log1p(x) if np.issubdtype(type(x), np.number) and x > 0 else x)

        if verbose:
            print(f"15. Logarithmic Transformation applied to columns: {columns}")

        return df

    # 16. Z-Score Outlier Detection
    @staticmethod
    def zscore_outlier_detection(df, columns=None, threshold=3, verbose=False):
        """
        Detect and remove outliers using Z-Score method for specific columns.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            columns (list or None): Columns to detect outliers in. If None, uses all numeric columns.
            threshold (int): Z-Score threshold to identify outliers.
            verbose (bool): If True, prints the number of outliers detected.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        # Validation
        DataCleaner._validate_dataframe(df)
        columns = columns or df.select_dtypes(include=[np.number]).columns
        DataCleaner._validate_columns(df, columns)

        z_scores = np.abs(stats.zscore(df[columns]))
        outliers = (z_scores > threshold).any(axis=1)

        if verbose:
            print(f"16. Z-Score Outlier Detection for columns: {columns}")
            print(f" - Outliers detected: {outliers.sum()}")

        return df[~outliers]