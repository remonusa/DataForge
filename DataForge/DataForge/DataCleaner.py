import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer

class DataCleaner:
        """ Class to clean and preprocess data. """


        @staticmethod
        def fill_missing_values(df, strategy='mean', columns=None):
            """ Fill missing values in DataFrame using specified strategy. """
            df = df.copy()
            if columns is None:
                columns = df.columns

            for column in columns:
                if strategy == 'mean':
                    df[column] = df[column].fillna(df[column].mean())
                elif strategy == 'median':
                    df[column] = df[column].fillna(df[column].median())
                elif strategy == 'mode':
                    mode_value = df[column].mode()
                    if not mode_value.empty:
                        df[column] = df[column].fillna(mode_value[0])
                elif strategy == 'constant':
                    df[column] = df[column].fillna(0)  # Or a specified constant value
            return df

        @staticmethod
        def remove_outliers(df, method='IQR', factor=1.5):
            """ Remove outliers from DataFrame based on IQR. """
            df = df.copy()
            if method == 'IQR':
                Q1 = df.quantile(0.25)
                Q3 = df.quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df < (Q1 - factor * IQR)) | (df > (Q3 + factor * IQR))).any(axis=1)]
            elif method == 'Z-score':
                df = df[(np.abs(stats.zscore(df)) < factor).all(axis=1)]
            return df

        @staticmethod
        def encode_categorical(df, method='onehot'):
            """ Encode categorical variables using one-hot or label encoding. """
            if method == 'onehot':
                return pd.get_dummies(df, drop_first=True)
            elif method == 'label':
                for column in df.select_dtypes(include=['object']).columns:
                    df[column] = LabelEncoder().fit_transform(df[column])
                return df

        @staticmethod
        def normalize_data(df, method='min-max'):
            """ Normalize data in DataFrame using min-max or z-score normalization. """
            if method == 'min-max':
                scaler = MinMaxScaler()
                df[df.columns] = scaler.fit_transform(df[df.columns])
            elif method == 'z-score':
                df[df.columns] = df[df.columns].apply(zscore)
            return df
        @staticmethod
        def knn_impute(df, n_neighbors=5):
            """Impute missing values using K-Nearest Neighbors.
            
            :param df: DataFrame to process.
            :param n_neighbors: Number of neighbors to use for imputation.
            :return: DataFrame with missing values imputed.
            """
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            return df_filled
        @staticmethod
        def convert_to_datetime(df, columns):
            """Convert columns in a DataFrame to datetime objects.
            
            :param df: DataFrame to process.
            :param columns: List of column names to convert.
            :return: DataFrame with specified columns converted to datetime.
            """
            for column in columns:
                df[column] = pd.to_datetime(df[column], errors='coerce')
            return df
        @staticmethod
        def remove_duplicates(df, columns=None, keep='first'):
            """Remove duplicate rows from DataFrame based on specified columns.
            
            :param df: DataFrame to process.
            :param columns: Columns to consider for identifying duplicates. If None, considers all columns.
            :param keep: Which duplicate to keep ('first', 'last', 'none').
            :return: DataFrame with duplicates removed.
            """
            return df.drop_duplicates(subset=columns, keep=keep)
        @staticmethod
        def clean_text_data(df, columns):
            """Clean text data in specified columns by trimming whitespace and removing special characters.
            
            :param df: DataFrame to process.
            :param columns: Columns to clean.
            :return: DataFrame with clean text data.
            """
            for column in columns:
                df[column] = df[column].str.strip()
                df[column] = df[column].str.replace(r"[^\w\s]", '', regex=True)
            return df



        @staticmethod
        def robust_scale(df, columns):
            """Scale data using RobustScaler, which is less sensitive to outliers.
            
            :param df: DataFrame to process.
            :param columns: List of columns to scale.
            :return: DataFrame with scaled data.
            """
            scaler = RobustScaler()
            df[columns] = scaler.fit_transform(df[columns])
            return df

        @staticmethod
        def quantile_transform(df, columns, output_distribution='normal'):
            """Transform data to follow a normal or uniform distribution using QuantileTransformer.
            
            :param df: DataFrame to process.
            :param columns: List of columns to transform.
            :param output_distribution: The distribution to transform to ('normal', 'uniform').
            :return: DataFrame with transformed data.
            """
            transformer = QuantileTransformer(output_distribution=output_distribution)
            df[columns] = transformer.fit_transform(df[columns])
            return df



