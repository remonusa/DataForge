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
