import pandas as pd

def missing_data_handler(df, column_name):
    return df[column_name].isna()
