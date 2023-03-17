import pandas as pd

def shape_of_dataset(df):
    return df.shape

def df_head(df, n):
    return df.head(n)

def df_tail(df, n):
    return df.tail(n)

def df_missing_value(df):
    return df.isna().sum()

def df_dtypes(df):
    return df.dtypes.astype(str)

def df_describe(df):
    return df.describe()

def df_numeric(df, column_name):
    return pd.to_numeric(df[column_name])

def df_rename(df, column_name, rename_column):
    return df.rename(columns = {"column_name" : "rename_column"})

def down_cast_integer_values(df, column_name):
    return pd.Series(df[column_name], downcast="integer")

def down_cast_float_values(df, column_name):
    return pd.Series(df[column_name], downcast="float")

def check_date(df, column_name):
    return pd.to_datetime(df[column_name])
