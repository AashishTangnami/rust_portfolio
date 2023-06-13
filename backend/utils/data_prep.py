import pandas as pd
#return the shape of data
def shape_of_dataset(df):
    """
    This function returns the shape of the dataset.
    """  
    return df.shape

def df_head(df, n):
    """
    This function returns the first n rows of the dataset.
    """
    return df.head(n)

def df_tail(df, n):
    """
    This function returns the last n rows of the dataset.
    """
    return df.tail(n)

def df_missing_value(df):
    """
    This function returns the missing values in the dataset.
    """
    return df.isna().sum()

def df_dtypes(df):
    """
    This function returns the data types of the dataset.
    """
    return df.dtypes.astype(str)

def df_describe(df):
    """
    This function returns the descriptive statistics of the dataset.
    """
    return df.describe()

def df_numeric(df, column_name):
    """
    This function returns the numeric values of the dataset.
    """
    return pd.to_numeric(df[column_name])

def df_rename(df, column_name, rename_column):
    """
    This function returns the renamed column of the dataset.
    """
    return df.rename(columns = {"column_name" : "rename_column"})

def down_cast_integer_values(df, column_name):
    """
    This function returns the downcasted integer values of the dataset.
    """
    return pd.Series(df[column_name], downcast="integer")

def down_cast_float_values(df, column_name):
    """This function returns the downcasted float values of the dataset.
    """
    return pd.Series(df[column_name], downcast="float")

def check_date(df, column_name):
    """
    This function returns the date of the dataset.
    """
    return pd.to_datetime(df[column_name])
