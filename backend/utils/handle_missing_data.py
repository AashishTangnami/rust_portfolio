import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer

def missing_data_handler(df, column_name):
    return df[column_name].isna()

def mean_imputation(df, column_name):
    """Imputes missing values in a column using the mean of the non-missing values"""
    
    mean_value = df[column_name].mean()
    df[column_name].fillna(mean_value, inplace=True)
    
    return df


def multiple_imputation(df, variable_names, num_imputations):
    """Performs multiple imputation on a dataframe using the IterativeImputer algorithm"""
    
    # Convert the dataframe to a numpy array
    data_array = df.to_numpy()
    
    # Initialize the imputer with the IterativeImputer algorithm
    imputer = IterativeImputer(max_iter=10, random_state=0)
    
    # Perform multiple imputations on the data array
    imputed_data = imputer.fit_transform(data_array)
    for i in range(1, num_imputations):
        imputed_data = np.dstack((imputed_data, imputer.fit_transform(data_array)))
    
    # Convert the imputed data back to a dataframe
    imputed_dataframe = pd.DataFrame(np.mean(imputed_data, axis=2), columns=df.columns)
    
    # Return the imputed dataframe
    return imputed_dataframe
