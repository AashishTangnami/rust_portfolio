import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


def feature_classification(X,y, no_samples):
    X, y = make_classification(
        n_samples=100_000, 
        n_features=20, 
        n_informative=2, 
        n_redundant=2, 
        random_state=42
    )
    return X, y

def train_test_split(X, y): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.33, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


