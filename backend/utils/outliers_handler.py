import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.linear_model import HuberRegressor

def inter_quartile_range(df, column_name):

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].copy()
    return outliers

def z_score(df, column_name, thres):
    # Calculate the z-score for each value in the column
    z = np.abs(stats.zscore(df['column_name']))

    # Define a threshold (e.g. 3 standard deviations)
    

    # Find the indices of the values that are above the threshold
    outliers = np.where(z > thres)

    # Replace the outliers with the median value of the column
    df[column_name][outliers] = df[column_name].median()
    return df

# Some more techniques
# Local Outlier Factor: The Local Outlier Factor (LOF) is a technique that measures the degree of abnormality of each observation based on the local density of nearby observations. This technique is appropriate when the outliers are in a high-dimensional space or when the distribution of the data is not well-defined.
def local_outlier_factor(df, X):
    n_outliers = len(df)
    ground_truth = np.ones(len(X), dtype=int)
    ground_truth[-n_outliers:] = -1
    # Create an instance of the LOF class
    lof = LocalOutlierFactor(n_neighbors=20)

    # Fit the LOF model to your data and compute the LOF scores
    lof_scores = lof.fit_predict(df[X])

    # The LOF scores will be negative for outliers and positive for inliers
    # You can use the absolute value to get the magnitude of the outlier score
    lof_scores = abs(lof_scores)
    y_pred = lof.fit_predict(X)
    n_errors = (y_pred != ground_truth).sum()
    X_scores = lof.negative_outlier_factor_

    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
    # plot circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    plt.scatter(
        X[:, 0],
        X[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.xlabel("prediction errors: %d" % (n_errors))
    legend = plt.legend(loc="upper left")
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()

# Isolation Forest: Isolation Forest is a machine learning algorithm that is specifically designed to identify outliers. It works by constructing isolation trees that isolate the outlier observations from the rest of the data. This technique is appropriate when the dataset is large and the outliers are difficult to identify using other techniques.
def isolation_forest(df, X):
    # Create an instance of the IsolationForest class
    isof = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.1), max_features=1.0)

    # Fit the Isolation Forest model to your data and predict the outliers
    isof.fit(X)
    outliers = isof.predict(X)

    # The outliers will be labeled as -1 in the 'outliers' array
    outlier_indices = np.where(outliers == -1)[0]
# DBSCAN Clustering: Density-Based Spatial Clustering of Applications with Noise is a clustering algorithm that can be used to identify outliers based on the density of data points in the vicinity of each point.


def dbscan_clustering(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_
    return labels

# his technique is appropriate when the outliers are in a high-dimensional space or when the distribution of the data is not well-defined
# Robust Regression: techinque that uses a modified loss function to minmize the effect of outliers on the regression coefficients. Should be used when the outliers are affection the regression analysis.
def robust_regression(X, y):
    huber = HuberRegressor()
    huber.fit(X, y)
    y_pred = huber.predict(X)
    return y_pred