import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot2')
import seaborn as sns

def feature_distribution(df, column_name, n):
    df[column_name].value_counts().max()
    ax = df[column_name].value_counts.head(n).plot(kind='bar', 
                                                   title = f'Top {n}, {column_name}')
    ax.set_xlabel(f'{column_name}')
    ax.set_ylabel("Count")
    ax.plot()

def feature_histogram(df, column_name):
    ax = df[column_name].plot(kind='hist', 
                              bins =20, 
                              title =f'{column_name}')
    ax.set_xlabel(f'{column_name}')
    ax.plot()

def feature_histogram_density(df, column_name):
    ax = df[column_name].plot(kind='kde', 
                              title =f'{column_name}')
    ax.set_xlabel(f'{column_name}')
    ax.plot()

def features_scatter_plot(df, feature_a, feature_b, hue_color):
    ax = sns.scatterplot(
            x = feature_a,
            y = feature_b,
            data = df,
            hue = hue_color,
            title = f'{feature_a} vs {feature_b}')
    ax.plot()
    
def features_pairplot(df, column_name_list, hue_color):
    ax = sns.pairplot(data =df, 
                      vars = column_name_list, 
                      hue = hue_color)
    ax.plot()

def features_correlation(df):
    df = df.corr()
    ax = sns.heatmap(df, annot = True)
    