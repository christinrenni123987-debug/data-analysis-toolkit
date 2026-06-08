import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def missing_value_analysis(df):
    return pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Missing Percentage": (df.isnull().sum() / len(df)) * 100
    })

def clean_data(df):
    df_clean = df.copy().drop_duplicates()
    for col in df_clean.select_dtypes(include=np.number).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in df_clean.select_dtypes(exclude=np.number).columns:
        if not df_clean[col].mode().empty:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    return df_clean

def descriptive_statistics(data):
    return pd.DataFrame({
        "Statistic": ["Count", "Mean", "Median", "Mode", "Standard Deviation", "Variance", "Minimum", "Maximum", "Q1", "Q3", "Skewness", "Kurtosis"],
        "Value": [data.count(), data.mean(), data.median(), data.mode()[0], data.std(), data.var(), data.min(), data.max(), data.quantile(0.25), data.quantile(0.75), data.skew(), data.kurtosis()]
    })

def normality_tests(data):
    shapiro = stats.shapiro(data)
    ks = stats.kstest(data, "norm", args=(data.mean(), data.std()))
    anderson = stats.anderson(data, dist="norm")
    return shapiro, ks, anderson

def confidence_interval(data, confidence=0.95):
    return stats.t.interval(confidence, len(data)-1, loc=data.mean(), scale=stats.sem(data))

def one_sample_t_test(data, hypothesized_mean):
    return stats.ttest_1samp(data, hypothesized_mean)
