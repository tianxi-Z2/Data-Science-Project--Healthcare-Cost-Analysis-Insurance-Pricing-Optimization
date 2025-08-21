import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
# Load data
df = pd.read_csv("C:/Users/yoooE/Desktop/insurance.csv")

# Check missing values and data types
print("Missing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# 1. Drop rows with any missing values``
df = df.dropna()

# 2. Fill missing numerical values with mean or median
# df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# 3. Fill missing categorical values with mode
# df['column_name'] = df['column_name'].fillna(df['column_name'].mode()[0])

# 4. Fill all missing values with a constant
# df = df.fillna(0)

#Handle outliers in 'charges' (top 1%) -remove values above 99th percentile
percentile_99 = df['charges'].quantile(0.99)

df_filted = df[df['charges']<= percentile_99]
#print(df_filted)