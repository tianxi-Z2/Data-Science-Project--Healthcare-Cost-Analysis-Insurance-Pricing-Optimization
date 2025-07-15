import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Key Steps:

# Load the Kaggle dataset (insurance.csv).
# Confirm no missing values and appropriate data types.
# Cap extreme values in charges at the 99th percentile.
# One-hot encode categorical features (smoker_yes, region_southwest, etc.).

# Load data
df = pd.read_csv('data/insurance.csv')

# Check missing values and data types
print("Missing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Handle outliers in 'charges' (top 1%)
Q99 = df['charges'].quantile(0.99)
df = df[df['charges'] <= Q99]

# Convert categorical variables
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print(df.head())