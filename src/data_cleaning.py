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

# handle missing values if any (not present in this dataset)
# Example ways to handle missing values:
# 1. Drop rows with any missing values``
# df = df.dropna()

# 2. Fill missing numerical values with mean or median
# df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# 3. Fill missing categorical values with mode
# df['column_name'] = df['column_name'].fillna(df['column_name'].mode()[0])

# 4. Fill all missing values with a constant
# df = df.fillna(0)

# Todo: Handle outliers in 'charges' (top 1%) -remove values above 99th percentile


# Optional practice: Convert categorical variables usig one-hot encoding
# Additional data cleaning steps that can be applied:
# - Normalize or standardize numerical features (e.g., age, bmi, charges) for certain models.
# - Detect and handle duplicate rows if present.
# - Bin continuous variables (e.g., age groups, bmi categories) for feature engineering.
# - Create interaction features (e.g., smoker*bmi).
# - Encode ordinal features if any (not present in this dataset, but useful for others).
# - Impute missing values if found in other datasets.
# - Remove or flag inconsistent or impossible values (e.g., negative ages).
# - Feature scaling for algorithms sensitive to magnitude.