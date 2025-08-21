# src/high_risk_identification.py
import pandas as pd
import numpy as np
import json
import os


def identify_high_risk_users(input_path='../data/insurance_cleaned.csv',
                             output_path='reports/high_risk_users.csv',
                             config_path='reports/risk_config.json'):
    """
    Identify high-risk users using SHAP-based thresholds and compute risk scores

    Parameters:
    - input_path: Path to insurance data
    - output_path: Output CSV path
    - config_path: Path to save threshold config
    """
    df = pd.read_csv(input_path)

    # Convert smoker to binary (if not yet)
    if df['smoker'].dtype == object:
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    # Remove outliers (99th percentile)
    cost_threshold = df['charges'].quantile(0.99)
    df = df[df['charges'] <= cost_threshold]

    # Thresholds based on SHAP features
    high_cost_group = df[df['charges'] >= df['charges'].quantile(0.9)]
    thresholds = {
        'age': {
            'threshold': high_cost_group['age'].median(),
            'charge_multiplier': 1.5
        },
        'bmi': {
            'threshold': 30,
            'charge_multiplier': 1.2
        },
        'smoker': {
            'threshold': 1,
            'charge_multiplier': 2.5
        }
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    # Save thresholds
    with open(config_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    # Dynamic high-risk mask
    high_risk_mask = (
            (df['age'] > thresholds['age']['threshold']) |
            (df['bmi'] > thresholds['bmi']['threshold']) |
            (df['smoker'] == thresholds['smoker']['threshold'])
    )

    # Get high-risk users
    high_risk_users = df[high_risk_mask].copy()

    # Calculate risk score (weighted sum, normalize age and bmi)
    high_risk_users['risk_score'] = (
            (high_risk_users['age'] / 70) * 0.4 +
            (high_risk_users['bmi'] / 40) * 0.3 +
            (high_risk_users['smoker']) * 0.3
    )

    # Get top 10% users by risk_score
    top_10_percent = high_risk_users.nlargest(int(len(high_risk_users) * 0.1), 'risk_score')
    top_10_percent.to_csv(output_path, index=False)

    print(f"Identified {len(top_10_percent)} high-risk users")
    return top_10_percent


if __name__ == "__main__":
    identify_high_risk_users()