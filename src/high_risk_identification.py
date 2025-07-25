# src/high_risk_identification.py
import pandas as pd
import numpy as np
import json

def identify_high_risk_users(input_path='data/insurance.csv', 
                           output_path='reports/high_risk_users.csv',
                           config_path='reports/risk_config.json'):
    """
    Identify high-risk users using data-driven thresholds
    
    Parameters:
    - input_path: Path to insurance data
    - output_path: Output CSV path
    - config_path: Path to save threshold config
    """
    df = pd.read_csv(input_path)
    
    # Remove outliers (99th percentile)
    cost_threshold = df['charges'].quantile(0.99)
    df = df[df['charges'] <= cost_threshold]
    
    # Calculate data-driven thresholds (or load from feature analysis)
    thresholds = {
        'age': {
            'threshold': df[df['charges'] >= df['charges'].quantile(0.9)]['age'].median(),
            'charge_multiplier': 1.5
        },
        'bmi': {
            'threshold': 30,  # Standard obesity threshold
            'charge_multiplier': 1.2
        },
        'smoker': {
            'threshold': 'yes',
            'charge_multiplier': 2.5
        }
    }

    # Todo: try using previous threshold and high contribution features from feature_importance_analysis.ipynb
    
    # Save thresholds for documentation
    with open(config_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    # Todo: change the features accordingly

    # Dynamic high-risk criteria
    # age_thresh = thresholds['age']['threshold']
    # bmi_thresh = thresholds['bmi']['threshold']
    
    # high_risk_mask = (
    #     (df['age'] > age_thresh) &
    #     (df['bmi'] > bmi_thresh) &
    #     (df['smoker'] == 'yes')
    # )
    
    # Calculate risk score
     # Get top 10% high-cost users among high-risk group
    high_risk_users = df[high_risk_mask].copy()
    # change the features and weights accordingly
    # high_risk_users['risk_score'] = (
    #     (high_risk_users['age'] / 70) * 0.3 +  # Age contributes 30% to risk
    #     (high_risk_users['bmi'] / 40) * 0.2   # BMI contributes 20%
    # )
    
    # Get top 10% by risk score
    top_10_percent = high_risk_users.nlargest(int(len(high_risk_users) * 0.1), 'risk_score')
    top_10_percent.to_csv(output_path, index=False)
    
    print(f"Identified {len(top_10_percent)} high-risk users")
    
    return top_10_percent

if __name__ == "__main__":
    identify_high_risk_users()