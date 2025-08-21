import pandas as pd
import numpy as np


df = pd.read_csv('../data/insurance_cleaned.csv')

df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]


num_cols = ['age', 'bmi', 'children', 'charges']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')


df['smoker'] = (df['smoker'].astype(str).str.strip().str.lower()
                .map({'yes':'yes','y':'yes','1':'yes','true':'yes',
                      'no':'no','n':'no','0':'no','false':'no'}))


df = df.drop_duplicates()
df.loc[df['age'] < 0, 'age'] = np.nan
df.loc[df['bmi'] <= 0, 'bmi'] = np.nan
df.loc[df['children'] < 0, 'children'] = np.nan
df.loc[df['charges'] < 0, 'charges'] = np.nan
df = df.dropna(subset=['age','bmi','smoker','children','charges'])


df['high_risk'] = (df['charges'] > 10000).astype(int)  # 或用 df['charges'].quantile(0.75)
print(df.head())

# 1. Data Preparation
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = df[['age','bmi','smoker','children']]
y = df['high_risk']

preprocess = ColumnTransformer([
    ('num','passthrough',['age','bmi','children']),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['smoker']),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Model Training
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ('prep', preprocess),
    ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=6,
        class_weight='balanced', random_state=42
    ))
])

model.fit(X_train, y_train)

# 3. Prediction
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

new_applicant = pd.DataFrame([{'age':45,'bmi':28.5,'smoker':'yes','children':2}])
risk_prediction = int(model.predict(new_applicant)[0])
risk_probability = float(model.predict_proba(new_applicant)[0,1])

# 4. Evaluation
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred, target_names=['Low Risk','High Risk']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

ohe = model.named_steps['prep'].named_transformers_['cat']
cat_names = ohe.get_feature_names_out(['smoker'])
feature_names = ['age','bmi','children'] + list(cat_names)

importances = model.named_steps['clf'].feature_importances_
(pd.Series(importances, index=feature_names)
   .sort_values().plot(kind='barh', title='Feature Importance'));

# 5. Model Deployment
import joblib
joblib.dump(model, 'risk_classifier.pkl')

from flask import Flask, request, jsonify
app = Flask(__name__)
production_model = joblib.load('risk_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_in = pd.DataFrame([data])
    proba = float(production_model.predict_proba(X_in)[0,1])
    return jsonify({'risk_level': int(proba >= 0.5), 'probability': proba})

# # 6. Monitoring (Optional)
# from alibi_detect.cd import KSDrift
# X_ref = model.named_steps['prep'].transform(X_train)
# X_cur = model.named_steps['prep'].transform(X_test)
# drift_detector = KSDrift(X_ref, p_val=0.05)
# drift = drift_detector.predict(X_cur)
# print('Data drift detected:', drift['data']['is_drift'])

