# pricing_simulator_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


# To run:
# python -m streamlit run src/pricing_simulator.py

class InsurancePricingSimulator:
    def __init__(self, model_name="Random Forest", data_path=None, model_dir=None):
        self.model_name = model_name.replace(" ", "_").lower()

        # Paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        if data_path is None:
            data_path = os.path.join(parent_dir, 'data', 'insurance_cleaned.csv')
        if model_dir is None:
            model_dir = os.path.join(parent_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)

        self.data_path = data_path
        self.model_path = os.path.join(model_dir, f'{self.model_name}_model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')

        self.df = pd.read_csv(self.data_path)
        self.preprocess_data()

        self.model_obj = self.get_model_instance()

        # Load if exists
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.X_scaled = self.scaler.transform(self.X)
        else:
            self.train_model()
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

        self.y_pred = self.model.predict(self.X_scaled)
        self.mae = mean_absolute_error(self.y, self.y_pred)
        self.r2 = r2_score(self.y, self.y_pred)
        self.baseline_price = np.mean(self.y)

    def preprocess_data(self):
        self.df['smoker'] = self.df['smoker'].map({'yes': 1, 'no': 0})
        self.df = pd.get_dummies(self.df, columns=['sex', 'region'], drop_first=True)

        self.X = self.df.drop('charges', axis=1)
        self.y = self.df['charges']
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def get_model_instance(self):
        model_dict = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "linear_regression": LinearRegression()
        }
        return model_dict[self.model_name]

    def train_model(self):
        self.model = self.model_obj
        self.model.fit(self.X_scaled, self.y)

    def predict_charge(self, inputs):
        input_df = pd.DataFrame([inputs], columns=self.X.columns)
        scaled_input = self.scaler.transform(input_df)
        return self.model.predict(scaled_input)[0]


# Streamlit UI
def main():
    st.title(" Insurance Pricing Simulator + Model Comparison")

    st.sidebar.header("Model & Input Settings")

    model_choice = st.sidebar.selectbox(
        "Choose a prediction model",
        ["Random Forest", "Gradient Boosting", "Linear Regression"]
    )

    simulator = InsurancePricingSimulator(model_name=model_choice)

    # Input controls
    age = st.sidebar.slider("Age", 18, 70, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
    children = st.sidebar.slider("Children", 0, 5, 0)
    smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    inputs = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker': 1 if smoker == "Yes" else 0,
        'sex_male': 1 if sex == "Male" else 0,
        'region_northwest': 1 if region == "Northwest" else 0,
        'region_southeast': 1 if region == "Southeast" else 0,
        'region_southwest': 1 if region == "Southwest" else 0
    }

    price = simulator.predict_charge(inputs)
    delta = price - simulator.baseline_price

    st.metric("Estimated Premium", f"${price:,.2f}", f"{delta:+.0f} vs avg")

    st.markdown("###  Model Performance")
    st.write(f"**Model:** {model_choice}")
    st.write(f"**MAE:** ${simulator.mae:.2f}")
    st.write(f"**R² Score:** {simulator.r2:.3f}")

    st.markdown("---")
    st.subheader(" Feature Impact on Premium")
    features = ['age', 'bmi', 'children', 'smoker']
    impacts = []

    baseline_input = {k: simulator.X[k].mean() for k in simulator.X.columns}
    for feat in features:
        temp_input = baseline_input.copy()
        temp_input.update(inputs)
        temp_input[feat] = baseline_input[feat]
        impact = price - simulator.predict_charge(temp_input)
        impacts.append(impact)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(features, impacts, color='skyblue')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title('Feature Impact')
    ax.set_xlabel('Change in Predicted Premium ($)')
    for bar in bars:
        width = bar.get_width()
        label = f"+${abs(width):.0f}" if width > 0 else f"-${abs(width):.0f}"
        ax.text(width / 2, bar.get_y() + bar.get_height() / 2, label, ha='center', va='center', color='black')

    st.pyplot(fig)


if __name__ == '__main__':
    main()
