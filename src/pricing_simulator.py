# pricing_simulator_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# First install
# pip install streamlit pandas scikit-learn matplotlib
# To run this
# python -m streamlit  run src/pricing_simulator.py   
# To view the app, go to http://localhost:8501/

# Todo: 
# 1. tweak the ML Models and features settings and compare the results
# 2. Change this to instead of train the model here, load the previously trained model (e.g. pkl file) and only do inferencing on new data

class InsurancePricingSimulator:
    def __init__(self, data_path=None):
             # Get the parent directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Set default path if none provided
        if data_path is None:
            data_path = os.path.join(parent_dir, 'data', 'insurance_cleaned.csv')

        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        self.train_model()

    def preprocess_data(self):
        """Prepare the insurance data"""
        self.df['smoker'] = self.df['smoker'].map({'yes': 1, 'no': 0})
        self.df = pd.get_dummies(self.df, columns=['sex', 'region'], drop_first=True)
        
        self.scaler = StandardScaler()
        self.X = self.df.drop('charges', axis=1)
        self.y = self.df['charges']
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def train_model(self):
        """Train pricing prediction model"""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_scaled, self.y)
        self.baseline_price = np.mean(self.y)
        
    def predict_charge(self, inputs):
        """Predict insurance charge for given inputs"""
        input_df = pd.DataFrame([inputs], columns=self.X.columns)
        scaled_input = self.scaler.transform(input_df)
        return self.model.predict(scaled_input)[0]

# Streamlit UI
def main():
    st.title("Insurance Pricing Simulator")
    st.write("Adjust the parameters to estimate medical insurance costs")

    # Initialize simulator
    simulator = InsurancePricingSimulator()

    # Create sidebar controls
    with st.sidebar:
        st.header("Input Parameters")
        age = st.slider("Age", 18, 70, 30)
        bmi = st.slider("BMI", 15.0, 40.0, 25.0)
        children = st.slider("Children", 0, 5, 0)
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        sex = st.selectbox("Sex", ["Male", "Female"])
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    # Prepare inputs
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

    # Predict and display
    price = simulator.predict_charge(inputs)
    color = "red" if price > simulator.baseline_price * 1.5 else "green"
    
    st.metric("Estimated Premium", f"${price:,.2f}", 
              delta=f"${price - simulator.baseline_price:,.2f} vs baseline")
    
    # Feature impact plot
    st.subheader("Feature Impact Analysis")
    features = ['age', 'bmi', 'children', 'smoker']
    values = [inputs[f] for f in features]
    impacts = []
    
    baseline_input = {k: simulator.X[k].mean() for k in simulator.X.columns}
    for feat in features:
        temp_input = baseline_input.copy()
        temp_input.update(inputs)
        temp_input[feat] = baseline_input[feat]
        impacts.append(price - simulator.predict_charge(temp_input))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(features, impacts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title('How Each Feature Affects Your Premium')
    ax.set_xlabel('Price Impact ($)')
    
    for bar in bars:
        width = bar.get_width()
        label = f"+${abs(width):.0f}" if width >0 else f"-${abs(width):.0f}"
        ax.text(width/2, bar.get_y() + bar.get_height()/2, label, 
               ha='center', va='center', color='white')
    
    st.pyplot(fig)

if __name__ == '__main__':
    main()