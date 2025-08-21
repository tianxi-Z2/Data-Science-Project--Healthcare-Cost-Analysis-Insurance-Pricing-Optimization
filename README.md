# Healthcare Cost Analysis & Insurance Pricing Optimization  

![Healthcare Analytics](https://img.shields.io/badge/Project-Healthcare%20Analytics-blue)  
![Python](https://img.shields.io/badge/Python-3.8%2B-green)  
![License](https://img.shields.io/badge/License-MIT-lightgrey)  

## 📌 Project Overview  
Analyze medical cost data to develop data-driven insurance pricing strategies. Students will complete an end-to-end workflow from **data cleaning → statistical analysis → business decision-making**, delivering actionable pricing recommendations.  

**Keywords**: ANOVA, Linear Regression, Insurance Pricing, Health Risk Assessment  
**Dataset**: [Kaggle Medical Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  

## 🎯 Learning Objectives  
### Technical Skills  
- ✅ Master ANOVA and Tukey HSD post-hoc tests  
- ✅ Build cost prediction models (Linear Regression/Decision Trees)  
- ✅ Feature engineering (BMI binning/cross-features)  
- ✅ Statistical visualization (Boxplots/Heatmaps)  

### Soft Skills  
- ✅ Explain statistical results to non-technical stakeholders  
- ✅ Create business-ready reports with data-driven insights  

## 📂 Project Structure  
```text
project/
├── data/                    # Raw data
│   └── insurance.csv        
├── notebooks/               # Analysis notebooks
│   ├── 1_EDA.ipynb          # Exploratory Analysis
│   ├── 2_Statistical_Analysis.ipynb  # Hypothesis Testing
│   └── 3_Pricing_Model.ipynb # Predictive Modeling
│   └── 4_feature_importance_analysis.ipynb # analysis feature contribution and determine threshold
├── reports/                 # Business deliverables
│   ├── cost_drivers.pptx    # Summary deck
│   └── high_risk_users.csv  # Top 10% high-cost users
└── src/                     # Reusable code
    ├── data_cleaning.py     # Data preprocessing
    └── pricing_simulator.py # Interactive calculator
```

### Module 1-2: Data Preparation
1. **Data Cleaning**  
   - Handle missing BMI values (median imputation)  
   - Remove cost outliers (>99th percentile)  
   - *Deliverable: Reusable `data_cleaning.py` script*  

2. **Exploratory Analysis (EDA)**  
   - Plot cost distribution (right-skewed?)  
   - Generate feature correlation heatmap  
   - *Deliverable: `notebooks/1_EDA.ipynb`*  

### Module 3-4: Statistical Analysis
1. **Hypothesis Testing**  
   - ANOVA: Region impact on costs (report F-value/p-value)  
   - T-test: Smoker vs non-smoker cost difference  
   - *Code example*:  
     ```python
     from scipy.stats import f_oneway
     f_oneway(df[df['region']=='southeast']['charges'], 
              df[df['region']=='northwest']['charges'])
     ```

2. **Post-hoc Analysis**  
   - Tukey HSD to identify regional pairwise differences  
   - *Deliverable: `notebooks/2_Statistical_Analysis.ipynb`*  

### Module 5-6: Modeling 
1. **Predictive Modeling**  
   - Linear regression (target: `charges`)  
   - Success criteria: R² > 0.75, VIF < 5  

2. **Model Interpretation**  
   - Translate coefficients (e.g., "Smoking adds $10,000/year")  
   - *Deliverable: `notebooks/3_Pricing_Model.ipynb`*  

### Module 7-8: Business Application 
1. **High-Risk User Identification**  
   - Determine rules using notebooks/4_feature_importance_analysis.ipynb (e.g., Age>50 + BMI>30 + Smoker)  
   - Identify high risk users.
   - Export top 10% high-cost users  
   - *Deliverable: `src/high_risk_identification.py`*  

2. **Pricing Simulator**  
   - Build interactive tool with Jupyter Widgets:  
     ```python
     @interact(age=(18,70), bmi=(15,40), smoker=[True,False])
     def quote_price(age, bmi, smoker):
         return base_price * risk_factor
     ```
   - *Deliverable: `src/pricing_simulator.py`*  

## 💡 Resources  
1. [Statsmodels ANOVA Docs](https://www.statsmodels.org/stable/anova.html)  
2. [Kaggle Insurance Pricing Example](https://www.kaggle.com/code/andresionek/insurance-pricing-model)  
3. [Healthcare Analytics Course (Coursera)](https://www.coursera.org/learn/healthcare-data-analytics)  

## ❓ FAQ  
**Q: Is medical knowledge required?**  
A: No. Only basic statistics is needed; data is anonymized.  

**Q: How to validate pricing strategies?**  
A: Backtest with historical data (e.g., actual vs predicted costs for high-risk groups).  

--- 
