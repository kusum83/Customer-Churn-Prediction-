# Customer Churn Prediction with Deep Learning & SHAP Explainability

## Overview
A deep neural network built to predict telecom customer churn with integrated 
SHAP explainability, identifying the key drivers of churn risk to support 
data-driven retention strategies.

## Key Results
- Binary classification on IBM Telco Churn dataset (7,043 records, 20 features)
- Batch Normalisation layers for training stability and generalisation
- SHAP summary and force plots identifying top churn-driving features

## Tech Stack
Python · TensorFlow / Keras · Batch Normalisation · SHAP · 
Scikit-learn · LabelEncoder · StandardScaler

## How to Run
1. Clone the repo: `git clone https://github.com/kusumik/churn-prediction-shap`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `churn_prediction.ipynb` in Jupyter

## Dataset
IBM Telco Customer Churn Dataset — available on 
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
