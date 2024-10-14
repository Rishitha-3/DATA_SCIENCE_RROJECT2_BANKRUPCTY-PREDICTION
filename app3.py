import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#Import the file
file_path = "Bankruptcy Prediction.csv"

try:
    #Load the data file
    data = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.stop()

#Define the input columns
input_cols = [
    ' ROA(C) before interest and depreciation before interest',
    ' ROA(A) before interest and % after tax',
    ' ROA(B) before interest and depreciation after tax',
    ' Operating Gross Margin',
    ' Realized Sales Gross Margin',
    ' Operating Profit Rate',
    ' Pre-tax net Interest Rate',
    ' After-tax net Interest Rate',
    ' Non-industry income and expenditure/revenue',
    ' Continuous interest rate (after tax)',
    ' Operating Expense Rate',
    ' Research and development expense rate',
    ' Cash flow rate',
    ' Interest-bearing debt interest rate',
    ' Tax rate (A)'
]

#Define the target variable
target_col = 'Bankrupt?'

#Split the data into training and testing datasets
X = data[input_cols]
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Appling PCA to the data
pca = PCA(n_components=15)  
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#Training an XGBoost model on the training data
xgb_model = XGBClassifier()
xgb_model.fit(X_train_pca, y_train)

#Saving the trained model to a file
xgb_model.save_model('xgb_model.json')

#Creating a Streamlit app
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon=":chart:", layout="wide")
st.markdown("<style>body, html {height: 100%; margin: 0; background-color: #ADD8E6;}</style>", unsafe_allow_html=True)
st.title("Bankruptcy Prediction App")
st.markdown("<h3 style='text-align: center; color: white;'>Enter the values for the following columns:</h3>", unsafe_allow_html=True)

#Creating input fields
from decimal import Decimal
input_values = []
for col in input_cols:
    value = st.text_input(col, value="0.000000")
    input_values.append(Decimal(value))

#Creating a button to predict bankruptcy
if st.button("Predict"):
    input_df = pd.DataFrame([input_values], columns=input_cols)
    #Scale the input data using StandardScaler
    input_scaled = scaler.transform(input_df)
    #Apply PCA to the scaled data
    input_pca = pca.transform(input_scaled)
    #Load the trained model from the file
    xgb_model = XGBClassifier()
    xgb_model.load_model('xgb_model.json')
    #Make predictions using the XGBoost model
    prediction = xgb_model.predict(input_pca)
    #Display the output
    if prediction[0] == 1:
        st.write("<h1 style='text-align: center; color: red; font-size: 48px;'>**Bankruptcy is likely to happen.**</h1>", unsafe_allow_html=True)
    else:
        st.write("<h1 style='text-align: center; color: green; font-size: 48px;'>**Bankruptcy is not likely to happen.**</h1>", unsafe_allow_html=True)
