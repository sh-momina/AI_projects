import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the pre-trained model and dataframe
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.title("Laptop Price Prediction")

# User inputs for various features
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024, 2048])
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
os = st.selectbox('OS', df['OS'].unique())

# Prediction Button
if st.button("Predict Price"):
    # Convert inputs to numeric values as needed
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    # Handle screen resolution: Split into x_res and y_res
    x_res = int(resolution.split("x")[0])
    y_res = int(resolution.split("x")[1])

    # Calculate PPI (Pixels Per Inch)
    ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size
    
    # Create query for the model
    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'IPS': ips,
        'ppi': ppi,
        'Cpu Brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu Brand': gpu,
        'OS': os
    }])

    # Make prediction using the pipeline
    predicted_price = int(pipe.predict(query)[0])

    # Display the predicted price
    st.title(f"The predicted price of this configuration is ${predicted_price}")
