import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("fault_predictor.pkl")

st.title("NeuraSON - Network Health Optimizer")

uploaded_file = st.file_uploader("Upload KPI CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("KPI Data")
    st.dataframe(df)

    features = ['RSRP', 'RSRQ', 'SINR', 'throughput_Mbps']
    df['predicted_fault'] = model.predict(df[features])

    st.subheader("Predicted Faults")
    st.write(df[df['predicted_fault'] == 1])

    st.subheader("KPI Trend")
    df[['RSRP', 'SINR']].plot(figsize=(10, 5))
    st.pyplot(plt)

