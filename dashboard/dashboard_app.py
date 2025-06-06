import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from utils.son_module import recommend_actions
from utils.son_rl_agent import SimpleRLSimulator

# Load trained model
model = joblib.load("models/fault_predictor.pkl")

# Load RL agent
rl_agent = SimpleRLSimulator()

# Page title
st.title("NeuraSON – AI-Powered Network Health Optimizer")

# File uploader
uploaded_file = st.file_uploader("Upload KPI CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw KPI Data")
    st.dataframe(df)

    # Fault prediction
    features = ['RSRP', 'RSRQ', 'SINR', 'throughput_Mbps']
    df['predicted_fault'] = model.predict(df[features])

    # Rule-based SON recommendation
    df['SON_recommendation'] = df.apply(recommend_actions, axis=1)

    # RL-based SON recommendation
    df['RL_Recommendation'] = df.apply(lambda row: rl_agent.get_action_recommendation(row), axis=1)

    # Output table
    st.subheader("Predictions & Recommendations")
    st.dataframe(df[['timestamp', 'cell_id', 'predicted_fault', 'SON_recommendation', 'RL_Recommendation']])

    # KPI Plot
    st.subheader("KPI Trend Plot (RSRP & SINR)")
    df[['RSRP', 'SINR']].plot(figsize=(10, 5))
    plt.xlabel("Index")
    plt.ylabel("Signal Value")
    st.pyplot(plt)
else:
    st.info("Please upload a KPI CSV file to begin.")

