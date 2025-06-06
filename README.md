# NeuraSON – AI-Powered Network Health Optimizer for 5G/LTE

**NeuraSON** is a smart analytics platform designed to improve mobile network operations through AI-driven fault prediction, real-time KPI monitoring, and basic Self-Optimizing Network (SON) logic. This project simulates the kind of automation and intelligence that modern LTE and 5G networks require to reduce manual intervention, minimize service disruptions, and improve customer experience.

---

## Project Objectives

- Develop a machine learning model to predict network faults based on telecom KPIs.
- Visualize live KPI trends using an interactive dashboard.
- Simulate basic SON behavior such as recommending parameter adjustments (e.g., power boost, tilt correction).
- Lay the foundation for future integration with real OSS/RAN systems or 5G testbeds.

---

## Key Features

- Real-time visualization of KPIs: RSRP, RSRQ, SINR, throughput
- Fault prediction using trained ML model
- SON recommendation engine with rule-based logic
- Upload and process KPI log files in CSV format
- Modular and extensible design for future enhancements

---

## Technology Stack

- **Language**: Python
- **ML/AI**: Scikit-learn, Joblib
- **Data Handling**: Pandas, Matplotlib
- **Dashboard**: Streamlit
- **Model Deployment**: Local or Cloud-based runtime

---

## KPIs Used

- `RSRP` – Reference Signal Received Power
- `RSRQ` – Reference Signal Received Quality
- `SINR` – Signal-to-Interference-plus-Noise Ratio
- `throughput_Mbps` – Downlink throughput
- `call_drops` – Number of dropped calls (used as label for faults)

---

## Project Structure
