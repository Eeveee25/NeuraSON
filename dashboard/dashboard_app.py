import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from son_module import recommend_actions
from son_rl_agent import SimpleRLSimulator
import io
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load trained model
model = joblib.load("models/fault_predictor.pkl")

# Load RL agent
rl_agent = SimpleRLSimulator()

# Set page config
st.set_page_config(
    page_title="NeuraSON Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
        padding-top: 2rem;
    }
    .son-tag {
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .fault-true {
        background-color: #ff4b4b;
        color: white;
    }
    .fault-false {
        background-color: #28a745;
        color: white;
    }
    .recommendation {
        background-color: #17a2b8;
        color: white;
    }
    .son-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #17a2b8;
    }
    .son-card h4 {
        color: #17a2b8;
        margin-bottom: 0.5rem;
    }
    .son-card p {
        margin: 0;
        color: #666;
    }
    .son-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/your-repo/neurason-logo.png", width=200)
    st.header("⚙️ Settings")
    theme = st.radio("Theme:", options=["Light", "Dark"])
    st.divider()
    st.markdown("### 📱 About NeuraSON")
    st.markdown("""
        AI-powered network optimization platform that:
        - Detects network faults
        - Provides SON recommendations
        - Optimizes network performance
    """)

# File uploader
uploaded_file = st.file_uploader("📤 Upload KPI CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Predict faults and get recommendations
    features = ['RSRP', 'RSRQ', 'SINR', 'throughput_Mbps']
    df['predicted_fault'] = model.predict(df[features])
    df['SON_recommendation'] = df.apply(recommend_actions, axis=1)
    df['RL_Recommendation'] = df.apply(lambda row: rl_agent.get_action_recommendation(row), axis=1)

    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "🗺️ Network Map", "📈 Trends", "⚡ SON Actions"])

    with tab1:
        # Summary metrics in a modern card layout
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cells", df['cell_id'].nunique())
        with col2:
            fault_count = df['predicted_fault'].sum()
            st.metric("Detected Faults", fault_count)
        with col3:
            fault_rate = (fault_count / len(df)) * 100
            st.metric("Fault Rate", f"{fault_rate:.1f}%")
        with col4:
            st.metric("Avg SINR", f"{df['SINR'].mean():.1f} dB")

        # Filters
        st.markdown("### 🔎 Filters")
        col1, col2 = st.columns(2)
        with col1:
            show_faults_only = st.checkbox("Show only detected faults")
        with col2:
            selected_cells = st.multiselect("Filter by Cell ID:", options=sorted(df['cell_id'].unique()))

        # Apply filters
        filtered_df = df.copy()
        if show_faults_only:
            filtered_df = filtered_df[filtered_df['predicted_fault'] == 1]
        if selected_cells:
            filtered_df = filtered_df[filtered_df['cell_id'].isin(selected_cells)]

        # Display filtered data with colored tags
        st.markdown("### 📋 Network Status")
        def format_row(row):
            fault_tag = f'<span class="son-tag fault-{str(row["predicted_fault"]).lower()}">{("❌ Fault" if row["predicted_fault"] else "✅ Normal")}</span>'
            son_tag = f'<span class="son-tag recommendation">{row["SON_recommendation"]}</span>'
            return pd.Series([fault_tag, son_tag])

        display_df = filtered_df.copy()
        display_df[['Fault Status', 'SON Action']] = filtered_df.apply(format_row, axis=1)
        st.write(display_df[['timestamp', 'cell_id', 'Fault Status', 'SON Action']], unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📊 KPI Distribution")
        # Interactive KPI distribution plots using Plotly
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered_df, x='RSRP', color='predicted_fault',
                             title='RSRP Distribution by Fault Status')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(filtered_df, x='SINR', color='predicted_fault',
                             title='SINR Distribution by Fault Status')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 🗺️ Network Map")
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Create map centered on mean coordinates
            m = folium.Map(
                location=[df['latitude'].mean(), df['longitude'].mean()],
                zoom_start=12
            )
            
            # Add markers for each cell
            for _, row in filtered_df.iterrows():
                color = 'red' if row['predicted_fault'] else 'green'
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    color=color,
                    fill=True,
                    popup=f"Cell ID: {row['cell_id']}<br>RSRP: {row['RSRP']}<br>SINR: {row['SINR']}"
                ).add_to(m)
            
            folium_static(m)
        else:
            st.info("Map view requires latitude/longitude data in the CSV file.")

    with tab4:
        st.markdown("### 📈 Network Trends")
        # Site-wise trend analysis
        st.markdown("#### Site-wise KPI Trends")
        selected_site = st.selectbox("Select Site:", options=sorted(filtered_df['cell_id'].unique()))
        
        site_data = filtered_df[filtered_df['cell_id'] == selected_site]
        
        # Create subplots for different KPIs
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('RSRP Trend', 'SINR Trend', 
                                         'RSRQ Trend', 'Throughput Trend'))
        
        # Add traces for each KPI
        fig.add_trace(
            go.Scatter(x=site_data['timestamp'], y=site_data['RSRP'], name='RSRP'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=site_data['timestamp'], y=site_data['SINR'], name='SINR'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=site_data['timestamp'], y=site_data['RSRQ'], name='RSRQ'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=site_data['timestamp'], y=site_data['throughput_Mbps'], name='Throughput'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### ⚡ SON Actions Dashboard")
        
        # SON Action Summary
        son_actions = filtered_df['SON_recommendation'].value_counts()
        
        # Create a pie chart for SON action distribution
        fig = px.pie(values=son_actions.values, 
                    names=son_actions.index,
                    title='Distribution of SON Recommendations')
        st.plotly_chart(fig, use_container_width=True)
        
        # Visual SON Action Cards
        st.markdown("#### 📋 Detailed SON Recommendations")
        
        # Group by cell and get latest recommendations
        latest_recommendations = filtered_df.sort_values('timestamp').groupby('cell_id').last()
        
        for cell_id, row in latest_recommendations.iterrows():
            with st.container():
                st.markdown(f"""
                    <div class="son-card">
                        <h4>Cell ID: {cell_id}</h4>
                        <p><strong>Current Status:</strong> {"❌ Fault Detected" if row['predicted_fault'] else "✅ Normal"}</p>
                        <p><strong>SON Recommendation:</strong> {row['SON_recommendation']}</p>
                        <p><strong>RL Recommendation:</strong> {row['RL_Recommendation']}</p>
                        <p><strong>Current KPIs:</strong> RSRP: {row['RSRP']:.1f} dBm, SINR: {row['SINR']:.1f} dB</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # KPI Impact Analysis
        st.markdown("#### 📊 KPI Impact Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Average KPI values by SON action
            kpi_by_action = filtered_df.groupby('SON_recommendation')[['RSRP', 'SINR']].mean()
            fig = px.bar(kpi_by_action, title='Average KPIs by SON Action')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fault rate by SON action
            fault_by_action = filtered_df.groupby('SON_recommendation')['predicted_fault'].mean() * 100
            fig = px.bar(fault_by_action, title='Fault Rate by SON Action (%)')
            st.plotly_chart(fig, use_container_width=True)

    # Download section
    st.markdown("### 💾 Export Results")
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Filtered Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="neurason_processed_output.csv",
        mime="text/csv"
    )

else:
    # Welcome screen
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1>Welcome to NeuraSON Dashboard</h1>
            <p style="font-size: 1.2rem; color: #666;">
                Upload your network KPI data to get started with AI-powered network optimization
            </p>
        </div>
    """, unsafe_allow_html=True)

