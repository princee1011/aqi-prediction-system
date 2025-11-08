import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ✅ Correct Imports
from data_loading import load_data_to_mongodb
from model_training import train_all_models
from aqi_prediction_pipeline import (
    get_real_time_air_quality,
    predict_all_pollutants,
    test_api_connection
)

# ✅ Global Settings
POLLUTANTS = ['pm25', 'pm10', 'o3', 'no2', 'so2']
PROJECT_CITIES = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata']

# ============================================================
# Page Setup
# ============================================================
st.set_page_config(page_title="🌍 AQI Prediction System", page_icon="🌱", layout="wide")
st.title("🌍 Multi-Pollutant Air Quality & AQI Prediction")

st.markdown("### Real-time monitoring, pollution forecasting, and AQI classification using LSTM")

# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.header("⚙️ Configuration")

if st.sidebar.button("🔌 Test API Connection"):
    with st.spinner("Checking WAQI API..."):
        ok = test_api_connection()
        st.sidebar.success("✅ API is Working" if ok else "❌ API Not Working")

city = st.sidebar.selectbox("🏙️ Select City", PROJECT_CITIES)
days = st.sidebar.slider("Prediction Days", 1, 14, 7)

st.sidebar.subheader("🧠 Model & Data Operations")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("📥 Load Historical Data"):
        with st.spinner("Loading data into MongoDB..."):
            st.success("✅ Data Loaded Successfully" if load_data_to_mongodb() else "❌ Failed to Load Data")

with col2:
    if st.button("🤖 Train LSTM Models"):
        with st.spinner("Training models..."):
            train_all_models(PROJECT_CITIES)
            st.success("✅ Model Training Completed")

if st.sidebar.button("🎯 Run Prediction", type="primary"):
    with st.spinner(f"Predicting for {city}..."):
        result = predict_all_pollutants(city, days=days)
        if result:
            st.session_state.predictions = result
            st.sidebar.success("✅ Prediction Completed")
        else:
            st.sidebar.error("❌ Prediction Failed (Check API or Models)")

# ============================================================
# Tabs Layout
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Real-Time Data",
    "🔮 Pollutant Predictions",
    "🎯 AQI Forecast",
    "🔍 Explainable AI",
    "ℹ️ About"
])

# ============================================================
# TAB 1 — Real-Time Data
# ============================================================
with tab1:
    st.header("🌐 Real-Time Air Quality")

    if st.button("🔄 Refresh Live Data"):
        st.session_state.current_data = get_real_time_air_quality(city)

    current_data = st.session_state.get('current_data', get_real_time_air_quality(city))

    if current_data:
        st.subheader(f"📍 {current_data['city']} — Current Status")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall AQI", f"{current_data['aqi']} {current_data.get('emoji','')}")
            st.markdown(
                f"<h4 style='color:{current_data['color']}'>{current_data['category']}</h4>", 
                unsafe_allow_html=True
            )
        with col2:
            st.metric("Dominant Pollutant", current_data['dominant_pollutant'])
        with col3:
            st.metric("Last Updated", current_data['timestamp'])

        st.markdown("---")

# ============================================================
# TAB 2 — Pollutant Predictions
# ============================================================
with tab2:
    st.header("🔮 Multi-Pollutant Forecast")

    if 'predictions' not in st.session_state:
        st.info("👆 Run prediction first.")
    else:
        results = st.session_state.predictions
        pred_dict = results["predictions"]

        st.subheader(f"📈 Next {days} Days — {results['city']}")

        df = pd.DataFrame({
            "Date": [(datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days)]
        })

        for p in POLLUTANTS:
            df[p.upper()] = pred_dict[p]["predictions"]

        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 Trend Graphs")
        for p in POLLUTANTS:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df[p.upper()], mode='lines+markers', line=dict(width=3)))
            fig.update_layout(title=f"{p.upper()} Trend — {results['city']}", xaxis_title="Date", yaxis_title="µg/m³")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — Daily AQI Forecast
# ============================================================
with tab3:
    st.header("🎯 AQI Forecast (Next 7 Days)")

    if 'predictions' not in st.session_state:
        st.info("👆 Run prediction first.")
    else:
        df = pd.DataFrame(st.session_state.predictions["daily_aqi"])
        st.dataframe(df, use_container_width=True)

        fig = px.line(df, x="day", y="aqi", markers=True, title=f"AQI Trend — {city}")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4 — Explainable AI
# ============================================================
with tab4:
    st.header("🔍 Explainable AI - Why Did AQI Change?")

    if 'predictions' not in st.session_state:
        st.info("👆 Run prediction first.")
    else:
        pred_dict = st.session_state.predictions["predictions"]

        st.subheader("📌 Pollutant Influence Overview")

        rows = [{"Pollutant": p.upper(), "Change (%)": abs(pred_dict[p]["explanation"]["change_percent"])}
                for p in POLLUTANTS]

        df_imp = pd.DataFrame(rows)
        fig = px.bar(df_imp, x="Change (%)", y="Pollutant", orientation="h", color="Change (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Per-Pollutant Reasoning")

        for p in POLLUTANTS:
            exp = pred_dict[p]["explanation"]
            st.write(f"### {p.upper()} — {exp['trend'].upper()} trend")
            st.write(f"• Current: {exp['current']} µg/m³")
            st.write(f"• Tomorrow: {exp['predicted']} µg/m³")
            st.write(f"• Change: {exp['change']} ({exp['change_percent']}%)")
            st.write("**Factors:**")
            for f in exp["factors"]:
                st.write(f"- {f}")
            st.markdown("---")

# ============================================================
# TAB 5 — About
# ============================================================
with tab5:
    st.header("ℹ️ About This System")
    st.markdown("""
This system forecasts air pollution levels using:
- **LSTM Deep Learning models**
- **Real-time WAQI API data**
- **CPCB Standard AQI computation**
- **Explainable AI trend analysis**

Covered Cities: **Delhi, Mumbai, Chennai, Kolkata**  
Covered Pollutants: **PM₂.₅, PM₁₀, O₃, NO₂, SO₂**
    """)
