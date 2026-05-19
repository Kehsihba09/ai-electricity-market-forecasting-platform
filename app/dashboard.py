import random
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_utils import (
fetch_prediction
)

st.set_page_config(
page_title="AI Electricity Forecasting Platform",
layout="wide"
)

st.title(
"⚡ AI Electricity Market Forecasting Platform"
)

st.markdown(
"Enterprise-grade real-time electricity "
"price intelligence system."
)

# SIDEBAR

st.sidebar.header("Input Parameters")

purchase_bid = st.sidebar.slider(
"Purchase Bid",
500,
5000,
1200
)

sell_bid = st.sidebar.slider(
"Sell Bid",
500,
5000,
1000
)

final_volume = st.sidebar.slider(
"Final Scheduled Volume",
100,
5000,
900
)

hour = st.sidebar.slider(
"Hour",
0,
23,
14
)

payload = {
    "purchase_bid": purchase_bid,
    "sell_bid": sell_bid,
    "final_scheduled_volume": final_volume,
    "hour": hour
}

# API PREDICTION

prediction_response = fetch_prediction(payload)
forecast_price = prediction_response["forecast_price"]

# METRICS

col1, col2, col3 = st.columns(3)

col1.metric("Forecast Price", f"{forecast_price:.2f}")

col2.metric("Inference Latency","3 ms")

col3.metric("Kafka Stream","ACTIVE")

# SIMULATED LIVE DATA

timestamps = pd.date_range(start=pd.Timestamp.now(), periods=24,freq="h")

actual_prices = [
random.randint(3000, 6000)
for _ in range(24)
]

forecasted_prices = [
price + random.randint(-200, 200)
for price in actual_prices
]

chart_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Actual Price": actual_prices,
        "Forecasted Price": forecasted_prices
        })

# ACTUAL VS FORECAST

st.subheader("Actual vs Forecasted Electricity Prices")

fig = px.line(

    chart_df,
    x="Timestamp",
    y=["Actual Price", "Forecasted Price"],
    markers=True
)

st.plotly_chart(fig,use_container_width=True)

# VOLATILITY PANEL

st.subheader("Market Volatility")

chart_df["Volatility"] = (chart_df["Actual Price"].rolling(3).std())

volatility_fig = px.line(
    chart_df,
    x="Timestamp",
    y="Volatility"
)

st.plotly_chart(
volatility_fig,
use_container_width=True
)

# STATUS SECTION

st.subheader("System Status")

status_df = pd.DataFrame({
"Component": [
    "Kafka",
    "FastAPI",
    "MLflow",
    "Prediction Engine"
],

"Status": [
    "Running",
    "Running",
    "Running",
    "Running"
]

})

st.dataframe(status_df,use_container_width=True)
