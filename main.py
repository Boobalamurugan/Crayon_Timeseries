import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pycaret.time_series import load_model, predict_model
import datetime as dt

def calculate_wape(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.mean(np.abs(y_pred)) * 100
    return (numerator / denominator) * 100

st.set_page_config(layout="wide")
st.title("📊 Time Series Forecast Dashboard")

# --- Top Right Controls ---
col1, col2 = st.columns([2, 1])

with col2:
    model_option = st.selectbox("Select Model", ["18-Month Model", "21-Month Model"])
    granularity = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)

# --- Load actual data and set model configurations ---
actual_df = pd.read_csv("21_months.csv", parse_dates=["Document date"])

if model_option == "18-Month Model":
    # Configuration for 18-month model
    model_path = "18_model_pkl"
    prediction_days = 92  # Oct-Dec 2024 (3 months)
    
    # Get only Oct-Dec 2024 data for comparison
    actual_df_filtered = actual_df[
        (actual_df["Document date"] >= "2024-10-01") & 
        (actual_df["Document date"] <= "2024-12-31")
    ].copy()
    actual_df_filtered = actual_df_filtered[["Document date", "TotalCollection"]]
    
    # Load model and make predictions for Oct-Dec 2024
    model = load_model(model_path)
    pred = predict_model(model, fh=prediction_days)
    pred_df = pd.DataFrame(np.expm1(pred)).reset_index()
    pred_df.columns = ['index', 'predictions']
    pred_df["Document date"] = pd.date_range(start="2024-10-01", end="2024-12-31", freq="D")
    
    # Create the combined dataframe
    full_df = pd.DataFrame()
    full_df["Document date"] = pd.date_range(start="2024-10-01", end="2024-12-31", freq="D")
    
    # Add actual and forecast columns
    full_df = pd.merge(full_df, actual_df_filtered[["Document date", "TotalCollection"]], 
                      on="Document date", how="left")
    full_df = pd.merge(full_df, pred_df[["Document date", "predictions"]], 
                      on="Document date", how="left")
    
    # Rename columns
    full_df = full_df.rename(columns={"TotalCollection": "Actual", "predictions": "Forecast"})

else:
    # Configuration for 21-month model
    model_path = "21_model_pkl"
    prediction_days = 31  # Jan-Mar 2025 (3 months)
    
    # Get the last actual date from data (should be end of Dec 2024)
    last_actual_date = actual_df["Document date"].max()
    
    # Load model and make predictions for Jan-Mar 2025
    model = load_model(model_path)
    pred = predict_model(model, fh=prediction_days)
    pred_df = pd.DataFrame(np.expm1(pred)).reset_index()
    pred_df.columns = ['index', 'predictions']
    pred_df["Document date"] = pd.date_range(start="2025-01-01", end="2025-01-31", freq="D")
    
    # Create the forecast dataframe (no actuals since we're predicting future)
    full_df = pred_df[["Document date", "predictions"]].copy()
    full_df = full_df.rename(columns={"predictions": "Forecast"})
    # Add Actual column as None since this is future prediction
    full_df["Actual"] = None

# --- Granularity logic ---
def resample_df(df, gran):
    df = df.set_index("Document date")
    if gran == "Weekly":
        df = df.resample("W").sum()
    elif gran == "Monthly":
        df = df.resample("M").sum()
    else:
        df = df.resample("D").sum()
    return df.reset_index()

plot_df = resample_df(full_df, granularity)

# --- Top KPIs ---
with col1:
    if model_option == "18-Month Model":
        wape = calculate_wape(full_df["Actual"], full_df["Forecast"])
        st.metric("WAPE (Oct-Dec 2024)", f"{wape:.1f}%")
        correlation = full_df["Actual"].corr(full_df["Forecast"])
        st.metric("Correlation (Oct-Dec 2024)", f"{correlation:.3f}")
        max_actual = full_df["Actual"].max()
        st.metric("Maximum Value (Oct-Dec 2024)", f"${max_actual/1e6:.1f}M")
    else:
        last_actual = actual_df[actual_df["Document date"] == last_actual_date]["TotalCollection"].iloc[0]
        max_pred = pred_df["predictions"].max()
        st.metric("Last Known Value (Dec 2024)", f"${last_actual/1e6:.1f}M")
        st.metric("Maximum Forecast (Jan-Mar 2025)", f"${max_pred/1e6:.1f}M")

# --- Plotly Chart ---
fig = go.Figure()

if model_option == "18-Month Model":
    # Actual values (Oct-Dec 2024)
    fig.add_trace(go.Bar(
        x=plot_df["Document date"],
        y=plot_df["Actual"],
        name="Actual Values (Oct-Dec 2024)",
        marker_color='green'
    ))

    # Predictions (Oct-Dec 2024)
    fig.add_trace(go.Bar(
        x=plot_df["Document date"],
        y=plot_df["Forecast"],
        name="Model Predictions",
        marker_color='lightgreen',
        opacity=0.7
    ))
else:
    # For 21-month model, we only show forecast since we're predicting future

    # Forecast values (only Oct-Dec 2024 for 18-month model)
    fig.add_trace(go.Bar(
        x=plot_df["Document date"],
        y=plot_df["Forecast"],
        name="Month Forecast",
        marker_color='lightgreen',
        opacity=0.7
    ))

title_text = f"{model_option} - {granularity} View"
if model_option == "18-Month Model":
    title_text += " (Oct-Dec 2024 Actual vs Predictions)"
else:
    title_text += " (Month Forecast)"

fig.update_layout(
    barmode='overlay',
    title=title_text,
    xaxis_title="Date",
    yaxis_title="Collection Amount",
    legend=dict(x=0.01, y=0.99),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Add statistics
st.subheader("Statistics")
if model_option == "18-Month Model":
    col3, col4, col5 = st.columns(3)
    with col3:
        mae = np.mean(np.abs(full_df["Actual"] - full_df["Forecast"]))
        st.metric("Mean Absolute Error", f"${mae/1e6:.1f}M")
    with col4:
        rmse = np.sqrt(np.mean((full_df["Actual"] - full_df["Forecast"])**2))
        st.metric("RMSE", f"${rmse/1e6:.1f}M")
    with col5:
        avg_actual = full_df["Actual"].mean()
        st.metric("Avg Value (Oct-Dec)", f"${avg_actual/1e6:.1f}M")
else:
    forecast_stats = pred_df["predictions"].describe()
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Average Forecast", f"${forecast_stats['mean']/1e6:.1f}M")
    with col4:
        st.metric("Minimum Forecast", f"${forecast_stats['min']/1e6:.1f}M")
    with col5:
        st.metric("Maximum Forecast", f"${forecast_stats['max']/1e6:.1f}M")
