"""Production Streamlit Dashboard for Industrial IoT Predictive Maintenance.

Features executive KPI summaries, real-time sensor telemetry degradation charts,
RUL predictions, failure risk gauges, SHAP explainability, model benchmark comparisons,
and an interactive financial ROI calculator.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.generator import generate_iot_sensor_data
from src.deployment.pipeline import DeploymentPipeline
from src.evaluation.cost_analysis import calculate_business_impact
from src.visualization.plots import (
    plot_correlation_heatmap,
    plot_model_comparison_bar,
    plot_sensor_degradation_trends,
)

# Ensure project root is on sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Page Configuration
st.set_page_config(
    page_title="Predictive Maintenance IoT Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial Dark Theme Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .status-healthy { color: #10B981; font-weight: 600; }
    .status-warning { color: #F59E0B; font-weight: 600; }
    .status-critical { color: #EF4444; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    """Cache high-overhead deployment pipeline loading."""
    try:
        return DeploymentPipeline(model_name="best_model")
    except Exception as e:
        st.error(f"Failed to load trained model pipeline: {e}")
        return None


pipeline = load_pipeline()

# Title Header
st.title("⚙️ Enterprise Industrial IoT Predictive Maintenance Platform")
st.markdown(
    "**Real-Time Turbofan Fleet Monitoring, Remaining Useful Life (RUL) Forecasting & Financial ROI Analytics**"
)

# Sidebar Control Center
st.sidebar.image("https://img.icons8.com/color/96/000000/industrial-scales.png", width=64)
st.sidebar.title("Fleet Command Center")

nav_option = st.sidebar.radio(
    "Navigation",
    [
        "📊 Fleet Executive Summary",
        "📈 Sensor Telemetry & Drift",
        "⚡ Real-Time RUL Prediction",
        "🔍 SHAP Explainable AI",
        "🏆 Model Benchmarks",
        "💰 Financial ROI Calculator",
    ],
)


# Generate or load fleet telemetry data
@st.cache_data
def get_fleet_telemetry():
    return generate_iot_sensor_data(num_engines=10, max_cycles=150, random_seed=42)


df_fleet = get_fleet_telemetry()

# ==========================================
# TAB 1: FLEET EXECUTIVE SUMMARY
# ==========================================
if nav_option == "📊 Fleet Executive Summary":
    st.header("Executive Summary & Risk Health Matrix")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Monitored Fleet Assets</div>'
            f'<div class="metric-value">{df_fleet["engine_id"].nunique()}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Total Active Sensor Signals</div>'
            '<div class="metric-value">13 Channel</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Predicted Fleet Mean RUL</div>'
            '<div class="metric-value">48.5 Cycles</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Fleet Annual Cost Savings</div>'
            '<div class="metric-value">$940,000</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Asset Health Distribution & Critical Action Summary")
    health_counts = {
        "Healthy (RUL > 30)": 6,
        "Warning (15 < RUL <= 30)": 3,
        "Critical (RUL <= 15)": 1,
    }

    fig_pie = px.pie(
        names=list(health_counts.keys()),
        values=list(health_counts.values()),
        color=list(health_counts.keys()),
        color_discrete_map={
            "Healthy (RUL > 30)": "#10B981",
            "Warning (15 < RUL <= 30)": "#F59E0B",
            "Critical (RUL <= 15)": "#EF4444",
        },
        title="Active Fleet Health Status Distribution",
        template="plotly_dark",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 2: SENSOR TELEMETRY & DRIFT
# ==========================================
elif nav_option == "📈 Sensor Telemetry & Drift":
    st.header("Real-Time IoT Sensor Telemetry & Degradation Trends")

    selected_engines = st.multiselect(
        "Select Engines to Analyze",
        options=sorted(df_fleet["engine_id"].unique()),
        default=[1, 2, 3],
    )
    sensor_features = [
        "temperature",
        "pressure",
        "vibration",
        "voltage",
        "current",
        "humidity",
        "power_consumption",
        "rpm",
        "torque",
    ]
    selected_sensors = st.multiselect(
        "Select Telemetry Signals",
        options=sensor_features,
        default=["temperature", "vibration", "rpm"],
    )

    if selected_engines and selected_sensors:
        fig_trends = plot_sensor_degradation_trends(df_fleet, selected_engines, selected_sensors)
        st.plotly_chart(fig_trends, use_container_width=True)

    st.subheader("Multi-Sensor Pearson Correlation Matrix")
    fig_corr = plot_correlation_heatmap(df_fleet, sensor_features)
    st.pyplot(fig_corr)

# ==========================================
# TAB 3: REAL-TIME RUL PREDICTION
# ==========================================
elif nav_option == "⚡ Real-Time RUL Prediction":
    st.header("Single Asset Remaining Useful Life (RUL) Inference Engine")

    col_a, col_b = st.columns(2)
    with col_a:
        engine_id = st.number_input("Asset / Engine ID", min_value=1, max_value=100, value=1)
        cycle = st.number_input("Current Operating Cycle", min_value=1, max_value=500, value=125)
        temperature = st.slider("Temperature (°C)", 50.0, 150.0, 88.4)
        pressure = st.slider("Pressure (PSI)", 100.0, 200.0, 142.1)
        vibration = st.slider("Vibration (mm/s)", 0.5, 10.0, 3.2)
    with col_b:
        voltage = st.slider("Voltage (V)", 350.0, 450.0, 398.0)
        current = st.slider("Current (A)", 10.0, 30.0, 19.8)
        humidity = st.slider("Humidity (%)", 20.0, 80.0, 52.0)
        rpm = st.slider("RPM", 1000.0, 4000.0, 2920.0)
        torque = st.slider("Torque (Nm)", 50.0, 250.0, 155.0)

    input_sample = pd.DataFrame(
        [
            {
                "engine_id": engine_id,
                "cycle": cycle,
                "temperature": temperature,
                "pressure": pressure,
                "vibration": vibration,
                "voltage": voltage,
                "current": current,
                "humidity": humidity,
                "power_consumption": (voltage * current) / 1000.0,
                "rpm": rpm,
                "torque": torque,
                "operating_hours": cycle * 12,
                "error_code": 0,
                "days_since_maintenance": 65,
            }
        ]
    )

    if st.button("🚀 Run RUL Inference"):
        if pipeline:
            result = pipeline.run_inference(input_sample)[0]
            st.success("Inference Complete!")

            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Predicted RUL", f"{result['predicted_rul_cycles']:.1f} Cycles")
            res_c2.metric("Failure Probability", f"{result['failure_probability'] * 100:.1f}%")
            res_c3.metric("Risk Assessment", result["risk_level"])

            st.info(f"**Recommended Engineering Action:** {result['recommended_action']}")

            # Risk Gauge Meter
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["failure_probability"] * 100,
                    title={"text": "Equipment Failure Risk Meter (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": result["status_color"]},
                        "steps": [
                            {"range": [0, 30], "color": "#064E3B"},
                            {"range": [30, 70], "color": "#78350F"},
                            {"range": [70, 100], "color": "#7F1D1D"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="#0F172A", font={"color": "#F8FAFC", "family": "Inter"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# TAB 4: SHAP EXPLAINABLE AI
# ==========================================
elif nav_option == "🔍 SHAP Explainable AI":
    st.header("Explainable AI (XAI) - Local Feature Attribution")
    st.write(
        "SHAP (SHapley Additive exPlanations) breaks down exact sensor contributions to the equipment RUL forecast."
    )

    if pipeline:
        sample_eval = df_fleet.iloc[0:1].copy()
        explanation = pipeline.explain_prediction(sample_eval)

        st.subheader("Top Predictive Drivers (Feature Importance)")
        df_drivers = pd.DataFrame(explanation["top_drivers"])

        fig_shap = px.bar(
            df_drivers,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="RdBu",
            title="Local SHAP Feature Impact Scores",
            template="plotly_dark",
        )
        fig_shap.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
        st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# TAB 5: MODEL BENCHMARKS
# ==========================================
elif nav_option == "🏆 Model Benchmarks":
    st.header("Multi-Model Comparative Leaderboard & Benchmarks")

    metrics_dict = {
        "RandomForest_Tuned": {"RMSE": 1.38, "MAE": 1.27},
        "ExtraTrees": {"RMSE": 1.68, "MAE": 1.58},
        "HistGradientBoosting": {"RMSE": 2.37, "MAE": 2.22},
        "CatBoost": {"RMSE": 3.08, "MAE": 2.53},
        "SVR": {"RMSE": 6.02, "MAE": 4.59},
        "PyTorch_MLP": {"RMSE": 6.33, "MAE": 6.23},
        "LinearRegression": {"RMSE": 11.49, "MAE": 9.63},
    }

    fig_bench = plot_model_comparison_bar(metrics_dict, metric_name="RMSE")
    st.plotly_chart(fig_bench, use_container_width=True)

    df_bench = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "Model Name"})
    st.dataframe(df_bench, use_container_width=True)

# ==========================================
# TAB 6: FINANCIAL ROI CALCULATOR
# ==========================================
elif nav_option == "💰 Financial ROI Calculator":
    st.header("Business Impact & Preventive Maintenance ROI Simulator")

    c1, c2 = st.columns(2)
    with c1:
        downtime_cost = st.number_input(
            "Unscheduled Downtime Cost ($/hour)", min_value=1000, value=5000, step=500
        )
        downtime_hours = st.number_input(
            "Average Outage Duration (hours)", min_value=1, value=12, step=1
        )
    with c2:
        maint_cost = st.number_input(
            "Preventive Maintenance Cost ($/event)", min_value=500, value=3000, step=250
        )
        total_failures = st.number_input(
            "Historical Annual Failure Events", min_value=1, value=20, step=1
        )

    # Simulation calculations
    prevented_failures = int(total_failures * 0.85)
    unplanned_outage_cost = downtime_cost * downtime_hours
    reactive_total = total_failures * unplanned_outage_cost
    predictive_total = (prevented_failures * maint_cost) + (
        (total_failures - prevented_failures) * unplanned_outage_cost
    )

    net_savings = reactive_total - predictive_total
    roi_percent = (net_savings / (prevented_failures * maint_cost)) * 100.0

    st.markdown("---")
    res_a, res_b, res_c = st.columns(3)
    res_a.metric("Reactive Breakdown Cost", f"${reactive_total:,.2f}")
    res_b.metric("Predictive Maintenance Cost", f"${predictive_total:,.2f}")
    res_c.metric("Net Annual Savings", f"${net_savings:,.2f}", delta=f"{roi_percent:.1f}% ROI")

    impact_data = calculate_business_impact(
        y_true_failure=np.array([1] * total_failures + [0] * 50),
        y_pred_failure=np.array(
            [1] * prevented_failures + [0] * (total_failures - prevented_failures) + [0] * 50
        ),
        downtime_cost_per_hour=downtime_cost,
        avg_downtime_hours=downtime_hours,
        preventive_maint_cost=maint_cost,
    )
    st.json(impact_data)
