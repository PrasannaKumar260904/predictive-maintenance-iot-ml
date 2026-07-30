"""Publication-grade visualization module producing Matplotlib, Seaborn, and Plotly charts."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Industrial Aesthetic Theme Styling
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
COLOR_PALETTE = ["#00F2FE", "#4FACFE", "#00C6FF", "#0072FF", "#FF416C", "#FF4B2B"]


def plot_sensor_degradation_trends(
    df: pd.DataFrame,
    engine_ids: List[int],
    sensor_cols: List[str],
    save_path: Optional[str] = None,
) -> go.Figure:
    """Creates interactive Plotly multi-sensor degradation timeline across operational cycles."""
    subset = df[df["engine_id"].isin(engine_ids)]

    fig = go.Figure()
    for sensor in sensor_cols:
        for engine in engine_ids:
            engine_data = subset[subset["engine_id"] == engine]
            fig.add_trace(
                go.Scatter(
                    x=engine_data["cycle"],
                    y=engine_data[sensor],
                    mode="lines",
                    name=f"Engine {engine} - {sensor}",
                    opacity=0.8,
                )
            )

    fig.update_layout(
        title="Industrial Sensor Degradation Dynamics Over Operating Cycles",
        xaxis_title="Operating Cycle",
        yaxis_title="Sensor Measurement Value",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font={"family": "Inter, sans-serif", "color": "#F8FAFC"},
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, feature_cols: List[str], save_path: Optional[str] = None
) -> plt.Figure:
    """Plots Seaborn correlation heatmap for selected IoT sensor features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[feature_cols].corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=len(feature_cols) <= 15,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    ax.set_title("IoT Sensor Feature Correlation Matrix", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_rul_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "LightGBM", save_path: Optional[str] = None
) -> go.Figure:
    """Plotly interactive line chart of Actual RUL vs Predicted RUL."""
    indices = np.arange(len(y_true))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=indices, y=y_true, mode="lines", name="Ground Truth RUL", line={"color": "#00C6FF", "width": 2})
    )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=y_pred,
            mode="lines",
            name=f"{model_name} Prediction",
            line={"color": "#FF416C", "width": 2, "dash": "dash"},
        )
    )

    fig.update_layout(
        title=f"Remaining Useful Life (RUL) Prediction Evaluation - {model_name}",
        xaxis_title="Sample Index / Engine Instance",
        yaxis_title="Remaining Useful Life (Cycles)",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font={"family": "Inter, sans-serif", "color": "#F8FAFC"},
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)

    return fig


def plot_model_comparison_bar(
    metrics_dict: Dict[str, Dict[str, float]], metric_name: str = "RMSE", save_path: Optional[str] = None
) -> go.Figure:
    """Interactive bar chart comparing performance metrics across multiple models."""
    models = list(metrics_dict.keys())
    values = [metrics_dict[m].get(metric_name, 0.0) for m in models]

    fig = px.bar(
        x=models,
        y=values,
        labels={"x": "Machine Learning Model", "y": metric_name},
        title=f"Model Comparison Benchmark ({metric_name})",
        color=values,
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )

    fig.update_layout(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font={"family": "Inter, sans-serif", "color": "#F8FAFC"},
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)

    return fig
