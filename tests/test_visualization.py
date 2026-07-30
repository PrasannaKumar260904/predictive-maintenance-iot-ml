"""Unit tests for visualization module."""

import numpy as np
import pandas as pd

from src.visualization.plots import (
    plot_correlation_heatmap,
    plot_model_comparison_bar,
    plot_rul_predictions,
    plot_sensor_degradation_trends,
)


def test_visualization_plots(tmp_path):
    df_sample = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "temperature": [70.0, 75.0, 68.0, 72.0],
            "pressure": [150.0, 145.0, 152.0, 148.0],
        }
    )

    fig1 = plot_sensor_degradation_trends(
        df_sample, engine_ids=[1, 2], sensor_cols=["temperature"], save_path=str(tmp_path / "trends.html")
    )
    assert fig1 is not None
    assert (tmp_path / "trends.html").exists()

    fig2 = plot_correlation_heatmap(df_sample, ["temperature", "pressure"], save_path=str(tmp_path / "corr.png"))
    assert fig2 is not None

    y_true = np.array([100, 50, 10])
    y_pred = np.array([98, 52, 12])
    fig3 = plot_rul_predictions(y_true, y_pred, save_path=str(tmp_path / "rul.html"))
    assert fig3 is not None

    metrics_dict = {"ModelA": {"RMSE": 10.0}, "ModelB": {"RMSE": 8.5}}
    fig4 = plot_model_comparison_bar(metrics_dict, metric_name="RMSE", save_path=str(tmp_path / "comp.html"))
    assert fig4 is not None
