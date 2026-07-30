"""Unit tests for evaluation metrics and business ROI cost analysis."""

import numpy as np

from src.evaluation.cost_analysis import calculate_business_impact
from src.evaluation.metrics import evaluate_classification, evaluate_regression


def test_evaluate_regression():
    y_true = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    y_pred = np.array([98.0, 82.0, 58.0, 41.0, 19.0])

    metrics = evaluate_regression(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert "R2" in metrics
    assert metrics["R2"] > 0.95


def test_evaluate_classification():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])

    metrics = evaluate_classification(y_true, y_pred)
    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1_Score" in metrics
    assert metrics["Accuracy"] > 0.8


def test_calculate_business_impact():
    y_true_failure = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred_failure = np.array([1, 1, 0, 0, 1, 0, 0, 0])  # 2 TP, 1 FN, 1 FP, 4 TN

    roi_dict = calculate_business_impact(
        y_true_failure=y_true_failure,
        y_pred_failure=y_pred_failure,
        downtime_cost_per_hour=5000,
        avg_downtime_hours=10,
        preventive_maint_cost=2000,
    )

    assert "net_cost_savings" in roi_dict
    assert "roi_percent" in roi_dict
    assert "downtime_reduction_pct" in roi_dict
    assert "executive_summary" in roi_dict
    assert roi_dict["failures_prevented"] == 2
