"""Business analytics and financial ROI estimation for predictive maintenance."""

from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_business_impact(
    y_true_failure: np.ndarray,
    y_pred_failure: np.ndarray,
    downtime_cost_per_hour: float = 5000.0,
    avg_downtime_hours: float = 12.0,
    preventive_maint_cost: float = 3000.0,
    false_alarm_inspection_cost: float = 500.0,
) -> dict[str, Any]:
    """Calculates financial savings, ROI, and downtime reduction estimates.

    Args:
        y_true_failure (np.ndarray): Binary array of ground truth failures (1=Failure, 0=Normal/Warning).
        y_pred_failure (np.ndarray): Binary array of predicted failures (1=Predicted Failure, 0=Normal/Warning).
        downtime_cost_per_hour (float): Cost of machine downtime per hour (default: $5,000/hr).
        avg_downtime_hours (float): Average duration of unscheduled failure repair (default: 12 hrs).
        preventive_maint_cost (float): Cost of planned preventive maintenance (default: $3,000).
        false_alarm_inspection_cost (float): Cost of unnecessary inspection (default: $500).

    Returns:
        Dict[str, Any]: Business metrics, cost breakdown, net savings, and executive text summary.
    """
    total_samples = len(y_true_failure)
    actual_failures = int(np.sum(y_true_failure))

    # Confusion matrix elements
    tp = int(
        np.sum((y_true_failure == 1) & (y_pred_failure == 1))
    )  # Correctly predicted failures (Prevented)
    fn = int(
        np.sum((y_true_failure == 1) & (y_pred_failure == 0))
    )  # Uncaught failures (Unscheduled downtime)
    fp = int(
        np.sum((y_true_failure == 0) & (y_pred_failure == 1))
    )  # False alarms (Unnecessary inspection)
    int(
        np.sum((y_true_failure == 0) & (y_pred_failure == 0))
    )  # Normal operation correctly identified

    # Financial breakdown
    catastrophic_failure_cost_per_event = (
        downtime_cost_per_hour * avg_downtime_hours
    )  # e.g., $60,000

    # Reactive Cost (Status Quo - No Predictive Maintenance)
    total_reactive_cost = actual_failures * catastrophic_failure_cost_per_event

    # Predictive Cost (With ML Model)
    prevented_failure_cost = tp * (
        preventive_maint_cost + (downtime_cost_per_hour * 2.0)
    )  # Planned fix takes ~2 hrs
    uncaught_failure_cost = fn * catastrophic_failure_cost_per_event
    false_alarm_cost = fp * false_alarm_inspection_cost

    total_predictive_cost = prevented_failure_cost + uncaught_failure_cost + false_alarm_cost

    net_savings = max(0.0, total_reactive_cost - total_predictive_cost)
    roi_percent = (
        (net_savings / (total_predictive_cost + 1e-5)) * 100.0 if total_predictive_cost > 0 else 0.0
    )
    downtime_reduction_pct = (tp / (actual_failures + 1e-5)) * 100.0 if actual_failures > 0 else 0.0

    executive_summary = (
        f"Predictive Maintenance model prevented {tp} out of {actual_failures} potential breakdowns, "
        f"reducing unexpected downtime by {downtime_reduction_pct:.1f}%. "
        f"Total estimated net cost savings: ${net_savings:,.2f} with an estimated ROI of {roi_percent:.1f}%."
    )

    results = {
        "total_units_evaluated": total_samples,
        "actual_failures": actual_failures,
        "failures_prevented": tp,
        "failures_uncaught": fn,
        "false_alarms": fp,
        "downtime_reduction_pct": round(downtime_reduction_pct, 2),
        "total_reactive_cost": round(total_reactive_cost, 2),
        "total_predictive_cost": round(total_predictive_cost, 2),
        "net_cost_savings": round(net_savings, 2),
        "roi_percent": round(roi_percent, 2),
        "executive_summary": executive_summary,
    }

    logger.info(
        f"Financial Cost Analysis Complete: Net Savings=${net_savings:,.2f}, ROI={roi_percent:.1f}%"
    )
    return results
