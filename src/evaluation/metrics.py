"""Model evaluation metrics calculation for regression and classification tasks."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes standard regression evaluation metrics.

    Metrics:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - MAPE (Mean Absolute Percentage Error)
        - R2 (Coefficient of Determination)

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # Avoid zero division in MAPE calculation
    y_true_safe = np.where(y_true == 0, 1e-5, y_true)
    mape = float(mean_absolute_percentage_error(y_true_safe, y_pred)) * 100.0
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
    }

    logger.info(f"Regression Metrics: {metrics}")
    return metrics


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, Any]:
    """Computes standard classification evaluation metrics.

    Metrics:
        - Accuracy, Precision, Recall, F1-Score
        - ROC-AUC (if y_prob provided)
        - Confusion Matrix

    Returns:
        Dict[str, Any]: Evaluation metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    roc_auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                roc_auc = float(roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob))
            else:
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "Confusion_Matrix": cm,
    }

    logger.info(f"Classification Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc}")
    return metrics
