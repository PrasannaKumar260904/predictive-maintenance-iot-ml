"""Feature selection module for eliminating redundant and collinear features."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_low_variance_features(df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.01) -> list[str]:
    """Filters out numeric features with variance below threshold."""
    variances = df[feature_cols].var()
    selected = variances[variances >= threshold].index.tolist()
    dropped = set(feature_cols) - set(selected)
    if dropped:
        logger.info(f"Dropped {len(dropped)} low-variance features (threshold={threshold}): {dropped}")
    return selected


def remove_high_correlation_features(
    df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.95
) -> list[str]:
    """Prunes multicollinear features with correlation coefficient higher than threshold."""
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected = [c for c in feature_cols if c not in to_drop]

    logger.info(
        f"Pruned {len(to_drop)} collinear features with correlation > {threshold}. "
        f"Retained {len(selected)} features."
    )
    return selected


def select_features(
    df: pd.DataFrame,
    candidate_cols: list[str] | None = None,
    variance_threshold: float = 0.001,
    correlation_threshold: float = 0.95,
) -> list[str]:
    """Pipeline for variance filtering and multicollinearity reduction.

    Args:
        df (pd.DataFrame): Dataframe containing features.
        candidate_cols (Optional[List[str]]): List of candidate feature names.
        variance_threshold (float): Minimum variance cutoff.
        correlation_threshold (float): Maximum correlation threshold.

    Returns:
        List[str]: Filtered list of feature column names.
    """
    if candidate_cols is None:
        exclude = {"engine_id", "cycle", "RUL", "RUL_clipped", "failure_risk", "is_failure", "machine_type"}
        candidate_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    cols_var = remove_low_variance_features(df, candidate_cols, threshold=variance_threshold)
    cols_final = remove_high_correlation_features(df, cols_var, threshold=correlation_threshold)

    return cols_final
