"""Advanced Feature Engineering & Signal Processing for Industrial IoT Data.

Computes time-series rolling statistics, lag features, physical sensor ratios,
exponential moving averages, cumulative degradation, and Fast Fourier Transform (FFT)
frequency metrics.
"""

import numpy as np
import pandas as pd
from scipy.fft import fft

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_rolling_features(
    df: pd.DataFrame, sensor_cols: list[str], window_sizes: list[int] = [5, 10, 20]
) -> pd.DataFrame:
    """Computes rolling statistics (mean, std, min, max) for sensor streams per engine.

    Args:
        df (pd.DataFrame): Input dataframe containing 'engine_id' and sensor columns.
        sensor_cols (List[str]): List of sensor column names.
        window_sizes (List[int]): Window cycle lengths (default: [5, 10, 20]).

    Returns:
        pd.DataFrame: Dataframe with appended rolling feature columns.
    """
    df_out = df.copy()

    for window in window_sizes:
        for sensor in sensor_cols:
            # Group by engine_id to prevent data leakage between engines
            grouped = df_out.groupby("engine_id")[sensor]

            df_out[f"{sensor}_roll_mean_{window}"] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df_out[f"{sensor}_roll_std_{window}"] = (
                grouped.transform(lambda x: x.rolling(window=window, min_periods=1).std()).fillna(0)
            )
            df_out[f"{sensor}_roll_min_{window}"] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df_out[f"{sensor}_roll_max_{window}"] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )

    return df_out


def compute_lag_and_diff_features(
    df: pd.DataFrame, sensor_cols: list[str], lags: list[int] = [1, 3, 5]
) -> pd.DataFrame:
    """Computes lag values and cycle-over-cycle rate of change differences."""
    df_out = df.copy()

    for lag in lags:
        for sensor in sensor_cols:
            grouped = df_out.groupby("engine_id")[sensor]
            lag_col = f"{sensor}_lag_{lag}"
            df_out[lag_col] = grouped.shift(lag).bfill()
            df_out[f"{sensor}_diff_{lag}"] = df_out[sensor] - df_out[lag_col]

    return df_out


def compute_sensor_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-specific physical sensor interaction ratios and efficiency indicators."""
    df_out = df.copy()
    cols = set(df_out.columns)

    # CMAPSS Sensor Ratios
    if "sensor_2" in cols and "sensor_3" in cols:
        df_out["temp_pressure_ratio"] = df_out["sensor_2"] / (df_out["sensor_3"] + 1e-5)
    if "sensor_11" in cols and "sensor_12" in cols:
        df_out["stat_total_pressure_ratio"] = df_out["sensor_11"] / (df_out["sensor_12"] + 1e-5)
    if "sensor_14" in cols and "sensor_9" in cols:
        df_out["flow_speed_ratio"] = df_out["sensor_14"] / (df_out["sensor_9"] + 1e-5)

    # Synthetic / Extended IoT Ratios
    if "temperature" in cols and "pressure" in cols:
        df_out["thermal_pressure_idx"] = df_out["temperature"] / (df_out["pressure"] + 1e-5)
    if "power_consumption" in cols and "rpm" in cols:
        df_out["power_rpm_efficiency"] = df_out["power_consumption"] / (df_out["rpm"] + 1e-5)
    if "voltage" in cols and "current" in cols:
        df_out["impedance_ratio"] = df_out["voltage"] / (df_out["current"] + 1e-5)
    if "torque" in cols and "rpm" in cols:
        df_out["mechanical_power_est"] = (df_out["torque"] * df_out["rpm"]) / 9550.0  # kW power estimate formula

    return df_out


def compute_fft_features(df: pd.DataFrame, target_sensor: str = "sensor_11", sample_rate: int = 100) -> pd.DataFrame:
    """Computes Fast Fourier Transform (FFT) spectral energy and peak frequency per engine window."""
    df_out = df.copy()
    if target_sensor not in df_out.columns:
        # Fallback sensor if target_sensor not present
        num_sensors = [c for c in df_out.columns if c.startswith("sensor_") or c in ["vibration", "temperature"]]
        if not num_sensors:
            return df_out
        target_sensor = num_sensors[0]

    fft_energies = []
    fft_peak_freqs = []

    for _, group in df_out.groupby("engine_id"):
        vals = group[target_sensor].values
        n = len(vals)
        if n > 1:
            fft_vals = np.abs(fft(vals))
            freqs = np.fft.fftfreq(n, d=1 / sample_rate)
            half = n // 2
            energy = float(np.sum(fft_vals[:half] ** 2) / n)
            peak_freq = float(np.abs(freqs[:half][np.argmax(fft_vals[:half])]))
        else:
            energy, peak_freq = 0.0, 0.0

        fft_energies.extend([energy] * n)
        fft_peak_freqs.extend([peak_freq] * n)

    df_out[f"{target_sensor}_fft_spectral_energy"] = fft_energies
    df_out[f"{target_sensor}_fft_peak_freq"] = fft_peak_freqs

    return df_out


def engineer_all_features(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    rolling_windows: list[int] = [5, 10, 20],
    lags: list[int] = [1, 3, 5],
) -> pd.DataFrame:
    """Master pipeline executing all feature engineering functions.

    Args:
        df (pd.DataFrame): Input dataframe.
        sensor_cols (Optional[List[str]]): List of raw sensor column names.
        rolling_windows (List[int]): Window cycle sizes.
        lags (List[int]): Lag cycle counts.

    Returns:
        pd.DataFrame: Fully feature-engineered dataframe.
    """
    if sensor_cols is None:
        exclude = {"engine_id", "cycle", "RUL", "RUL_clipped", "failure_risk", "is_failure", "machine_type"}
        sensor_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    logger.info(f"Starting feature engineering on {len(sensor_cols)} sensor channels...")

    df_feats = compute_rolling_features(df, sensor_cols, window_sizes=rolling_windows)
    df_feats = compute_lag_and_diff_features(df_feats, sensor_cols, lags=lags)
    df_feats = compute_sensor_interaction_features(df_feats)
    df_feats = compute_fft_features(df_feats)

    # Cumulative cycle ratio / degradation index
    grouped_cycle = df_feats.groupby("engine_id")["cycle"]
    df_feats["cumulative_cycle_count"] = grouped_cycle.cumcount() + 1
    df_feats["cycle_rate_of_change"] = df_feats["cycle"] / (grouped_cycle.transform("max") + 1e-5)

    new_feats_count = df_feats.shape[1] - df.shape[1]
    logger.info(f"Feature engineering complete. Created {new_feats_count} new features. Total: {df_feats.shape[1]}")

    return df_feats
