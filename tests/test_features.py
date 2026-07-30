"""Unit tests for feature engineering and feature selection."""

import pandas as pd
import pytest

from src.features.feature_engineering import (
    compute_fft_features,
    compute_lag_and_diff_features,
    compute_rolling_features,
    compute_sensor_interaction_features,
    engineer_all_features,
)
from src.features.selection import (
    select_features,
)


@pytest.fixture
def sample_telemetry_df():
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "temperature": [70.0, 71.5, 73.0, 75.0, 78.0, 68.0, 69.0, 70.0, 71.0, 72.0],
            "pressure": [150.0, 148.0, 145.0, 142.0, 138.0, 152.0, 151.0, 150.0, 149.0, 148.0],
            "vibration": [0.5, 0.8, 1.2, 1.8, 2.5, 0.4, 0.5, 0.6, 0.7, 0.8],
            "voltage": [400.0, 398.0, 396.0, 394.0, 390.0, 402.0, 401.0, 400.0, 399.0, 398.0],
            "current": [15.0, 16.0, 17.5, 19.0, 21.0, 14.5, 15.0, 15.5, 16.0, 16.5],
            "rpm": [3000.0, 2980.0, 2950.0, 2910.0, 2850.0, 3010.0, 3000.0, 2990.0, 2980.0, 2970.0],
            "RUL": [100, 99, 98, 97, 96, 120, 119, 118, 117, 116],
            "RUL_clipped": [100, 99, 98, 97, 96, 125, 125, 118, 117, 116],
        }
    )


def test_compute_rolling_features(sample_telemetry_df):
    df_roll = compute_rolling_features(sample_telemetry_df, ["temperature"], window_sizes=[3])
    assert "temperature_roll_mean_3" in df_roll.columns
    assert "temperature_roll_std_3" in df_roll.columns


def test_compute_lag_and_diff_features(sample_telemetry_df):
    df_lag = compute_lag_and_diff_features(sample_telemetry_df, ["temperature"], lags=[1])
    assert "temperature_lag_1" in df_lag.columns
    assert "temperature_diff_1" in df_lag.columns


def test_compute_sensor_interaction_features(sample_telemetry_df):
    df_inter = compute_sensor_interaction_features(sample_telemetry_df)
    assert "thermal_pressure_idx" in df_inter.columns
    assert "impedance_ratio" in df_inter.columns


def test_compute_fft_features(sample_telemetry_df):
    df_fft = compute_fft_features(sample_telemetry_df, target_sensor="vibration")
    assert "vibration_fft_spectral_energy" in df_fft.columns
    assert "vibration_fft_peak_freq" in df_fft.columns


def test_engineer_all_features(sample_telemetry_df):
    df_all = engineer_all_features(sample_telemetry_df, rolling_windows=[2], lags=[1])
    assert df_all.shape[1] > sample_telemetry_df.shape[1]


def test_feature_selection(sample_telemetry_df):
    df_all = engineer_all_features(sample_telemetry_df, rolling_windows=[2], lags=[1])
    selected = select_features(df_all, correlation_threshold=0.95)
    assert isinstance(selected, list)
    assert len(selected) > 0
