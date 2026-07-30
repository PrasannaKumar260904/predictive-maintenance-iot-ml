"""Unit tests for data generation, loading, and preprocessing."""

import pandas as pd

from src.data.data_loader import calculate_rul, load_cmapss_fd001
from src.data.generator import generate_iot_sensor_data
from src.data.preprocessor import DataPreprocessor


def test_generate_iot_sensor_data():
    df = generate_iot_sensor_data(num_engines=5, min_cycles=50, max_cycles=80, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "engine_id" in df.columns
    assert "temperature" in df.columns
    assert "vibration" in df.columns
    assert "RUL" in df.columns
    assert df["engine_id"].nunique() == 5


def test_calculate_rul():
    mock_data = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "sensor_1": [10.0, 11.0, 12.0, 15.0, 16.0],
        }
    )
    df_rul = calculate_rul(mock_data, max_rul_clip=100)
    assert "RUL" in df_rul.columns
    assert "RUL_clipped" in df_rul.columns
    assert "failure_risk" in df_rul.columns
    assert df_rul.loc[0, "RUL"] == 2  # 3 - 1


def test_data_preprocessor():
    mock_data = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1],
            "cycle": [1, 2, 3, 4],
            "constant_sensor": [5.0, 5.0, 5.0, 5.0],
            "active_sensor": [10.0, 12.0, 14.0, 16.0],
            "RUL": [100, 99, 98, 97],
            "RUL_clipped": [100, 99, 98, 97],
        }
    )
    preprocessor = DataPreprocessor()
    df_transformed = preprocessor.fit_transform(mock_data, ["constant_sensor", "active_sensor"])

    assert "constant_sensor" in preprocessor.constant_columns
    assert "active_sensor" in preprocessor.feature_columns
    assert preprocessor.is_fitted
    assert df_transformed.shape[0] == 4


def test_load_cmapss_fd001():
    train_df, test_df, test_rul = load_cmapss_fd001()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) > 0
    assert "RUL" in train_df.columns
