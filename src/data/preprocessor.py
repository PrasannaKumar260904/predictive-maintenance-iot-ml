"""Data preprocessing and transformation module."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses raw telemetry data, handling missing values, constant features,

    outliers, scaling, and sequence creation for deep learning models.
    """

    def __init__(self, scaler_type: str = "standard", z_threshold: float = 4.0):
        self.scaler_type = scaler_type
        self.z_threshold = z_threshold
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.constant_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.is_fitted: bool = False

    def remove_constant_features(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Identifies and drops columns with zero or near-zero variance."""
        constant_cols = [c for c in feature_cols if df[c].nunique() <= 1 or df[c].std() < 1e-6]
        self.constant_columns = constant_cols
        active_cols = [c for c in feature_cols if c not in constant_cols]
        logger.info(f"Removed {len(constant_cols)} constant features: {constant_cols}")
        return df, active_cols

    def handle_outliers(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Clips extreme outliers based on Z-score threshold."""
        df_clean = df.copy()
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 1e-6:
                    upper = mean + (self.z_threshold * std)
                    lower = mean - (self.z_threshold * std)
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        return df_clean

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "DataPreprocessor":
        """Fits scalers and identifies constant columns on training data."""
        df_temp, active_cols = self.remove_constant_features(df, feature_cols)
        self.feature_columns = active_cols

        df_clean = self.handle_outliers(df_temp, self.feature_columns)
        self.scaler.fit(df_clean[self.feature_columns])
        self.is_fitted = True
        logger.info(f"Fitted DataPreprocessor on {len(self.feature_columns)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input dataframe using fitted scaler and active feature set."""
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before calling transform().")

        df_copy = df.copy()
        # Impute missing values
        df_copy[self.feature_columns] = df_copy[self.feature_columns].ffill().bfill().fillna(0)
        df_clean = self.handle_outliers(df_copy, self.feature_columns)

        scaled_vals = self.scaler.transform(df_clean[self.feature_columns])
        scaled_df = pd.DataFrame(scaled_vals, columns=self.feature_columns, index=df.index)

        # Preserve non-feature metadata columns (e.g. engine_id, cycle, RUL, failure_risk)
        metadata_cols = [
            c
            for c in df.columns
            if c not in self.feature_columns and c not in self.constant_columns
        ]
        for col in metadata_cols:
            scaled_df[col] = df[col].values

        return scaled_df

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Fits on data and transforms it in one step."""
        return self.fit(df, feature_cols).transform(df)

    def create_lstm_sequences(
        self, df: pd.DataFrame, sequence_length: int = 30, target_col: str = "RUL"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Creates 3D sequence tensor (samples, sequence_length, features) for LSTM/GRU neural nets.

        Args:
            df (pd.DataFrame): Transformed dataframe with feature_columns and target_col.
            sequence_length (int): Sliding window sequence length (default: 30).
            target_col (str): Target column name.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_seq, y_seq)
        """
        X_sequences, y_sequences = [], []

        for _, engine_df in df.groupby("engine_id"):
            engine_features = engine_df[self.feature_columns].values
            engine_targets = engine_df[target_col].values

            num_samples = len(engine_features)
            if num_samples < sequence_length:
                # Pad if engine history is shorter than sequence_length
                pad_len = sequence_length - num_samples
                padded_feats = np.pad(engine_features, ((pad_len, 0), (0, 0)), mode="edge")
                X_sequences.append(padded_feats)
                y_sequences.append(engine_targets[-1])
            else:
                for i in range(num_samples - sequence_length + 1):
                    X_sequences.append(engine_features[i : i + sequence_length])
                    y_sequences.append(engine_targets[i + sequence_length - 1])

        return np.array(X_sequences), np.array(y_sequences)
