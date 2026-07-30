"""Inference Engine for RUL Prediction and Industrial Failure Risk Assessment."""

from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_engineering import engineer_all_features
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictiveMaintenanceInferenceEngine:
    """Production Inference Engine loading best trained model & preprocessor."""

    def __init__(self, model_name: str = "best_model", models_dir: str = None):
        self.registry = ModelRegistry(models_dir)
        self.model, self.feature_names, self.preprocessor, self.metrics = self.registry.load_model(
            model_name
        )
        logger.info(f"Initialized Inference Engine with model '{model_name}'.")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineers features and aligns with training feature set."""
        df_feats = engineer_all_features(df)

        # Missing feature handling: add 0.0 for any missing features
        for col in self.feature_names:
            if col not in df_feats.columns:
                df_feats[col] = 0.0

        # Transform using fitted preprocessor
        df_scaled = self.preprocessor.transform(df_feats)
        X = df_scaled[self.feature_names].values
        return X

    def predict_rul(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts Remaining Useful Life (RUL) in operational cycles.

        Args:
            df (pd.DataFrame): Sensor telemetry readings.

        Returns:
            np.ndarray: Predicted RUL values.
        """
        X = self._prepare_features(df)
        preds = self.model.predict(X)
        # RUL cannot be negative
        preds_clipped = np.clip(preds, a_min=0.0, a_max=None)
        return preds_clipped

    def predict_failure_risk(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Predicts RUL, Failure Risk Category, Failure Probability %, and Actionable Recommendation.

        Returns:
            List[Dict[str, Any]]: List of inference dictionaries for each input record.
        """
        ruls = self.predict_rul(df)
        results = []

        for _idx, rul in enumerate(ruls):
            rul_val = float(np.round(rul, 1))

            # Failure Probability Sigmoid curve based on RUL
            # RUL <= 15 -> Failure prob > 80%
            # RUL 15..30 -> Warning prob 40..80%
            prob_failure = float(np.clip(1.0 / (1.0 + np.exp((rul_val - 20.0) / 4.0)), 0.01, 0.99))

            if rul_val <= 15:
                risk_level = "CRITICAL / FAILURE IMMINENT"
                action = "🚨 EMERGENCY SHUTDOWN REQUIRED: Schedule immediate component replacement within 24 hours."
                status_color = "#FF416C"
            elif rul_val <= 30:
                risk_level = "WARNING"
                action = (
                    "⚠️ ELEVATED WEAR: Schedule preventive maintenance within 5 operating cycles."
                )
                status_color = "#FFB302"
            else:
                risk_level = "HEALTHY / NORMAL"
                action = "✅ OPTIMAL OPERATION: Continue standard monitoring schedule."
                status_color = "#00C6FF"

            results.append(
                {
                    "predicted_rul_cycles": rul_val,
                    "failure_probability": round(prob_failure * 100.0, 1),
                    "risk_level": risk_level,
                    "status_color": status_color,
                    "recommended_action": action,
                }
            )

        return results
