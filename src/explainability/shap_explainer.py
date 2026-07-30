"""SHAP Explainability and Model Interpretability module."""

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """Computes SHAP values, feature importance, and individual sample explanations."""

    def __init__(self, model: Any, feature_names: list[str], model_type: str = "tree"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer: shap.Explainer | None = None

    def fit_explainer(self, X_background: np.ndarray) -> "ModelExplainer":
        """Fits appropriate SHAP explainer (TreeExplainer or Kernel/Linear/Explainer)."""
        try:
            if self.model_type in ["tree", "xgboost", "lightgbm", "catboost", "random_forest"]:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use Kernel or Permutation Explainer with background sample
                bg_sample = shap.sample(X_background, 50) if len(X_background) > 50 else X_background
                self.explainer = shap.Explainer(self.model.predict, bg_sample)
            logger.info(f"Fitted SHAP {type(self.explainer).__name__} for model explainability.")
        except Exception as e:
            logger.warning(f"Defaulting to KernelExplainer due to: {e}")
            bg_sample = shap.sample(X_background, 30) if len(X_background) > 30 else X_background
            self.explainer = shap.KernelExplainer(self.model.predict, bg_sample)
        return self

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Computes SHAP value matrix for input feature matrix X."""
        if self.explainer is None:
            self.fit_explainer(X)

        shap_vals = self.explainer(X)
        if hasattr(shap_vals, "values"):
            values = shap_vals.values
        else:
            values = shap_vals

        if isinstance(values, list):
            values = values[0]  # If multi-class, select primary output class

        return values

    def get_global_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Calculates global mean absolute SHAP feature importances."""
        shap_vals = self.compute_shap_values(X)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        df_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": mean_abs_shap}
        ).sort_values("importance", ascending=False)

        return df_importance

    def explain_instance(self, instance: np.ndarray) -> dict[str, Any]:
        """Provides local SHAP explanation for a single prediction instance.

        Args:
            instance (np.ndarray): 1D array of feature values for a single cycle.

        Returns:
            Dict[str, Any]: Top contributing features, directional impacts, and base value.
        """
        X_inst = instance.reshape(1, -1)
        shap_vals = self.compute_shap_values(X_inst)[0]

        base_val = (
            float(self.explainer.expected_value)
            if hasattr(self.explainer, "expected_value") and not isinstance(self.explainer.expected_value, np.ndarray)
            else 0.0
        )

        contribs = []
        for feat, val, shap_val in zip(self.feature_names, instance, shap_vals):
            contribs.append(
                {
                    "feature": feat,
                    "value": float(val),
                    "shap_value": float(shap_val),
                    "impact": "Increases RUL/Health" if shap_val > 0 else "Decreases RUL (Degradation)",
                }
            )

        # Sort by absolute SHAP magnitude
        contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": round(base_val, 2),
            "predicted_contribution_sum": round(float(np.sum(shap_vals)), 2),
            "top_drivers": contribs[:10],
        }
