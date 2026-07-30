"""Unified Deployment Pipeline for API and Streamlit Dashboard Services."""

from typing import Any

import pandas as pd

from src.explainability.shap_explainer import ModelExplainer
from src.models.predict import PredictiveMaintenanceInferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeploymentPipeline:
    """High-level pipeline wrapper encapsulating inference, SHAP explanations, and telemetry analysis."""

    def __init__(self, model_name: str = "best_model"):
        self.engine = PredictiveMaintenanceInferenceEngine(model_name=model_name)
        self.explainer: ModelExplainer | None = None

    def run_inference(self, telemetry_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Runs full prediction pipeline on raw telemetry dataframe.

        Returns:
            List[Dict[str, Any]]: Failure risk assessment and RUL predictions per record.
        """
        return self.engine.predict_failure_risk(telemetry_df)

    def explain_prediction(self, single_record_df: pd.DataFrame) -> dict[str, Any]:
        """Generates SHAP feature contribution breakdown for a single telemetry record.

        Args:
            single_record_df (pd.DataFrame): 1-row telemetry dataframe.

        Returns:
            Dict[str, Any]: SHAP explanation breakdown with top degradation drivers.
        """
        if self.explainer is None:
            self.explainer = ModelExplainer(
                model=self.engine.model,
                feature_names=self.engine.feature_names,
                model_type="tree",
            )

        X_inst = self.engine._prepare_features(single_record_df)
        explanation = self.explainer.explain_instance(X_inst[0])
        return explanation
