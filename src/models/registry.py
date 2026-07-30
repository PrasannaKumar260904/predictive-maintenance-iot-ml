"""Model artifact persistence and versioning manager."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Manages serialization, deserialization, metadata tracking, and artifact persistence."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            config = load_config()
            models_dir = config["models"]["dir"]

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        feature_names: List[str],
        preprocessor: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Saves trained model, feature names, preprocessor, and metrics JSON to disk.

        Args:
            model (Any): Trained estimator object.
            model_name (str): Unique model name tag.
            feature_names (List[str]): List of feature names used during training.
            preprocessor (Optional[Any]): Fitted DataPreprocessor instance.
            metrics (Optional[Dict[str, float]]): Model performance metrics dictionary.

        Returns:
            Path: Directory path where model artifacts were saved.
        """
        artifact_dir = self.models_dir / model_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_file = artifact_dir / "model.joblib"
        features_file = artifact_dir / "feature_names.json"
        preprocessor_file = artifact_dir / "preprocessor.joblib"
        metrics_file = artifact_dir / "metrics.json"

        # Save model with high compression ratio
        joblib.dump(model, model_file, compress=3)

        # Save feature names
        with open(features_file, "w", encoding="utf-8") as f:
            json.dump({"feature_names": feature_names}, f, indent=2)

        # Save preprocessor if provided
        if preprocessor is not None:
            joblib.dump(preprocessor, preprocessor_file, compress=3)

        # Save metrics if provided
        if metrics is not None:
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        logger.info(f"Successfully saved model artifacts for '{model_name}' to {artifact_dir}")
        return artifact_dir

    def load_model(self, model_name: str) -> Tuple[Any, List[str], Optional[Any], Optional[Dict[str, float]]]:
        """Loads model estimator, feature names, preprocessor, and metrics."""
        artifact_dir = self.models_dir / model_name
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {artifact_dir}")

        model_file = artifact_dir / "model.joblib"
        features_file = artifact_dir / "feature_names.json"
        preprocessor_file = artifact_dir / "preprocessor.joblib"
        metrics_file = artifact_dir / "metrics.json"

        model = joblib.load(model_file)

        with open(features_file, "r", encoding="utf-8") as f:
            feature_names = json.load(f)["feature_names"]

        preprocessor = joblib.load(preprocessor_file) if preprocessor_file.exists() else None

        metrics = None
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        logger.info(f"Loaded model '{model_name}' from {artifact_dir}")
        return model, feature_names, preprocessor, metrics
