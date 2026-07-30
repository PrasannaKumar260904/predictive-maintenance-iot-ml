"""Unit tests for SHAP Explainability module."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.explainability.shap_explainer import ModelExplainer


def test_shap_explainer():
    X = np.random.randn(30, 5)
    y = np.random.randn(30)
    feature_names = [f"feat_{i}" for i in range(5)]

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    explainer = ModelExplainer(model=model, feature_names=feature_names, model_type="tree")
    explainer.fit_explainer(X)

    shap_vals = explainer.compute_shap_values(X)
    assert shap_vals.shape == (30, 5)

    df_imp = explainer.get_global_feature_importance(X)
    assert isinstance(df_imp, pd.DataFrame)
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns

    explanation = explainer.explain_instance(X[0])
    assert "top_drivers" in explanation
    assert "base_value" in explanation
    assert len(explanation["top_drivers"]) <= 5
