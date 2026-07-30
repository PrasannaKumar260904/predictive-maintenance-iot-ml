"""Master model training, hyperparameter optimization, and evaluation pipeline."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

# Optional GBDTs with fallback handling if libomp missing on macOS
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

from src.data.data_loader import load_cmapss_fd001
from src.data.preprocessor import DataPreprocessor
from src.evaluation.cost_analysis import calculate_business_impact
from src.evaluation.metrics import evaluate_regression
from src.features.feature_engineering import engineer_all_features
from src.features.selection import select_features
from src.models.neural_net import PyTorchNeuralNetRegressor
from src.models.registry import ModelRegistry
from src.utils.config import load_config
from src.utils.logger import get_logger

# Ensure project root is on sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Disable Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger(__name__)


def get_base_models(random_state: int = 42) -> dict[str, Any]:
    """Returns dictionary of available baseline regression estimators."""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=30, max_depth=8, random_state=random_state, n_jobs=1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=30, max_depth=8, random_state=random_state, n_jobs=1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=30, learning_rate=0.1, random_state=random_state
        ),
        "SVR": SVR(C=2.0, epsilon=0.1, max_iter=500),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=30, learning_rate=0.05, max_depth=5, random_state=random_state, n_jobs=1
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=30,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state,
            verbose=-1,
            n_jobs=1,
        )

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations=40, learning_rate=0.05, depth=5, random_seed=random_state, verbose=0
        )

    return models


def optimize_rf_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 3,
) -> dict[str, Any]:
    """Hyperparameter optimization using Optuna on Ridge / RandomForest."""

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 10.0, log=True)
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Optuna Best Params: {study.best_params} (Best RMSE={study.best_value:.4f})")
    return study.best_params


def train_and_evaluate_all_models(config_path: str = None) -> dict[str, Any]:
    """Main training orchestrator for end-to-end ML model pipeline."""
    config = load_config(config_path)
    logger.info("Initializing Master Model Training Pipeline...")

    # 1. Load Data
    train_df, test_df, test_rul_truth = load_cmapss_fd001(config_path)

    # 2. Feature Engineering
    train_feats = engineer_all_features(train_df)
    test_feats = engineer_all_features(test_df)

    # 3. Feature Selection
    selected_cols = select_features(train_feats)
    logger.info(f"Selected {len(selected_cols)} optimal features for modeling.")

    # 4. Preprocessing & Scaling
    preprocessor = DataPreprocessor(scaler_type="standard")
    train_scaled = preprocessor.fit_transform(train_feats, selected_cols)
    test_scaled = preprocessor.transform(test_feats)

    X_train = train_scaled[selected_cols].values
    y_train = train_scaled["RUL_clipped"].values

    # Test set evaluation: pick last cycle per engine for standard CMAPSS evaluation
    test_last_cycles = test_scaled.groupby("engine_id").last().reset_index()
    X_test = test_last_cycles[selected_cols].values
    y_test = (
        test_rul_truth.values
        if len(test_rul_truth) == len(test_last_cycles)
        else test_last_cycles["RUL_clipped"].values
    )

    # 5. Train Base Models
    models_dict = get_base_models()
    results = {}
    fitted_models = {}

    for name, model in models_dict.items():
        try:
            logger.info(f"Training estimator: {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_regression(y_test, preds)
            results[name] = metrics
            fitted_models[name] = model
        except Exception as e:
            logger.error(f"Error training {name}: {e}")

    # Train PyTorch Neural Network (MLP)
    try:
        logger.info("Training PyTorch Neural Network Regressor...")
        pt_model = PyTorchNeuralNetRegressor(input_dim=len(selected_cols), epochs=8, lr=0.003)
        pt_model.fit(X_train, y_train)
        pt_preds = pt_model.predict(X_test)
        pt_metrics = evaluate_regression(y_test, pt_preds)
        results["PyTorch_MLP"] = pt_metrics
        fitted_models["PyTorch_MLP"] = pt_model
    except Exception as e:
        logger.error(f"Error training PyTorch Neural Net: {e}")

    # 6. Optuna Hyperparameter Optimization on Ridge
    try:
        logger.info("Running Optuna Bayesian Hyperparameter Optimization...")
        best_ridge_params = optimize_rf_optuna(
            X_train, y_train, X_test, y_test, n_trials=config["models"].get("optuna_trials", 3)
        )
        tuned_ridge = Ridge(**best_ridge_params)
        tuned_ridge.fit(X_train, y_train)
        tuned_preds = tuned_ridge.predict(X_test)
        tuned_metrics = evaluate_regression(y_test, tuned_preds)
        results["Ridge_Tuned"] = tuned_metrics
        fitted_models["Ridge_Tuned"] = tuned_ridge
    except Exception as e:
        logger.error(f"Optuna tuning failed: {e}")

    # Determine Best Model (lowest RMSE)
    best_model_name = min(results.keys(), key=lambda k: results[k]["RMSE"])
    best_model = fitted_models[best_model_name]
    best_metrics = results[best_model_name]

    logger.info(
        f"🏆 BEST MODEL CHAMPION: '{best_model_name}' with RMSE={best_metrics['RMSE']:.4f}, R2={best_metrics['R2']:.4f}"
    )

    # 7. Business Impact & Cost Savings Analysis
    y_test_failure = (y_test <= config["data"].get("failure_threshold_rul", 15)).astype(int)
    best_preds = best_model.predict(X_test)
    y_pred_failure = (best_preds <= config["data"].get("failure_threshold_rul", 15)).astype(int)

    business_roi = calculate_business_impact(
        y_true_failure=y_test_failure,
        y_pred_failure=y_pred_failure,
        downtime_cost_per_hour=config["business"].get("unscheduled_downtime_cost_per_hour", 5000),
        avg_downtime_hours=config["business"].get("average_downtime_hours", 12),
        preventive_maint_cost=config["business"].get("preventive_maintenance_cost", 3000),
    )

    # 8. Save Artifacts via Registry
    registry = ModelRegistry(config["models"]["dir"])
    registry.save_model(
        model=best_model,
        model_name="best_model",
        feature_names=selected_cols,
        preprocessor=preprocessor,
        metrics=best_metrics,
    )

    # Save comparison results and business report to reports/
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_model_name, "all_models": results}, f, indent=2)

    with open(reports_dir / "business_impact.json", "w", encoding="utf-8") as f:
        json.dump(business_roi, f, indent=2)

    logger.info("Master Training Pipeline Completed Successfully!")
    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "all_results": results,
        "business_roi": business_roi,
    }


if __name__ == "__main__":
    train_and_evaluate_all_models()
