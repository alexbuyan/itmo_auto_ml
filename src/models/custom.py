import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features.eda_insights import load_eda_insights
from src.features.engineering import CategoricalEncoder
from src.utils.metrics import compute_log_mae

logger = logging.getLogger(__name__)


class CustomPipeline:
    """Custom ML pipeline with preprocessing and LightGBM/XGBoost models."""

    MODEL_DEFAULTS = {
        "lightgbm": {
            "n_estimators": 500,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "mae",
            "n_jobs": -1,
            "verbose": -1,
        },
        "xgboost": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:absoluteerror",
            "n_jobs": -1,
        },
    }

    def __init__(
        self,
        model_type: str = "lightgbm",
        use_log_target: Optional[bool] = None,
        n_folds: int = 5,
        random_state: int = 42,
        model_params: Optional[Dict[str, Any]] = None,
        eda_insights: Optional[Dict[str, Any]] = None,
    ) -> None:
        if model_type not in self.MODEL_DEFAULTS:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = model_type
        self.n_folds = n_folds
        self.random_state = random_state

        try:
            self.eda_insights = eda_insights or load_eda_insights()
        except FileNotFoundError:
            self.eda_insights = {"use_log_target": True}

        self.use_log_target = use_log_target if use_log_target is not None else self.eda_insights.get("use_log_target", True)

        self.model_params = self.MODEL_DEFAULTS[model_type].copy()
        if model_params:
            self.model_params.update(model_params)

        self._pipeline: Optional[Pipeline] = None
        self._cat_columns: List[str] = []
        self._cont_columns: List[str] = []
        self._oof_predictions: Optional[np.ndarray] = None
        self._cv_scores: List[float] = []
        self._cv_raw_scores: List[float] = []

    def _create_model(self) -> BaseEstimator:
        if self.model_type == "lightgbm":
            return LGBMRegressor(random_state=self.random_state, **self.model_params)
        elif self.model_type == "xgboost":
            return XGBRegressor(random_state=self.random_state, **self.model_params)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_preprocessor(self, cat_columns: List[str], cont_columns: List[str]) -> ColumnTransformer:
        cat_transformer = Pipeline([("encoder", CategoricalEncoder())])
        cont_transformer = Pipeline([("scaler", StandardScaler())])

        return ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, cat_columns),
                ("cont", cont_transformer, cont_columns),
            ],
            remainder="drop",
        )

    def _build_pipeline(self, cat_columns: List[str], cont_columns: List[str]) -> Pipeline:
        return Pipeline([
            ("preprocessor", self._create_preprocessor(cat_columns, cont_columns)),
            ("model", self._create_model()),
        ])

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        cat_columns: Optional[List[str]] = None,
        cont_columns: Optional[List[str]] = None,
    ) -> "CustomPipeline":
        y_orig = np.array(y)

        self._cat_columns = cat_columns or [c for c in X.columns if c.startswith("cat")]
        self._cont_columns = cont_columns or [c for c in X.columns if c.startswith("cont")]

        y_train = np.log1p(y) if self.use_log_target else np.array(y)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self._oof_predictions = np.zeros(len(X))
        self._cv_scores = []
        self._cv_raw_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y_train[train_idx]
            y_val_orig = y_orig[val_idx]

            fold_pipeline = self._build_pipeline(self._cat_columns, self._cont_columns)
            fold_pipeline.fit(X_tr, y_tr)

            val_pred = fold_pipeline.predict(X_val)
            self._oof_predictions[val_idx] = val_pred

            val_pred_orig = np.expm1(val_pred) if self.use_log_target else val_pred

            fold_log_mae = compute_log_mae(y_val_orig, val_pred_orig)
            fold_raw_mae = mean_absolute_error(y_val_orig, val_pred_orig)
            self._cv_scores.append(fold_log_mae)
            self._cv_raw_scores.append(fold_raw_mae)
            logger.info(f"Fold {fold + 1}: Log MAE = {fold_log_mae:.4f}, Raw MAE = {fold_raw_mae:.4f}")

        self._pipeline = self._build_pipeline(self._cat_columns, self._cont_columns)
        self._pipeline.fit(X, y_train)

        logger.info(f"CV Log MAE: {np.mean(self._cv_scores):.4f} (+/- {np.std(self._cv_scores):.4f})")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        predictions = self._pipeline.predict(X)
        if self.use_log_target:
            predictions = np.expm1(predictions)
        return np.maximum(predictions, 0)

    def get_oof_predictions(self) -> np.ndarray:
        if self._oof_predictions is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        oof = self._oof_predictions.copy()
        return np.expm1(oof) if self.use_log_target else oof

    def get_cv_score(self, metric: str = "log_mae") -> Tuple[float, float]:
        if not self._cv_scores:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        scores = self._cv_scores if metric == "log_mae" else self._cv_raw_scores
        return np.mean(scores), np.std(scores)


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_folds: int = 5,
        random_state: int = 42,
        eda_insights: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_folds = n_folds
        self.random_state = random_state

        try:
            self.eda_insights = eda_insights or load_eda_insights()
        except FileNotFoundError:
            self.eda_insights = {"use_log_target": True}

        self._best_params: Optional[Dict[str, Any]] = None
        self._study: Optional[optuna.Study] = None

    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        if self.model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
        elif self.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            }
        return {}

    def tune(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        cat_columns: Optional[List[str]] = None,
        cont_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cat_columns = cat_columns or [c for c in X.columns if c.startswith("cat")]
        cont_columns = cont_columns or [c for c in X.columns if c.startswith("cont")]

        def objective(trial: optuna.Trial) -> float:
            params = self._get_param_space(trial)
            pipeline = CustomPipeline(
                model_type=self.model_type,
                n_folds=self.n_folds,
                random_state=self.random_state,
                model_params=params,
                eda_insights=self.eda_insights,
            )
            try:
                pipeline.fit(X, y, cat_columns, cont_columns)
                mean_log_mae, _ = pipeline.get_cv_score(metric="log_mae")
                return mean_log_mae
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("inf")

        self._study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")
        self._study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)

        self._best_params = self._study.best_params
        logger.info(f"Best Log MAE: {self._study.best_value:.4f}")
        return self._best_params

    def get_best_pipeline(self) -> CustomPipeline:
        if self._best_params is None:
            raise ValueError("No tuning performed. Call tune() first.")
        return CustomPipeline(
            model_type=self.model_type,
            model_params=self._best_params,
            eda_insights=self.eda_insights,
        )


class EnsembleModel(BaseEstimator, RegressorMixin):
    """Ensemble of multiple models using averaging or stacking."""

    def __init__(
        self,
        models: Optional[List[BaseEstimator]] = None,
        weights: Optional[List[float]] = None,
        method: str = "average",
        use_log_target: bool = True,
        random_state: int = 42,
    ) -> None:
        self.models = models or []
        self.weights = weights
        self.method = method
        self.use_log_target = use_log_target
        self.random_state = random_state
        self._fitted_models: List[BaseEstimator] = []
        self._meta_model: Optional[BaseEstimator] = None

    def add_model(self, model: BaseEstimator, weight: Optional[float] = None) -> "EnsembleModel":
        self.models.append(model)
        if self.weights is None:
            self.weights = []
        if weight is not None:
            self.weights.append(weight)
        return self

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "EnsembleModel":
        y_train = np.log1p(y) if self.use_log_target else np.array(y)

        if self.method == "average":
            self._fitted_models = []
            for i, model in enumerate(self.models):
                logger.info(f"Fitting model {i + 1}/{len(self.models)}")
                fitted = clone(model)
                fitted.fit(X, y_train)
                self._fitted_models.append(fitted)

            if self.weights is None or len(self.weights) != len(self.models):
                self.weights = [1.0 / len(self.models)] * len(self.models)
            else:
                weight_sum = sum(self.weights)
                self.weights = [w / weight_sum for w in self.weights]

        elif self.method == "stacking":
            estimators = [(f"model_{i}", model) for i, model in enumerate(self.models)]
            self._meta_model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5,
                n_jobs=-1,
            )
            self._meta_model.fit(X, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.method == "average":
            predictions = sum(w * m.predict(X) for m, w in zip(self._fitted_models, self.weights))
        else:
            predictions = self._meta_model.predict(X)

        if self.use_log_target:
            predictions = np.expm1(predictions)
        return np.maximum(predictions, 0)


def create_submission(predictions: np.ndarray, test_ids: np.ndarray, output_path: str = "submission.csv") -> pd.DataFrame:
    """Create Kaggle submission file."""
    submission = pd.DataFrame({"id": test_ids, "loss": predictions})
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}, mean={predictions.mean():.2f}, std={predictions.std():.2f}")
    return submission
