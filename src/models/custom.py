import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features.eda_insights import load_eda_insights
from src.features.engineering import CategoricalEncoder
from src.utils.metrics import compute_log_mae

logger = logging.getLogger(__name__)


class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for CatBoostRegressor.
    
    This wrapper ensures compatibility with sklearn's Pipeline by inheriting
    from BaseEstimator and RegressorMixin, which provide __sklearn_tags__.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._catboost_params = kwargs
        self._model: Optional[CatBoostRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostRegressorWrapper":
        self._model = CatBoostRegressor(**self._catboost_params)
        self._model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self._catboost_params.copy()

    def set_params(self, **params: Any) -> "CatBoostRegressorWrapper":
        self._catboost_params.update(params)
        return self


class CustomPipeline:
    """Custom ML pipeline with preprocessing and LightGBM/XGBoost/CatBoost models."""

    MODEL_CONFIGS = {
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
        "catboost": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "MAE",
            "verbose": False,
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
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(self.MODEL_CONFIGS.keys())}")

        self.model_type = model_type
        self.n_folds = n_folds
        self.random_state = random_state

        try:
            self.eda_insights = eda_insights or load_eda_insights()
        except FileNotFoundError:
            self.eda_insights = {"use_log_target": True}

        self.use_log_target = use_log_target if use_log_target is not None else self.eda_insights.get("use_log_target", True)

        self.model_params = self.MODEL_CONFIGS[model_type].copy()
        if model_params:
            self.model_params.update(model_params)

        self._pipeline: Optional[Pipeline] = None
        self._cat_columns: List[str] = []
        self._cont_columns: List[str] = []
        self._oof_predictions: Optional[np.ndarray] = None
        self._cv_scores: List[float] = []
        self._cv_raw_scores: List[float] = []

        logger.info(f"Initialized CustomPipeline with model_type={model_type}, n_folds={n_folds}")
        logger.debug(f"Model parameters: {self.model_params}")
        logger.debug(f"Using log target: {self.use_log_target}")

    def _create_model(self) -> BaseEstimator:
        if self.model_type == "lightgbm":
            return LGBMRegressor(random_state=self.random_state, **self.model_params)
        elif self.model_type == "xgboost":
            return XGBRegressor(random_state=self.random_state, **self.model_params)
        elif self.model_type == "catboost":
            return CatBoostRegressorWrapper(random_state=self.random_state, **self.model_params)
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
        fit_start_time = time.time()
        y_orig = np.array(y)

        self._cat_columns = cat_columns or [c for c in X.columns if c.startswith("cat")]
        self._cont_columns = cont_columns or [c for c in X.columns if c.startswith("cont")]

        logger.info(f"Starting {self.n_folds}-fold CV with {self.model_type}")
        logger.info(f"Data shape: {X.shape}, Cat cols: {len(self._cat_columns)}, Cont cols: {len(self._cont_columns)}")
        logger.info(f"Target stats: mean={np.mean(y_orig):.2f}, std={np.std(y_orig):.2f}, min={np.min(y_orig):.2f}, max={np.max(y_orig):.2f}")

        y_train = np.log1p(y) if self.use_log_target else np.array(y)
        if self.use_log_target:
            logger.info(f"Applied log1p transform to target. New stats: mean={np.mean(y_train):.4f}, std={np.std(y_train):.4f}")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self._oof_predictions = np.zeros(len(X))
        self._cv_scores = []
        self._cv_raw_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_start_time = time.time()
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y_train[train_idx]
            y_val_orig = y_orig[val_idx]

            logger.info(f"Fold {fold + 1}/{self.n_folds}: Train={len(train_idx):,}, Val={len(val_idx):,}")

            fold_pipeline = self._build_pipeline(self._cat_columns, self._cont_columns)
            fold_pipeline.fit(X_tr, y_tr)

            val_pred = fold_pipeline.predict(X_val)
            self._oof_predictions[val_idx] = val_pred

            val_pred_orig = np.expm1(val_pred) if self.use_log_target else val_pred

            fold_log_mae = compute_log_mae(y_val_orig, val_pred_orig)
            fold_raw_mae = mean_absolute_error(y_val_orig, val_pred_orig)
            self._cv_scores.append(fold_log_mae)
            self._cv_raw_scores.append(fold_raw_mae)

            fold_duration = time.time() - fold_start_time
            logger.info(f"Fold {fold + 1}/{self.n_folds}: Log MAE = {fold_log_mae:.4f}, Raw MAE = {fold_raw_mae:.2f} (took {fold_duration:.1f}s)")

        logger.info("Training final model on full dataset...")
        self._pipeline = self._build_pipeline(self._cat_columns, self._cont_columns)
        self._pipeline.fit(X, y_train)

        total_duration = time.time() - fit_start_time
        logger.info(f"CV Log MAE: {np.mean(self._cv_scores):.4f} (+/- {np.std(self._cv_scores):.4f})")
        logger.info(f"CV Raw MAE: {np.mean(self._cv_raw_scores):.2f} (+/- {np.std(self._cv_raw_scores):.2f})")
        logger.info(f"Training complete. Total time: {total_duration:.1f}s")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        logger.info(f"Generating predictions for {len(X):,} samples...")
        predictions = self._pipeline.predict(X)
        if self.use_log_target:
            predictions = np.expm1(predictions)
        predictions = np.maximum(predictions, 0)
        logger.info(f"Predictions: mean={predictions.mean():.2f}, std={predictions.std():.2f}, min={predictions.min():.2f}, max={predictions.max():.2f}")
        return predictions

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

    def save_model(self, path: str) -> None:
        """Save trained pipeline and metadata to disk.
        
        Args:
            path: File path to save the model (recommended: .joblib extension)
        """
        if self._pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "pipeline": self._pipeline,
            "cat_columns": self._cat_columns,
            "cont_columns": self._cont_columns,
            "oof_predictions": self._oof_predictions,
            "cv_scores": self._cv_scores,
            "cv_raw_scores": self._cv_raw_scores,
            "model_type": self.model_type,
            "use_log_target": self.use_log_target,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "model_params": self.model_params,
            "eda_insights": self.eda_insights,
        }
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> "CustomPipeline":
        """Load trained pipeline from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            CustomPipeline instance with restored state
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        state = joblib.load(path)
        
        # Create instance with saved configuration
        instance = cls(
            model_type=state["model_type"],
            use_log_target=state["use_log_target"],
            n_folds=state["n_folds"],
            random_state=state["random_state"],
            model_params=state["model_params"],
            eda_insights=state.get("eda_insights"),
        )
        
        # Restore trained state
        instance._pipeline = state["pipeline"]
        instance._cat_columns = state["cat_columns"]
        instance._cont_columns = state["cont_columns"]
        instance._oof_predictions = state["oof_predictions"]
        instance._cv_scores = state["cv_scores"]
        instance._cv_raw_scores = state["cv_raw_scores"]
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Model type: {instance.model_type}, CV Log MAE: {np.mean(instance._cv_scores):.4f}")
        return instance


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
        elif self.model_type == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            }
        return {}

    def tune(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        cat_columns: Optional[List[str]] = None,
        cont_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        tune_start_time = time.time()
        cat_columns = cat_columns or [c for c in X.columns if c.startswith("cat")]
        cont_columns = cont_columns or [c for c in X.columns if c.startswith("cont")]

        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        logger.info(f"Tuning config: n_trials={self.n_trials}, timeout={self.timeout}s, n_folds={self.n_folds}")
        logger.info(f"Data shape: {X.shape}")

        def objective(trial: optuna.Trial) -> float:
            params = self._get_param_space(trial)
            logger.debug(f"Trial {trial.number}: Testing params {params}")
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
                logger.info(f"Trial {trial.number}: Log MAE = {mean_log_mae:.4f}")
                return mean_log_mae
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("inf")

        self._study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)

        self._best_params = self._study.best_params
        tune_duration = time.time() - tune_start_time
        logger.info(f"Tuning complete. Best Log MAE: {self._study.best_value:.4f}")
        logger.info(f"Best parameters: {self._best_params}")
        logger.info(f"Total tuning time: {tune_duration:.1f}s")
        return self._best_params

    def get_best_pipeline(self) -> CustomPipeline:
        if self._best_params is None:
            raise ValueError("No tuning performed. Call tune() first.")
        return CustomPipeline(
            model_type=self.model_type,
            model_params=self._best_params,
            eda_insights=self.eda_insights,
        )


def create_submission(predictions: np.ndarray, test_ids: np.ndarray, output_path: str = "submission.csv") -> pd.DataFrame:
    """Create Kaggle submission file."""
    submission = pd.DataFrame({"id": test_ids, "loss": predictions})
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}, mean={predictions.mean():.2f}, std={predictions.std():.2f}")
    return submission
