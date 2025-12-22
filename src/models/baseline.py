import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task

from src.features.eda_insights import load_eda_insights
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class LAMABaseline:
    """LightAutoML baseline model wrapper."""

    CONFIGS = {
        "default": {
            "general_params": {"use_algos": [["lgb", "linear_l2"]]},
            "tuning_params": {"max_tuning_iter": 10, "max_tuning_time": 60},
        },
        "tuned": {
            "general_params": {"use_algos": [["linear_l2", "lgb"], ["lgb_tuned"]]},
            "tuning_params": {"max_tuning_iter": 100, "max_tuning_time": 300},
            "use_utilized": True,
        },
        "full": {
            "general_params": {"use_algos": [["lgb", "lgb_tuned", "linear_l2", "cb", "cb_tuned"]]},
            "tuning_params": {"max_tuning_iter": 50, "max_tuning_time": 300},
        },
    }

    def __init__(
        self,
        config_name: str = "default",
        timeout: int = 600,
        n_threads: int = 4,
        n_folds: int = 5,
        random_state: int = 42,
        use_utilized: bool = False,
        eda_insights: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config_name not in self.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(self.CONFIGS.keys())}")

        self.config_name = config_name
        self.timeout = timeout
        self.n_threads = n_threads
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_utilized = use_utilized

        try:
            self.eda_insights = eda_insights or load_eda_insights()
        except FileNotFoundError:
            self.eda_insights = {"use_log_target": True}

        self._automl: Optional[TabularAutoML] = None
        self._oof_predictions: Optional[np.ndarray] = None
        self._config = self.CONFIGS[config_name]

    def _create_automl(self) -> TabularAutoML:
        task = Task(name="reg", metric="mae")
        use_utilized = self._config.get("use_utilized", self.use_utilized)
        AutoMLClass = TabularUtilizedAutoML if use_utilized else TabularAutoML

        return AutoMLClass(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.n_threads,
            general_params=self._config["general_params"],
            tuning_params=self._config["tuning_params"],
            reader_params={
                "n_jobs": self.n_threads,
                "cv": self.n_folds,
                "random_state": self.random_state,
            },
            lgb_params={"default_params": {"num_threads": self.n_threads}},
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "loss",
        drop_cols: Optional[list] = None,
    ) -> "LAMABaseline":
        if drop_cols is None:
            drop_cols = ["id"]

        logger.info(f"Fitting LAMA baseline ({self.config_name}), shape: {train_df.shape}")
        self._automl = self._create_automl()

        self._oof_predictions = self._automl.fit_predict(
            train_df,
            roles={"target": target_col, "drop": drop_cols},
            verbose=1,
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._automl is None:
            raise ValueError("Model not fitted. Call fit() first.")
        predictions = self._automl.predict(test_df)
        return predictions.data[:, 0]

    def get_oof_predictions(self) -> np.ndarray:
        if self._oof_predictions is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._oof_predictions.data[:, 0]

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        return_all_metrics: bool = False,
    ) -> Union[float, Dict[str, float]]:
        if y_pred is None:
            y_pred = self.get_oof_predictions()

        metrics = compute_metrics(y_true, y_pred)
        logger.info(f"[{self.config_name}] Log MAE: {metrics['log_mae']:.4f}, Raw MAE: {metrics['mae']:.4f}")

        return metrics if return_all_metrics else metrics["log_mae"]

