from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transformation to specified columns."""

    def __init__(self, columns: Optional[List[str]] = None, shift: float = 1.0) -> None:
        self.columns = columns
        self.shift = shift
        self._fitted_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "LogTransformer":
        if self.columns is not None:
            self._fitted_columns = [c for c in self.columns if c in X.columns]
        else:
            self._fitted_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self._fitted_columns:
            if col in X_copy.columns:
                X_copy[col] = np.log(X_copy[col] + self.shift)
        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Label encode categorical variables."""

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns
        self._encoders: dict = {}
        self._fitted_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder":
        if self.columns is not None:
            self._fitted_columns = [c for c in self.columns if c in X.columns]
        else:
            self._fitted_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in self._fitted_columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self._encoders[col] = le

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self._fitted_columns:
            if col not in X_copy.columns:
                continue
            le = self._encoders[col]
            X_copy[col] = X_copy[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        return X_copy


