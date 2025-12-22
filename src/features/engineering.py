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

    def fit(self, X: pd.DataFrame, y=None) -> "LogTransformer":
        if self.columns is not None:
            self.fitted_columns_ = [c for c in self.columns if c in X.columns]
        else:
            self.fitted_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.fitted_columns_:
            if col in X_copy.columns:
                X_copy[col] = np.log(X_copy[col] + self.shift)
        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Label encode categorical variables. Works with both DataFrames and numpy arrays."""

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, X, y=None) -> "CategoricalEncoder":
        X_arr = np.asarray(X)
        self.n_features_in_ = X_arr.shape[1] if X_arr.ndim > 1 else 1
        self.encoders_ = {}

        for i in range(self.n_features_in_):
            col_data = X_arr[:, i] if X_arr.ndim > 1 else X_arr
            le = LabelEncoder()
            le.fit(col_data.astype(str))
            self.encoders_[i] = le

        return self

    def transform(self, X) -> np.ndarray:
        X_arr = np.asarray(X)
        result = np.zeros_like(X_arr, dtype=np.float64)

        for i in range(self.n_features_in_):
            col_data = X_arr[:, i] if X_arr.ndim > 1 else X_arr
            le = self.encoders_[i]
            encoded = []
            for val in col_data.astype(str):
                if val in le.classes_:
                    encoded.append(le.transform([val])[0])
                else:
                    encoded.append(-1)
            if X_arr.ndim > 1:
                result[:, i] = encoded
            else:
                result = np.array(encoded)

        return result


