"""
preprocessing.py

It is Streaming preprocessing utilities for NumCompute Stream.


 Python standard library and Numpy were used here.


This module provides: SimpleImputer, StandardScaler and  OneHotEncoder


In this module, each class supports streaming-compatible partial_fit().
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class SimpleImputer:
    """
    Streaming imputer for missing numeric values.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}, default="mean"
        Strategy used to replace NaN values.

    fill_value : float, default=0.0
        Value used when strategy="constant".

    Notes
    -----
    This class expects X with shape:
    (n_samples, n_features)

    For streaming learning, partial_fit() updates the missing-value
    estimates chunk by chunk.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in {"mean", "median", "constant"}:
            raise ValueError("strategy must be 'mean', 'median', or 'constant'.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.reset()

    def reset(self) -> None:
        """
        Reset the imputer.
        """

        self.n_features_: Optional[int] = None
        self.statistics_: Optional[np.ndarray] = None
        self.count_: Optional[np.ndarray] = None
        self.sum_: Optional[np.ndarray] = None
        self.values_: List[np.ndarray] = []

    def partial_fit(self, X: np.ndarray) -> "SimpleImputer":
        """
        Update imputation statistics using one chunk.

        Parameters
        ----------
        X : np.ndarray
            Input chunk with shape (n_samples, n_features).

        Returns
        -------
        self : SimpleImputer
        """

        X = self._validate_X(X)

        if self.n_features_ is None:
            self._initialise(X.shape[1])

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        if self.strategy == "constant":
            self.statistics_ = np.full(self.n_features_, self.fill_value, dtype=float)
            return self

        valid_mask = ~np.isnan(X)

        if self.strategy == "mean":
            chunk_count = np.sum(valid_mask, axis=0)
            chunk_sum = np.sum(np.where(valid_mask, X, 0.0), axis=0)

            self.count_ += chunk_count
            self.sum_ += chunk_sum

            self.statistics_ = np.divide(
                self.sum_,
                self.count_,
                out=np.zeros_like(self.sum_, dtype=float),
                where=self.count_ > 0,
            )

        elif self.strategy == "median":
            self.values_.append(X.copy())
            all_values = np.vstack(self.values_)

            self.statistics_ = np.nanmedian(all_values, axis=0)

            # If a whole column is NaN, nanmedian returns NaN.
            # Replace that with 0.0 for numerical stability.
            self.statistics_ = np.where(
                np.isnan(self.statistics_),
                0.0,
                self.statistics_,
            )

        return self

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        """
        Fit imputer from scratch.
        """

        self.reset()
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace NaN values using learned statistics.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_samples, n_features).

        Returns
        -------
        X_out : np.ndarray
            Transformed data with NaN values replaced.
        """

        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        X_out = X.copy()
        nan_mask = np.isnan(X_out)

        if np.any(nan_mask):
            X_out[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])

        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit imputer and transform X.
        """

        return self.fit(X).transform(X)

    def _initialise(self, n_features: int) -> None:
        """
        Going to Initialise arrays
        """

        self.n_features_ = n_features
        self.statistics_ = np.zeros(n_features, dtype=float)
        self.count_ = np.zeros(n_features, dtype=float)
        self.sum_ = np.zeros(n_features, dtype=float)

    def _check_fitted(self) -> None:
        """
        Going to check if imputer has been fitted.
        """

        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer has not been fitted yet.")

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
        I am validating input feature matrix.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        return X


class StandardScaler:
    """
    This is Streaming standard scaler.

    Formula of Standardisation is given below:
    X_scaled = (X - mean) / standard_deviation

    This class is being used to update running mean and variance chunk by chunk.

   
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Reset scaler.
        """

        self.n_features_: Optional[int] = None
        self.count_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.M2_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def partial_fit(self, X: np.ndarray) -> "StandardScaler":
        """
       It is going to Update running mean and variance from one chunk.

        Parameters
        ----------
        X : np.ndarray
            Input chunk with shape (n_samples, n_features).

        Returns
        -------
        self : StandardScaler
        """

        X = self._validate_X(X)

        if self.n_features_ is None:
            self._initialise(X.shape[1])

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        valid_mask = ~np.isnan(X)
        chunk_count = np.sum(valid_mask, axis=0)
        safe_X = np.where(valid_mask, X, 0.0)

        chunk_sum = np.sum(safe_X, axis=0)

        chunk_mean = np.divide(
            chunk_sum,
            chunk_count,
            out=np.zeros_like(chunk_sum, dtype=float),
            where=chunk_count > 0,
        )

        centered = np.where(valid_mask, X - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered ** 2, axis=0)

        self._merge_stats(chunk_count, chunk_mean, chunk_M2)
        self._update_variance_and_scale()

        return self

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
       It is working to fit scaler from scratch.
        """

        self.reset()
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale X using running mean and standard deviation.
        """

        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        X_out = X.copy()

        # If transform receives NaN values, replace them with running mean
        # before scaling to avoid NaN outputs.
        nan_mask = np.isnan(X_out)
        if np.any(nan_mask):
            X_out[nan_mask] = np.take(self.mean_, np.where(nan_mask)[1])

        return (X_out - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform X.
        """

        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        I am converting scaled values back to original scale here
        """

        self._check_fitted()
        X_scaled = self._validate_X(X_scaled)

        if X_scaled.shape[1] != self.n_features_:
            raise ValueError(
                f"X_scaled has {X_scaled.shape[1]} features, "
                f"but expected {self.n_features_}."
            )

        return X_scaled * self.scale_ + self.mean_

    def _merge_stats(
        self,
        chunk_count: np.ndarray,
        chunk_mean: np.ndarray,
        chunk_M2: np.ndarray,
    ) -> None:
        """
       It is being used to merge current running statistics with chunk statistics.
        """

        old_count = self.count_
        old_mean = self.mean_
        old_M2 = self.M2_

        new_count = old_count + chunk_count

        delta = chunk_mean - old_mean

        new_mean = np.where(
            new_count > 0,
            old_mean + delta * (chunk_count / np.where(new_count == 0, 1, new_count)),
            old_mean,
        )

        correction = (
            delta ** 2
            * old_count
            * chunk_count
            / np.where(new_count == 0, 1, new_count)
        )

        new_M2 = old_M2 + chunk_M2 + correction

        unchanged = chunk_count == 0
        new_mean = np.where(unchanged, old_mean, new_mean)
        new_M2 = np.where(unchanged, old_M2, new_M2)

        self.count_ = new_count
        self.mean_ = new_mean
        self.M2_ = new_M2

    def _update_variance_and_scale(self) -> None:
        """
        It is being done to update variance and scale arrays.
        """

        self.var_ = np.divide(
            self.M2_,
            self.count_,
            out=np.zeros_like(self.M2_, dtype=float),
            where=self.count_ > 0,
        )

        self.scale_ = np.sqrt(self.var_)

        # Avoid division by zero for constant columns.
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

    def _initialise(self, n_features: int) -> None:
        """
        It is being done to initialise scaler arrays.
        """

        self.n_features_ = n_features
        self.count_ = np.zeros(n_features, dtype=float)
        self.mean_ = np.zeros(n_features, dtype=float)
        self.M2_ = np.zeros(n_features, dtype=float)
        self.var_ = np.zeros(n_features, dtype=float)
        self.scale_ = np.ones(n_features, dtype=float)

    def _check_fitted(self) -> None:
        """
         Going to check whether scaler has been fitted here.
        """

        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
        Validate input X.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        return X


class OneHotEncoder:
    """
    This is Streaming one-hot encoder for categorical integer/string values.

    
    """

    def __init__(self, handle_unknown: str = "ignore") -> None:
        if handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")

        self.handle_unknown = handle_unknown
        self.reset()

    def reset(self) -> None:
        """
        Reset encoder.
        """

        self.n_features_: Optional[int] = None
        self.categories_: Optional[List[np.ndarray]] = None
        self.category_to_index_: Optional[List[Dict[object, int]]] = None

    def partial_fit(self, X: np.ndarray) -> "OneHotEncoder":
        """
        Update categories using one chunk.

        Parameters
        ----------
        X : np.ndarray
            Categorical data with shape (n_samples, n_features).

        Returns
        -------
        self : OneHotEncoder
        """

        X = self._validate_X(X)

        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self.categories_ = []
            self.category_to_index_ = []

            for feature_idx in range(self.n_features_):
                cats = self._unique_non_nan(X[:, feature_idx])
                self.categories_.append(cats)
                self.category_to_index_.append(
                    {category: idx for idx, category in enumerate(cats)}
                )

            return self

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        for feature_idx in range(self.n_features_):
            old_categories = self.categories_[feature_idx]
            new_categories = self._unique_non_nan(X[:, feature_idx])

            combined = np.unique(np.concatenate((old_categories, new_categories)))

            self.categories_[feature_idx] = combined
            self.category_to_index_[feature_idx] = {
                category: idx for idx, category in enumerate(combined)
            }

        return self

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        """
        Fit encoder from scratch.
        """

        self.reset()
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform categorical values into one-hot numeric array.
        """

        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        n_samples = X.shape[0]
        output_width = sum(len(categories) for categories in self.categories_)
        output = np.zeros((n_samples, output_width), dtype=float)

        start_col = 0

        for feature_idx in range(self.n_features_):
            categories = self.categories_[feature_idx]
            mapping = self.category_to_index_[feature_idx]

            for sample_idx in range(n_samples):
                value = X[sample_idx, feature_idx]

                if self._is_nan_like(value):
                    continue

                if value not in mapping:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category found: {value}")
                    continue

                col = start_col + mapping[value]
                output[sample_idx, col] = 1.0

            start_col += len(categories)

        return output

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit encoder and transform X.
        """

        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Return output feature names.
        """

        self._check_fitted()

        if input_features is None:
            input_features = [f"x{idx}" for idx in range(self.n_features_)]

        if len(input_features) != self.n_features_:
            raise ValueError(
                "Length of input_features must match number of fitted features."
            )

        names = []

        for feature_name, categories in zip(input_features, self.categories_):
            for category in categories:
                names.append(f"{feature_name}_{category}")

        return names

    def _check_fitted(self) -> None:
        """
        Check if encoder has been fitted.
        """

        if self.categories_ is None or self.category_to_index_ is None:
            raise RuntimeError("OneHotEncoder has not been fitted yet.")

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
        Validate categorical input.

        Unlike numeric preprocessors, this does not force dtype=float
        because categories may be strings.
        """

        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        return X

    @staticmethod
    def _is_nan_like(value: object) -> bool:
        """
        Check whether value is NaN-like.
        """

        try:
            return bool(np.isnan(value))
        except TypeError:
            return value is None or value == ""

    @classmethod
    def _unique_non_nan(cls, values: np.ndarray) -> np.ndarray:
        """
        Return unique values excluding NaN-like values.
        """

        clean_values = []

        for value in values:
            if not cls._is_nan_like(value):
                clean_values.append(value)

        if len(clean_values) == 0:
            return np.array([], dtype=object)

        return np.unique(np.array(clean_values, dtype=object))