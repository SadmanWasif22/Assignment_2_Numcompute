"""
stats.py

Streaming statistical utilities for NumCompute Stream.

Allowed libraries:
- Python standard library
- NumPy

This module supports chunk-wise updates for:
- mean
- variance
- standard deviation
- min / max
- approximate quantiles
- histograms
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class StreamingStats:
    """
    It is Streaming statistics for 2D numeric data.

   
    This class will ignore NaN values when updating statistics.
    """

    def __init__(self, store_values: bool = True, histogram_bins: int = 10) -> None:
        if histogram_bins <= 0:
            raise ValueError("Here, histogram_bins should be greater than 0.")

        self.store_values = store_values
        self.histogram_bins = histogram_bins
        self.reset()

    def reset(self) -> None:
        """
        Reset all stored statistics.
        """

        self.n_features_: Optional[int] = None
        self.count_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.M2_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

        self.values_: list[np.ndarray] = []
        self.total_chunks_: int = 0
        self.total_samples_seen_: int = 0

    def update_stats(self, X_chunk: np.ndarray) -> "StreamingStats":
        """
       It will update statistics using one data chunk.

        
        """

        X_chunk = self._validate_X(X_chunk)

        if self.n_features_ is None:
            self._initialise(X_chunk.shape[1])

        if X_chunk.shape[1] != self.n_features_:
            raise ValueError(
                f"X_chunk has {X_chunk.shape[1]} features, "
                f"but expected {self.n_features_}."
            )

        valid_mask = ~np.isnan(X_chunk)

        chunk_count = np.sum(valid_mask, axis=0)
        safe_X = np.where(valid_mask, X_chunk, 0.0)

        chunk_sum = np.sum(safe_X, axis=0)

        chunk_mean = np.divide(
            chunk_sum,
            chunk_count,
            out=np.zeros_like(chunk_sum, dtype=float),
            where=chunk_count > 0,
        )

        centered = np.where(valid_mask, X_chunk - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered ** 2, axis=0)

        self._merge_stats(chunk_count, chunk_mean, chunk_M2)

        chunk_min = np.nanmin(
            np.where(valid_mask, X_chunk, np.nan),
            axis=0,
        )
        chunk_max = np.nanmax(
            np.where(valid_mask, X_chunk, np.nan),
            axis=0,
        )

        self.min_ = np.fmin(self.min_, chunk_min)
        self.max_ = np.fmax(self.max_, chunk_max)

        if self.store_values:
            self.values_.append(X_chunk.copy())

        self.total_chunks_ += 1
        self.total_samples_seen_ += X_chunk.shape[0]

        return self

    def _merge_stats(
        self,
        chunk_count: np.ndarray,
        chunk_mean: np.ndarray,
        chunk_M2: np.ndarray,
    ) -> None:
        """
       it will merge current statistics with chunk statistics using Welford-style update.
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

    def mean(self) -> np.ndarray:
        """
        Return running mean for each feature.
        """

        self._check_fitted()
        return self.mean_.copy()

    def variance(self, ddof: int = 0) -> np.ndarray:
        """
       It returns running variance for each feature.

        
        """

        self._check_fitted()

        if ddof < 0:
            raise ValueError("ddof must be non-negative.")

        denominator = self.count_ - ddof

        return np.divide(
            self.M2_,
            denominator,
            out=np.zeros_like(self.M2_, dtype=float),
            where=denominator > 0,
        )

    def std(self, ddof: int = 0) -> np.ndarray:
        """
        Return running standard deviation for each feature.
        """

        return np.sqrt(self.variance(ddof=ddof))

    def min(self) -> np.ndarray:
        """
        Return running minimum for each feature.
        """

        self._check_fitted()
        return self.min_.copy()

    def max(self) -> np.ndarray:
        """
        Return running maximum for each feature.
        """

        self._check_fitted()
        return self.max_.copy()

    def quantile(self, q: float) -> np.ndarray:
        """
        It returns approximate quantile for each feature.

        
        """

        self._check_fitted()

        if not self.store_values:
            raise RuntimeError(
                "quantile() requires store_values=True."
            )

        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1.")

        all_values = np.vstack(self.values_)
        return np.nanquantile(all_values, q, axis=0)

    def histogram(self) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        """
       It returns histogram counts and bin edges for each feature.

        
        """

        self._check_fitted()

        if not self.store_values:
            raise RuntimeError(
                "histogram() requires store_values=True."
            )

        all_values = np.vstack(self.values_)
        histograms = {}

        for feature_idx in range(all_values.shape[1]):
            feature_values = all_values[:, feature_idx]
            feature_values = feature_values[~np.isnan(feature_values)]

            if feature_values.size == 0:
                counts = np.zeros(self.histogram_bins, dtype=int)
                edges = np.linspace(0, 1, self.histogram_bins + 1)
            else:
                counts, edges = np.histogram(
                    feature_values,
                    bins=self.histogram_bins,
                )

            histograms[feature_idx] = (counts, edges)

        return histograms

    def result(self) -> Dict[str, np.ndarray]:
        """
        Returning all main statistics in a dictionary.
        """

        self._check_fitted()

        return {
            "count": self.count_.copy(),
            "mean": self.mean(),
            "variance": self.variance(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
        }

    def _initialise(self, n_features: int) -> None:
        """
        going to initialise arrays after seeing the first chunk
        """

        self.n_features_ = n_features
        self.count_ = np.zeros(n_features, dtype=float)
        self.mean_ = np.zeros(n_features, dtype=float)
        self.M2_ = np.zeros(n_features, dtype=float)
        self.min_ = np.full(n_features, np.inf, dtype=float)
        self.max_ = np.full(n_features, -np.inf, dtype=float)

    def _check_fitted(self) -> None:
        """
        Checking whether update_stats() has been called.
        """

        if self.n_features_ is None:
            raise RuntimeError("StreamingStats has not been updated yet.")

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
       I am going to validate input X here.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X_chunk should be a 2D array with shape "
                "(n_samples, n_features)."
            )

        if X.shape[0] == 0:
            raise ValueError("Here, X_chunk should contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError(" At least one feature should be contained by X_chunk.")

        return X