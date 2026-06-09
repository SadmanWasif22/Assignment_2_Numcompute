"""
tree.py

This is Decision tree classifier for NumCompute Stream.

I used Numpy and Python standard library in tree.py


This module supports: 
- Gini or entropy splitting
- max_depth
- min_samples_split
- max_features
- partial_fit() for streaming-style updates
- NaN-safe feature handling
- deterministic tie resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np


@dataclass
class _Node:
    """
    Internal decision tree node.
    """

    is_leaf: bool
    prediction: Any
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier.

    

    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: Optional[Any] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[_Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

        self._X_seen: Optional[np.ndarray] = None
        self._y_seen: Optional[np.ndarray] = None

        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Fit the decision tree from scratch.

       
        """

        X, y = self._validate_X_y(X, y)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

        self._X_seen = X.copy()
        self._y_seen = y.copy()

        self.root_ = self._build_tree(X, y, depth=0)

        return self

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> "DecisionTreeClassifier":
        """
        Incrementally update the tree with a new chunk.

       
       
        """

        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        if self._X_seen is None:
            return self.fit(X_chunk, y_chunk)

        if X_chunk.shape[1] != self.n_features_:
            raise ValueError(
                f"X_chunk has {X_chunk.shape[1]} features, "
                f"but expected {self.n_features_}."
            )

        self._X_seen = np.vstack((self._X_seen, X_chunk))
        self._y_seen = np.concatenate((self._y_seen, y_chunk))

        self.classes_ = np.unique(self._y_seen)
        self.root_ = self._build_tree(self._X_seen, self._y_seen, depth=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

      
        """

        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        predictions = np.array([self._predict_one(row, self.root_) for row in X])
        return predictions

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """
        Recursively build the decision tree.
        """

        prediction = self._majority_class(y)

        if self._should_stop(X, y, depth):
            return _Node(is_leaf=True, prediction=prediction)

        feature_index, threshold, gain = self._best_split(X, y)

        if feature_index is None or threshold is None or gain <= 0:
            return _Node(is_leaf=True, prediction=prediction)

        feature_values = X[:, feature_index]

        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold

        # Send NaN values to the larger side for stable handling
        nan_mask = np.isnan(feature_values)
        if np.any(nan_mask):
            if np.sum(left_mask) >= np.sum(right_mask):
                left_mask = left_mask | nan_mask
                right_mask = right_mask & ~nan_mask
            else:
                right_mask = right_mask | nan_mask
                left_mask = left_mask & ~nan_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return _Node(is_leaf=True, prediction=prediction)

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            is_leaf=False,
            prediction=prediction,
            feature_index=feature_index,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def _should_stop(self, X: np.ndarray, y: np.ndarray, depth: int) -> bool:
        """
        Decide whether to stop splitting.
        """

        if depth >= self.max_depth:
            return True

        if X.shape[0] < self.min_samples_split:
            return True

        if np.unique(y).size == 1:
            return True

        return False

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the best feature and threshold.
        """

        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        feature_indices = self._choose_feature_indices(X.shape[1])

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            non_nan_values = feature_values[~np.isnan(feature_values)]

            if non_nan_values.size == 0:
                continue

            thresholds = np.unique(non_nan_values)

            if thresholds.size <= 1:
                continue

            # Use midpoints between sorted unique values
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                nan_mask = np.isnan(feature_values)
                if np.any(nan_mask):
                    if np.sum(left_mask) >= np.sum(right_mask):
                        left_mask = left_mask | nan_mask
                        right_mask = right_mask & ~nan_mask
                    else:
                        right_mask = right_mask | nan_mask
                        left_mask = left_mask & ~nan_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left == 0 or n_right == 0:
                    continue

                gain = self._information_gain(
                    y,
                    y[left_mask],
                    y[right_mask],
                    parent_impurity,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _information_gain(
        self,
        y_parent: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray,
        parent_impurity: float,
    ) -> float:
        """
        Compute information gain from a split.
        """

        n = y_parent.shape[0]
        n_left = y_left.shape[0]
        n_right = y_right.shape[0]

        left_weight = n_left / n
        right_weight = n_right / n

        child_impurity = (
            left_weight * self._impurity(y_left)
            + right_weight * self._impurity(y_right)
        )

        return parent_impurity - child_impurity

    def _impurity(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity or entropy.
        """

        if y.size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities ** 2))

        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _majority_class(self, y: np.ndarray):
        """
        Return majority class with deterministic tie resolution.

        If there is a tie, np.unique returns sorted labels, so np.argmax will
        choose the smallest sorted label among tied classes.
        """

        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _choose_feature_indices(self, n_features: int) -> np.ndarray:
        """
        Choose feature indices according to max_features.
        """

        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, int):
            count = self.max_features
        elif isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("float max_features must be in (0, 1].")
            count = int(np.ceil(self.max_features * n_features))
        elif self.max_features == "sqrt":
            count = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            count = int(np.log2(n_features))
        else:
            raise ValueError(
                "max_features must be None, int, float, 'sqrt', or 'log2'."
            )

        count = max(1, min(count, n_features))

        return self.rng_.choice(n_features, size=count, replace=False)

    def _predict_one(self, row: np.ndarray, node: _Node):
        """
        Predict one sample.
        """

        while not node.is_leaf:
            value = row[node.feature_index]

            # If value is NaN, use the node prediction directly.
            # This avoids unstable branching on missing values.
            if np.isnan(value):
                return node.prediction

            if value <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def _check_fitted(self) -> None:
        """
        Check that the tree has been fitted.
        """

        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier has not been fitted yet.")

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
        Validate feature matrix.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        return X

    @staticmethod
    def _validate_y(y: np.ndarray) -> np.ndarray:
        """
        Validate target vector.
        """

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array with shape (n_samples,).")

        if y.shape[0] == 0:
            raise ValueError("y must contain at least one sample.")

        return y

    @classmethod
    def _validate_X_y(cls, X: np.ndarray, y: np.ndarray):
        """
        Validate X and y together.
        """

        X = cls._validate_X(X)
        y = cls._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must contain the same number of samples. "
                f"Got X={X.shape[0]} and y={y.shape[0]}."
            )

        return X, y