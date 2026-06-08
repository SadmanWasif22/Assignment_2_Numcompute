"""
ensemble.py

 This is Tree-based ensemble classifier for NumCompute Stream.

Allowed libraries here are numpy and standard python libraries. I have used them.


"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np

from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """
    Bagging / Random Forest style classifier using multiple decision trees.

    Parameters
    ----------
    n_estimators : int, default=5
        Number of decision trees.

    max_depth : int, default=5
        Maximum depth of each tree.

    min_samples_split : int, default=2
        Minimum number of samples required to split each tree node.

    criterion : {"gini", "entropy"}, default="gini"
        Split criterion used by each tree.

    max_features : int, float, {"sqrt", "log2"} or None, default="sqrt"
        Number of features considered at each split.

    bootstrap : bool, default=True
        Whether each tree trains on a bootstrap sample.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Notes
    -----
    partial_fit() appends each incoming chunk to all previously seen data
    and retrains the ensemble. This is simple and streaming-compatible.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: Optional[Any] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0.")

        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.rng_ = np.random.default_rng(random_state)

        self.trees_: list[DecisionTreeClassifier] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

        self._X_seen: Optional[np.ndarray] = None
        self._y_seen: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        """
        Fit the ensemble from scratch.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features).

        y : np.ndarray
            Target vector with shape (n_samples,).

        Returns
        -------
        self : EnsembleClassifier
        """

        X, y = self._validate_X_y(X, y)

        self._X_seen = X.copy()
        self._y_seen = y.copy()

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

        self._fit_trees(X, y)

        return self

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> "EnsembleClassifier":
        """
        Incrementally update the ensemble with a new chunk.

        Parameters
        ----------
        X_chunk : np.ndarray
            Feature chunk with shape (n_samples, n_features).

        y_chunk : np.ndarray
            Target chunk with shape (n_samples,).

        Returns
        -------
        self : EnsembleClassifier
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

        self._fit_trees(self._X_seen, self._y_seen)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using majority voting.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels with shape (n_samples,).
        """

        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        all_predictions = np.array([tree.predict(X) for tree in self.trees_])

        return self._majority_vote(all_predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate class probabilities from tree votes.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features).

        Returns
        -------
        probabilities : np.ndarray
            Array with shape (n_samples, n_classes).
        """

        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features_}."
            )

        all_predictions = np.array([tree.predict(X) for tree in self.trees_])
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes_)), dtype=float)

        for class_index, class_label in enumerate(self.classes_):
            probabilities[:, class_index] = np.mean(
                all_predictions == class_label,
                axis=0,
            )

        return probabilities

    def _fit_trees(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit all decision trees.
        """

        self.trees_ = []

        n_samples = X.shape[0]

        for tree_index in range(self.n_estimators):
            tree_seed = None
            if self.random_state is not None:
                tree_seed = self.random_state + tree_index

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            if self.bootstrap:
                sample_indices = self.rng_.integers(
                    low=0,
                    high=n_samples,
                    size=n_samples,
                )
            else:
                sample_indices = np.arange(n_samples)

            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

    def _majority_vote(self, all_predictions: np.ndarray) -> np.ndarray:
        """
        Majority vote across trees.

        Parameters
        ----------
        all_predictions : np.ndarray
            Shape (n_estimators, n_samples)

        Returns
        -------
        final_predictions : np.ndarray
            Shape (n_samples,)
        """

        if all_predictions.ndim != 2:
            raise ValueError(
                "all_predictions must have shape (n_estimators, n_samples)."
            )

        n_samples = all_predictions.shape[1]
        final_predictions = []

        for sample_index in range(n_samples):
            sample_votes = all_predictions[:, sample_index]
            labels, counts = np.unique(sample_votes, return_counts=True)

            # Deterministic tie resolution:
            # np.unique sorts labels, and np.argmax chooses first maximum.
            final_predictions.append(labels[np.argmax(counts)])

        return np.asarray(final_predictions)

    def _check_fitted(self) -> None:
        """
        Check whether ensemble has been fitted.
        """

        if len(self.trees_) == 0:
            raise RuntimeError("EnsembleClassifier has not been fitted yet.")

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