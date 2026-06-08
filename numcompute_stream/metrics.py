"""
metrics.py

It is Streaming classification metrics for NumCompute Stream.

I have used Numpy and Standard Python library. 

This module is important as it supports accuracy, precision, Recall, F1 score, Confusion Matrix
and rolling window accuracy.

"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def _validate_y_true_y_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate y_true and y_pred.

    Both must be 1D arrays with the same length.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array with shape (n_samples,).")

    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array with shape (n_samples,).")

    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must contain at least one sample.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length. "
            f"Got y_true={y_true.shape[0]} and y_pred={y_pred.shape[0]}."
        )

    return y_true, y_pred


class Accuracy:
    """
    Streaming accuracy metric.

    Accuracy = correct predictions / total predictions
    """

    def __init__(self) -> None:
        self.reset()

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "Accuracy":
        """
        Update accuracy using one chunk.
        """

        y_true, y_pred = _validate_y_true_y_pred(y_true, y_pred)

        self.correct_ += int(np.sum(y_true == y_pred))
        self.total_ += int(y_true.shape[0])

        return self

    def result(self) -> float:
        """
        Return cumulative accuracy.
        """

        if self.total_ == 0:
            return 0.0

        return self.correct_ / self.total_

    def reset(self) -> None:
        """
        Reset metric.
        """

        self.correct_ = 0
        self.total_ = 0


class ConfusionMatrix:
    """
    Streaming confusion matrix.

    Parameters
    ----------
    labels : array-like or None, default=None
        Class labels. If None, labels are discovered as chunks arrive.

    Notes
    -----
    Rows represent true classes.
    Columns represent predicted classes.
    """

    def __init__(self, labels: Optional[np.ndarray] = None) -> None:
        self.initial_labels = None if labels is None else np.asarray(labels)
        self.reset()

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> "ConfusionMatrix":
        """
        Update confusion matrix using one chunk.
        """

        y_true, y_pred = _validate_y_true_y_pred(y_true, y_pred)

        new_labels = np.unique(np.concatenate((y_true, y_pred)))
        self._ensure_labels(new_labels)

        true_indices = self._label_to_index(y_true)
        pred_indices = self._label_to_index(y_pred)

        np.add.at(self.matrix_, (true_indices, pred_indices), 1)

        return self

    def result(self) -> np.ndarray:
        """
        Return confusion matrix.
        """

        return self.matrix_.copy()

    def labels(self) -> np.ndarray:
        """
        Return labels used in the matrix.
        """

        return self.labels_.copy()

    def reset(self) -> None:
        """
        Reset confusion matrix.
        """

        if self.initial_labels is None:
            self.labels_ = np.array([])
            self.matrix_ = np.zeros((0, 0), dtype=int)
        else:
            self.labels_ = np.asarray(self.initial_labels)
            n_labels = len(self.labels_)
            self.matrix_ = np.zeros((n_labels, n_labels), dtype=int)

    def _ensure_labels(self, new_labels: np.ndarray) -> None:
        """
        Expand matrix if unseen labels appear.
        """

        missing = np.setdiff1d(new_labels, self.labels_)

        if missing.size == 0:
            return

        old_labels = self.labels_
        old_matrix = self.matrix_

        self.labels_ = np.sort(np.concatenate((old_labels, missing)))
        n_labels = len(self.labels_)

        new_matrix = np.zeros((n_labels, n_labels), dtype=int)

        if old_labels.size > 0:
            old_positions = np.searchsorted(self.labels_, old_labels)
            new_matrix[np.ix_(old_positions, old_positions)] = old_matrix

        self.matrix_ = new_matrix

    def _label_to_index(self, y: np.ndarray) -> np.ndarray:
        """
        Convert labels to confusion matrix indices.
        """

        return np.searchsorted(self.labels_, y)


class Precision:
    """
    Streaming precision metric.

    Supports:
    - binary average
    - macro average

    Parameters
    ----------
    average : {"binary", "macro"}, default="binary"
        Averaging method.

    positive_label : int or float, default=1
        Positive class for binary precision.

    zero_division : float, default=0.0
        Value returned when division by zero occurs.
    """

    def __init__(
        self,
        average: str = "binary",
        positive_label=1,
        zero_division: float = 0.0,
    ) -> None:
        if average not in {"binary", "macro"}:
            raise ValueError("average must be either 'binary' or 'macro'.")

        self.average = average
        self.positive_label = positive_label
        self.zero_division = zero_division
        self.cm = ConfusionMatrix()

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "Precision":
        """
        Update precision using one chunk.
        """

        self.cm.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """
        Return cumulative precision.
        """

        matrix = self.cm.result()
        labels = self.cm.labels()

        if matrix.size == 0:
            return 0.0

        if self.average == "binary":
            if self.positive_label not in labels:
                return self.zero_division

            idx = np.where(labels == self.positive_label)[0][0]
            tp = matrix[idx, idx]
            fp = np.sum(matrix[:, idx]) - tp

            denominator = tp + fp

            if denominator == 0:
                return self.zero_division

            return float(tp / denominator)

        tp = np.diag(matrix)
        fp = np.sum(matrix, axis=0) - tp
        denominator = tp + fp

        precision_per_class = np.divide(
            tp,
            denominator,
            out=np.full_like(tp, self.zero_division, dtype=float),
            where=denominator > 0,
        )

        return float(np.mean(precision_per_class))

    def reset(self) -> None:
        """
        Reset precision.
        """

        self.cm.reset()


class Recall:
    """
    Streaming recall metric.

    Supports:
    - binary average
    - macro average
    """

    def __init__(
        self,
        average: str = "binary",
        positive_label=1,
        zero_division: float = 0.0,
    ) -> None:
        if average not in {"binary", "macro"}:
            raise ValueError("average must be either 'binary' or 'macro'.")

        self.average = average
        self.positive_label = positive_label
        self.zero_division = zero_division
        self.cm = ConfusionMatrix()

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "Recall":
        """
        Update recall using one chunk.
        """

        self.cm.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """
        Return cumulative recall.
        """

        matrix = self.cm.result()
        labels = self.cm.labels()

        if matrix.size == 0:
            return 0.0

        if self.average == "binary":
            if self.positive_label not in labels:
                return self.zero_division

            idx = np.where(labels == self.positive_label)[0][0]
            tp = matrix[idx, idx]
            fn = np.sum(matrix[idx, :]) - tp

            denominator = tp + fn

            if denominator == 0:
                return self.zero_division

            return float(tp / denominator)

        tp = np.diag(matrix)
        fn = np.sum(matrix, axis=1) - tp
        denominator = tp + fn

        recall_per_class = np.divide(
            tp,
            denominator,
            out=np.full_like(tp, self.zero_division, dtype=float),
            where=denominator > 0,
        )

        return float(np.mean(recall_per_class))

    def reset(self) -> None:
        """
        Reset recall.
        """

        self.cm.reset()


class F1Score:
    """
    Streaming F1 score.

    F1 = 2 * precision * recall / (precision + recall)
    """

    def __init__(
        self,
        average: str = "binary",
        positive_label=1,
        zero_division: float = 0.0,
    ) -> None:
        self.precision = Precision(
            average=average,
            positive_label=positive_label,
            zero_division=zero_division,
        )
        self.recall = Recall(
            average=average,
            positive_label=positive_label,
            zero_division=zero_division,
        )
        self.zero_division = zero_division

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "F1Score":
        """
        Update F1 score using one chunk.
        """

        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """
        Return cumulative F1 score.
        """

        p = self.precision.result()
        r = self.recall.result()

        denominator = p + r

        if denominator == 0:
            return self.zero_division

        return float(2 * p * r / denominator)

    def reset(self) -> None:
        """
        Reset F1 score.
        """

        self.precision.reset()
        self.recall.reset()


class RollingAccuracy:
    """
    Rolling-window streaming accuracy.

    Parameters
    ----------
    window_size : int, default=100
        Number of most recent predictions used.
    """

    def __init__(self, window_size: int = 100) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")

        self.window_size = window_size
        self.reset()

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "RollingAccuracy":
        """
        Update rolling accuracy using one chunk.
        """

        y_true, y_pred = _validate_y_true_y_pred(y_true, y_pred)

        correct_values = (y_true == y_pred).astype(int)

        for value in correct_values:
            self.window_.append(int(value))

        return self

    def result(self) -> float:
        """
        Return rolling accuracy.
        """

        if len(self.window_) == 0:
            return 0.0

        return float(np.mean(np.array(self.window_)))

    def reset(self) -> None:
        """
        Reset rolling accuracy.
        """

        self.window_ = deque(maxlen=self.window_size)