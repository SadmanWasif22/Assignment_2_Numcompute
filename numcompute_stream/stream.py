
"""
stream.py

Streaming training utilities for the NumCompute Stream assignment.

This module is being used for managing chunk-wise learning, prediction, metric updates,
and simple logging for streaming machine learning experiments.


 Python standard library and Numpy were used here



"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class StreamTrainer:
    """
    Managing chunk-wise training, scoring, logging, and metric tracking here.

    

        

    Notes:
    I have designed this for streaming learning. It trains one chunk
    at a time. It stores metric values after each chunk.
    """

    def __init__(
        self,
        pipeline: Any,
        metrics: Optional[Dict[str, Any]] = None,
        name: str = "stream_model",
        log_memory: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = metrics if metrics is not None else {}
        self.name = name
        self.log_memory = log_memory

        self.chunk_index = 0
        self.samples_seen = 0

        self.logs: List[Dict[str, Any]] = []
        self.metric_history: Dict[str, List[float]] = {
            metric_name: [] for metric_name in self.metrics
        }

        self._validate_pipeline()

    def _validate_pipeline(self) -> None:
        """
        Check that the pipeline/model has required methods.
        """

        if not hasattr(self.pipeline, "partial_fit"):
            raise TypeError("pipeline must have a partial_fit(X, y) method.")

        if not hasattr(self.pipeline, "predict"):
            raise TypeError("pipeline must have a predict(X) method.")

    def fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Training the pipeline on one data chunk.

        
        """

        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        start_time = time.perf_counter()

        self.pipeline.partial_fit(X_chunk, y_chunk)

        train_time = time.perf_counter() - start_time

        self.chunk_index += 1
        self.samples_seen += X_chunk.shape[0]

        log = {
            "model": self.name,
            "chunk": self.chunk_index,
            "samples_in_chunk": X_chunk.shape[0],
            "samples_seen": self.samples_seen,
            "train_time_seconds": train_time,
        }

        if self.log_memory:
            log["memory_bytes"] = self._estimate_memory(X_chunk, y_chunk)

        self.logs.append(log)

        return log

    def score_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, float]:
        """
        Going to predict and update metrics for one data chunk.

        
        
        """

        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        y_pred = self.pipeline.predict(X_chunk)

        y_pred = np.asarray(y_pred)

        if y_pred.ndim != 1:
            raise ValueError("Predictions must be a 1D array.")

        if y_pred.shape[0] != y_chunk.shape[0]:
            raise ValueError(
                "Predictions and y_chunk must have the same length. "
                f"Got predictions={y_pred.shape[0]} and y_chunk={y_chunk.shape[0]}."
            )

        scores = {}

        for metric_name, metric in self.metrics.items():
            if not hasattr(metric, "update"):
                raise TypeError(f"Metric '{metric_name}' must have update().")

            if not hasattr(metric, "result"):
                raise TypeError(f"Metric '{metric_name}' must have result().")

            metric.update(y_chunk, y_pred)
            value = metric.result()

            scores[metric_name] = float(value)
            self.metric_history[metric_name].append(float(value))

        if len(self.logs) > 0:
            self.logs[-1].update(scores)

        return scores

    def fit_score_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
        score_before_fit: bool = False,
    ) -> Dict[str, Any]:
        """
        Training and scoring one chunk.


        """

        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        if score_before_fit:
            scores = self.score_chunk(X_chunk, y_chunk)
            log = self.fit_chunk(X_chunk, y_chunk)
            log.update(scores)
            return log

        log = self.fit_chunk(X_chunk, y_chunk)
        scores = self.score_chunk(X_chunk, y_chunk)
        log.update(scores)

        return log

    def fit_stream(
        self,
        stream: Iterable[Tuple[np.ndarray, np.ndarray]],
        score_before_fit: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Training over many chunks from a stream here.

        
        """

        for X_chunk, y_chunk in stream:
            self.fit_score_chunk(
                X_chunk,
                y_chunk,
                score_before_fit=score_before_fit,
            )

        return self.get_logs()

    def predict_chunk(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Predicting labels for one chunk.

        
        """

        X_chunk = self._validate_X(X_chunk)
        y_pred = self.pipeline.predict(X_chunk)
        return np.asarray(y_pred)

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Returns training logs.

        """

        return list(self.logs)

    def get_metric_history(self, metric_name: Optional[str] = None):
        """
        Returns metric history.

      
        """

        if metric_name is None:
            return {
                name: list(values)
                for name, values in self.metric_history.items()
            }

        if metric_name not in self.metric_history:
            raise KeyError(f"Unknown metric name: {metric_name}")

        return list(self.metric_history[metric_name])

    def reset(self) -> None:
        """
        Reset logs, counters, and metric objects.
        """

        self.chunk_index = 0
        self.samples_seen = 0
        self.logs = []

        for metric_name, metric in self.metrics.items():
            if hasattr(metric, "reset"):
                metric.reset()

            self.metric_history[metric_name] = []

    def summary(self) -> Dict[str, Any]:
        """
        Return a simple summary of the streaming run.

        
        """

        latest_metrics = {}

        for metric_name, history in self.metric_history.items():
            if len(history) > 0:
                latest_metrics[metric_name] = history[-1]
            else:
                latest_metrics[metric_name] = None

        return {
            "model": self.name,
            "chunks_processed": self.chunk_index,
            "samples_seen": self.samples_seen,
            "latest_metrics": latest_metrics,
        }

    def print_logs(self) -> None:
        """
        Print logs in a simple readable format.
        """

        if len(self.logs) == 0:
            print("No logs available.")
            return

        for log in self.logs:
            print(log)

    @staticmethod
    def _estimate_memory(X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """
        Estimate memory footprint of one chunk.

        Parameters:
        
        X : np.ndarray
            Feature chunk.

        y : np.ndarray or None
            Target chunk.

        """

        memory = sys.getsizeof(X) + X.nbytes

        if y is not None:
            memory += sys.getsizeof(y) + y.nbytes

        return int(memory)

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
        Validating feature matrix.

        X should be 2D:
        (n_samples, n_features)
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
        Validating target vector.

      
        """

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array with shape (n_samples,).")

        if y.shape[0] == 0:
            raise ValueError("y must contain at least one sample.")

        return y

    @classmethod
    def _validate_X_y(
        cls,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate X and y together.
        """

        X = cls._validate_X(X)
        y = cls._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples. "
                f"Got X={X.shape[0]} and y={y.shape[0]}."
            )

        return X, y

