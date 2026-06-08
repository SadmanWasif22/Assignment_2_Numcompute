"""
benchmark_vectorisation.py

Benchmark Python loop vs NumPy vectorised calculation.


python benchmark/benchmark_vectorisation.py
"""

from __future__ import annotations

import time

import numpy as np


def loop_column_mean(X: np.ndarray) -> list[float]:
    """
    Calculate column means using Python loops.
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]

    means = []

    for feature_index in range(n_features):
        total = 0.0

        for sample_index in range(n_samples):
            total += X[sample_index, feature_index]

        means.append(total / n_samples)

    return means


def vectorised_column_mean(X: np.ndarray) -> np.ndarray:
    """
    Calculate column means using NumPy vectorisation.
    """

    return np.mean(X, axis=0)


def loop_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean squared error using Python loops.
    """

    total = 0.0

    for i in range(len(y_true)):
        error = y_true[i] - y_pred[i]
        total += error * error

    return total / len(y_true)


def vectorised_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean squared error using NumPy vectorisation.
    """

    return float(np.mean((y_true - y_pred) ** 2))


def time_function(function, *args) -> tuple[object, float]:
    """
    Measure execution time of a function.
    """

    start = time.perf_counter()
    result = function(*args)
    end = time.perf_counter()

    return result, end - start


def main() -> None:
    """
    Run vectorisation benchmark.
    """

    rng = np.random.default_rng(42)

    X = rng.normal(size=(10000, 20))
    y_true = rng.normal(size=10000)
    y_pred = y_true + rng.normal(scale=0.1, size=10000)

    loop_mean_result, loop_mean_time = time_function(loop_column_mean, X)
    vector_mean_result, vector_mean_time = time_function(vectorised_column_mean, X)

    loop_mse_result, loop_mse_time = time_function(loop_squared_error, y_true, y_pred)
    vector_mse_result, vector_mse_time = time_function(vectorised_squared_error, y_true, y_pred)

    print("\nLoop vs Vectorised Benchmark")
    print("=" * 78)
    print(f"{'Task':<25} {'Python Loop (s)':<20} {'NumPy Vectorised (s)':<22}")
    print("-" * 78)
    print(f"{'Column mean':<25} {loop_mean_time:<20.6f} {vector_mean_time:<22.6f}")
    print(f"{'Mean squared error':<25} {loop_mse_time:<20.6f} {vector_mse_time:<22.6f}")
    print("=" * 78)

    print("\nCorrectness checks:")
    print("Column means close:", np.allclose(loop_mean_result, vector_mean_result))
    print("MSE close:", np.allclose(loop_mse_result, vector_mse_result))

    print("\nInterpretation:")
    print("- NumPy vectorised operations are expected to be faster than Python loops.")
    print("- This supports the assignment requirement for vectorised core logic.")


if __name__ == "__main__":
    main()