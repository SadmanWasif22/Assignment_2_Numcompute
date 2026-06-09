

"""
io.py

Custom input/output utilities for the NumCompute streaming assignment.

I did not use pandas and scikit-learn in this module. I have used plain python and Numpy.


Main features of this module are given below:
1. loading CSV files
2.spliting data into train/test sets
3. spliting arrays into chunks for streaming learning
4. validating input shapes
These works were done here. 
"""

from __future__ import annotations

import csv
from typing import Generator, Optional, Tuple

import numpy as np


def load_csv(
    file_path: str,
    target_column: int = -1,
    has_header: bool = True,
    delimiter: str = ",",
    dtype: type = float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loading a CSV file using only Python csv and NumPy here

    
  


    
    """

    rows = []

    try:
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=delimiter)

            if has_header:
                next(reader, None)

            for row in reader:
                if len(row) == 0:
                    continue

                cleaned_row = []
                for value in row:
                    value = value.strip()

                    if value == "":
                        cleaned_row.append(np.nan)
                    else:
                        cleaned_row.append(dtype(value))

                rows.append(cleaned_row)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {file_path}") from exc

    if len(rows) == 0:
        raise ValueError("CSV file is empty or contains no valid data rows.")

    row_lengths = np.array([len(row) for row in rows])

    if not np.all(row_lengths == row_lengths[0]):
        raise ValueError("CSV rows have inconsistent numbers of columns.")

    data = np.array(rows, dtype=float)

    if data.ndim != 2:
        raise ValueError("Loaded CSV data must be a 2D array.")

    n_columns = data.shape[1]

    if target_column < 0:
        target_column = n_columns + target_column

    if target_column < 0 or target_column >= n_columns:
        raise ValueError(
            f"target_column must be between 0 and {n_columns - 1}, "
            f"but got {target_column}."
        )

    y = data[:, target_column]
    X = np.delete(data, target_column, axis=1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    spliting arrays into training and testing sets.

    

    """

    X, y = _validate_X_y(X, y)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_count = int(np.ceil(n_samples * test_size))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    if len(train_indices) == 0:
        raise ValueError("Training set is empty. Use a smaller test_size.")

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def make_chunks(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    chunk_size: int = 32,
) -> Generator:
    """
    Spliting data into chunks to simulate a streaming learning scenario.

   
    
    """

    X = _validate_X(X)

    if chunk_size <= 0:
        raise ValueError("chunk_size should be a positive integer.")

    if y is not None:
        X, y = _validate_X_y(X, y)

    n_samples = X.shape[0]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        if y is None:
            yield X[start:end]
        else:
            yield X[start:end], y[start:end]


def save_csv(
    file_path: str,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    header: Optional[list[str]] = None,
    delimiter: str = ",",
) -> None:
    """
    Saving NumPy arrays to a CSV file here.

   

    """

    X = _validate_X(X)

    if y is not None:
        X, y = _validate_X_y(X, y)
        data = np.column_stack((X, y))
    else:
        data = X

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=delimiter)

        if header is not None:
            if len(header) != data.shape[1]:
                raise ValueError(
                    f"Header length must match number of columns. "
                    f"Expected {data.shape[1]}, got {len(header)}."
                )
            writer.writerow(header)

        writer.writerows(data.tolist())


def _validate_X(X: np.ndarray) -> np.ndarray:
    """
    Validating feature matrix X.

    
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


def _validate_y(y: np.ndarray) -> np.ndarray:
    """
   I am Validating target array y.

   
    """

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array with shape (n_samples,).")

    if y.shape[0] == 0:
        raise ValueError("y must contain at least one sample.")

    return y


def _validate_X_y(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validating X and y together.

  
    """

    X = _validate_X(X)
    y = _validate_y(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X={X.shape[0]} and y={y.shape[0]}."
        )

    return X, y

