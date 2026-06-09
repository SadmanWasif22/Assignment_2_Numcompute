
"""
visualise.py

Reusable plotting utilities for NumCompute Stream.


 matplotlib and Numpy were used here. 


Required assignment plots:
- plot_metric_over_time(metric_values, title, ylabel)
- compare_models(metric1, metric2, labels)
- plot_predictions_vs_ground_truth(y_true, y_pred)

These functions can display plots inline or save them to file.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(
    metric_values: Sequence[float],
    title: str = "Metric over time",
    ylabel: str = "Metric",
    xlabel: str = "Chunk",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a metric across streaming chunks.

    Parameters
    ----------
    metric_values : sequence of float
        Metric values recorded after each chunk.

    title : str, default="Metric over time"
        Plot title.

    ylabel : str, default="Metric"
        Y-axis label.

    xlabel : str, default="Chunk"
        X-axis label.

    save_path : str or None, default=None
        If provided, save the plot to this path.

    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    values = _validate_1d_numeric(metric_values, "metric_values")
    chunks = np.arange(1, values.shape[0] + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    _save_show(fig, save_path, show)

    return fig, ax


def compare_models(
    metric1: Sequence[float],
    metric2: Sequence[float],
    labels: Sequence[str] = ("Model 1", "Model 2"),
    title: str = "Model comparison",
    ylabel: str = "Metric",
    xlabel: str = "Chunk",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare two models across streaming chunks.

    Parameters
    ----------
    metric1 : sequence of float
        Metric values for first model.

    metric2 : sequence of float
        Metric values for second model.

    labels : sequence of str, default=("Model 1", "Model 2")
        Names of the two models.

    title : str, default="Model comparison"
        Plot title.

    ylabel : str, default="Metric"
        Y-axis label.

    xlabel : str, default="Chunk"
        X-axis label.

    save_path : str or None, default=None
        If provided, save the plot to this path.

    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    values1 = _validate_1d_numeric(metric1, "metric1")
    values2 = _validate_1d_numeric(metric2, "metric2")

    if values1.shape[0] != values2.shape[0]:
        raise ValueError(
            "metric1 and metric2 must have the same length. "
            f"Got {values1.shape[0]} and {values2.shape[0]}."
        )

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two names.")

    chunks = np.arange(1, values1.shape[0] + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks, values1, marker="o", label=labels[0])
    ax.plot(chunks, values2, marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_show(fig, save_path, show)

    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true: Sequence,
    y_pred: Sequence,
    title: str = "Predictions vs ground truth",
    xlabel: str = "Sample index",
    ylabel: str = "Class label",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot predicted labels and true labels for the latest chunk.

  

    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")

    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array.")

    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must contain at least one sample.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length. "
            f"Got y_true={y_true.shape[0]} and y_pred={y_pred.shape[0]}."
        )

    index = np.arange(y_true.shape[0])

    fig, ax = plt.subplots()
    ax.plot(index, y_true, marker="o", linestyle="-", label="Ground truth")
    ax.plot(index, y_pred, marker="x", linestyle="--", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_show(fig, save_path, show)

    return fig, ax


def plot_error_over_time(
    accuracy_values: Sequence[float],
    title: str = "Error over time",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot error rate over chunks using accuracy values.

    error = 1 - accuracy
    """

    accuracy = _validate_1d_numeric(accuracy_values, "accuracy_values")

    if np.any((accuracy < 0) | (accuracy > 1)):
        raise ValueError("accuracy_values must be between 0 and 1.")

    error = 1.0 - accuracy

    return plot_metric_over_time(
        error,
        title=title,
        ylabel="Error",
        xlabel="Chunk",
        save_path=save_path,
        show=show,
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Optional[Sequence] = None,
    title: str = "Confusion matrix",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a confusion matrix using matplotlib.

    
    """

    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    if labels is None:
        labels = np.arange(matrix.shape[0])
    else:
        labels = np.asarray(labels)

    if len(labels) != matrix.shape[0]:
        raise ValueError("Number of labels must match matrix size.")

    fig, ax = plt.subplots()
    image = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)

    _save_show(fig, save_path, show)

    return fig, ax


def _validate_1d_numeric(values: Sequence[float], name: str) -> np.ndarray:
    """
    Validate 1D numeric values.
    """

    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence.")

    if values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value.")

    if np.any(np.isnan(values)):
        raise ValueError(f"{name} must not contain NaN values.")

    return values


def _save_show(
    fig: plt.Figure,
    save_path: Optional[str],
    show: bool,
) -> None:
    """
    Save and/or show a matplotlib figure.
    """

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

