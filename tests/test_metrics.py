import numpy as np
import pytest

from numcompute_stream.metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    RollingAccuracy,
)


def test_accuracy_single_chunk():
    metric = Accuracy()

    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    metric.update(y_true, y_pred)

    assert metric.result() == 0.75


def test_accuracy_multiple_chunks():
    metric = Accuracy()

    metric.update(np.array([1, 0]), np.array([1, 0]))
    metric.update(np.array([1, 1]), np.array([0, 1]))

    assert metric.result() == 0.75


def test_accuracy_reset():
    metric = Accuracy()

    metric.update(np.array([1, 0]), np.array([1, 0]))
    metric.reset()

    assert metric.result() == 0.0


def test_accuracy_shape_mismatch():
    metric = Accuracy()

    with pytest.raises(ValueError):
        metric.update(np.array([1, 0]), np.array([1]))


def test_confusion_matrix_binary():
    cm = ConfusionMatrix()

    cm.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))

    result = cm.result()

    assert result.shape == (2, 2)
    assert np.sum(result) == 4


def test_confusion_matrix_labels():
    cm = ConfusionMatrix(labels=np.array([0, 1]))

    cm.update(np.array([0, 1]), np.array([1, 1]))

    assert np.array_equal(cm.labels(), np.array([0, 1]))


def test_precision_binary():
    metric = Precision(average="binary", positive_label=1)

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    metric.update(y_true, y_pred)

    assert metric.result() == 0.5


def test_recall_binary():
    metric = Recall(average="binary", positive_label=1)

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    metric.update(y_true, y_pred)

    assert metric.result() == 0.5


def test_f1_score_binary():
    metric = F1Score(average="binary", positive_label=1)

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    metric.update(y_true, y_pred)

    assert metric.result() == 0.5


def test_precision_macro():
    metric = Precision(average="macro")

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metric.update(y_true, y_pred)

    assert 0.0 <= metric.result() <= 1.0


def test_recall_macro():
    metric = Recall(average="macro")

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metric.update(y_true, y_pred)

    assert 0.0 <= metric.result() <= 1.0


def test_rolling_accuracy():
    metric = RollingAccuracy(window_size=3)

    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 0])

    metric.update(y_true, y_pred)

    assert metric.result() == 2 / 3


def test_rolling_accuracy_reset():
    metric = RollingAccuracy(window_size=3)

    metric.update(np.array([1, 0]), np.array([1, 0]))
    metric.reset()

    assert metric.result() == 0.0


def test_invalid_average_precision():
    with pytest.raises(ValueError):
        Precision(average="wrong")


def test_invalid_average_recall():
    with pytest.raises(ValueError):
        Recall(average="wrong")


def test_empty_metric_input():
    metric = Accuracy()

    with pytest.raises(ValueError):
        metric.update(np.array([]), np.array([]))