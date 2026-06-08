
import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


def test_tree_fit_predict_shape():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape


def test_tree_fit_predict_values_binary():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    pred = model.predict(X)

    assert np.mean(pred == y) >= 0.75


def test_tree_partial_fit_two_chunks():
    X1 = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    y1 = np.array([0, 0])

    X2 = np.array([
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y2 = np.array([1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.partial_fit(X1, y1)
    model.partial_fit(X2, y2)

    X_all = np.vstack((X1, X2))
    pred = model.predict(X_all)

    assert pred.shape == (4,)


def test_tree_gini_criterion():
    X = np.array([
        [1.0],
        [2.0],
        [5.0],
        [6.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2, criterion="gini")
    model.fit(X, y)

    pred = model.predict(X)

    assert np.mean(pred == y) >= 0.75


def test_tree_entropy_criterion():
    X = np.array([
        [1.0],
        [2.0],
        [5.0],
        [6.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    model.fit(X, y)

    pred = model.predict(X)

    assert np.mean(pred == y) >= 0.75


def test_tree_predict_before_fit_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    model = DecisionTreeClassifier(max_depth=2)

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_tree_invalid_criterion_error():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="wrong")


def test_tree_invalid_max_depth_error():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=0)


def test_tree_invalid_min_samples_split_error():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)


def test_tree_shape_mismatch_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    y = np.array([0])

    model = DecisionTreeClassifier(max_depth=2)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_tree_predict_wrong_feature_count_error():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    X_wrong = np.array([
        [1.0, 2.0, 3.0],
    ])

    with pytest.raises(ValueError):
        model.predict(X_wrong)


def test_tree_handles_nan_values():
    X = np.array([
        [1.0, 2.0],
        [1.0, np.nan],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape