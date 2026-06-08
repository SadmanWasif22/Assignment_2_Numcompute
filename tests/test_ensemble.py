
import numpy as np
import pytest

from numcompute_stream.ensemble import EnsembleClassifier


def test_ensemble_fit_predict_shape():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=42,
    )
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape


def test_ensemble_fit_predict_values():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=5,
        max_depth=2,
        random_state=42,
    )
    model.fit(X, y)

    pred = model.predict(X)

    assert np.mean(pred == y) >= 0.75


def test_ensemble_partial_fit_two_chunks():
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

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=42,
    )

    model.partial_fit(X1, y1)
    model.partial_fit(X2, y2)

    X_all = np.vstack((X1, X2))
    pred = model.predict(X_all)

    assert pred.shape == (4,)


def test_ensemble_predict_proba_shape():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=42,
    )
    model.fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape == (4, 2)


def test_ensemble_predict_proba_rows_sum_to_one():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=5,
        max_depth=2,
        random_state=42,
    )
    model.fit(X, y)

    proba = model.predict_proba(X)

    assert np.allclose(np.sum(proba, axis=1), np.ones(X.shape[0]))


def test_ensemble_invalid_n_estimators_error():
    with pytest.raises(ValueError):
        EnsembleClassifier(n_estimators=0)


def test_ensemble_invalid_max_depth_error():
    with pytest.raises(ValueError):
        EnsembleClassifier(max_depth=0)


def test_ensemble_invalid_min_samples_split_error():
    with pytest.raises(ValueError):
        EnsembleClassifier(min_samples_split=1)


def test_ensemble_invalid_criterion_error():
    with pytest.raises(ValueError):
        EnsembleClassifier(criterion="wrong")


def test_ensemble_predict_before_fit_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    model = EnsembleClassifier(n_estimators=3)

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_ensemble_shape_mismatch_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    y = np.array([0])

    model = EnsembleClassifier(n_estimators=3)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_ensemble_predict_wrong_feature_count_error():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(n_estimators=3, random_state=42)
    model.fit(X, y)

    X_wrong = np.array([
        [1.0, 2.0, 3.0],
    ])

    with pytest.raises(ValueError):
        model.predict(X_wrong)


def test_ensemble_without_bootstrap():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        bootstrap=False,
        random_state=42,
    )
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape