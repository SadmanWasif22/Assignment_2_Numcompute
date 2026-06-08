
import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier


def test_pipeline_partial_fit_predict_tree():
    X = np.array([
        [1.0, 2.0],
        [1.0, np.nan],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    pipe.partial_fit(X, y)
    pred = pipe.predict(X)

    assert pred.shape == y.shape


def test_pipeline_partial_fit_predict_ensemble():
    X = np.array([
        [1.0, 2.0],
        [1.0, np.nan],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", EnsembleClassifier(n_estimators=3, max_depth=2, random_state=42)),
    ])

    pipe.partial_fit(X, y)
    pred = pipe.predict(X)

    assert pred.shape == y.shape


def test_pipeline_fit_predict():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    pred = pipe.fit_predict(X, y)

    assert pred.shape == y.shape


def test_pipeline_transform_shape():
    X = np.array([
        [1.0, 2.0],
        [1.0, np.nan],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    pipe.partial_fit(X, y)
    X_transformed = pipe.transform(X)

    assert X_transformed.shape == X.shape


def test_pipeline_get_step():
    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    step = pipe.get_step("scaler")

    assert isinstance(step, StandardScaler)


def test_pipeline_set_step():
    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    new_scaler = StandardScaler()
    pipe.set_step("scaler", new_scaler)

    assert pipe.get_step("scaler") is new_scaler


def test_pipeline_empty_steps_error():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_steps_must_be_list_error():
    with pytest.raises(TypeError):
        Pipeline(("model", DecisionTreeClassifier()))


def test_pipeline_duplicate_names_error():
    with pytest.raises(ValueError):
        Pipeline([
            ("step", SimpleImputer()),
            ("step", StandardScaler()),
            ("model", DecisionTreeClassifier()),
        ])


def test_pipeline_missing_step_error():
    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    with pytest.raises(KeyError):
        pipe.get_step("unknown")


def test_pipeline_shape_mismatch_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    y = np.array([0])

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    with pytest.raises(ValueError):
        pipe.partial_fit(X, y)


def test_pipeline_predict_before_fit_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    with pytest.raises(RuntimeError):
        pipe.predict(X)