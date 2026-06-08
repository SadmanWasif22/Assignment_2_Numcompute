import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    SimpleImputer,
    StandardScaler,
    OneHotEncoder,
)


def test_simple_imputer_mean():
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
    ])

    imputer = SimpleImputer(strategy="mean")
    X_out = imputer.fit_transform(X)

    assert not np.any(np.isnan(X_out))
    assert X_out[0, 1] == 4.0


def test_simple_imputer_constant():
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
    ])

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_out = imputer.fit_transform(X)

    assert X_out[0, 1] == 0.0


def test_simple_imputer_median():
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [5.0, 8.0],
    ])

    imputer = SimpleImputer(strategy="median")
    X_out = imputer.fit_transform(X)

    assert not np.any(np.isnan(X_out))
    assert X_out[0, 1] == 6.0


def test_simple_imputer_partial_fit():
    imputer = SimpleImputer(strategy="mean")

    X1 = np.array([[1.0, np.nan]])
    X2 = np.array([[3.0, 4.0]])

    imputer.partial_fit(X1)
    imputer.partial_fit(X2)

    X_out = imputer.transform(np.array([[np.nan, np.nan]]))

    assert np.allclose(X_out, np.array([[2.0, 4.0]]))


def test_simple_imputer_invalid_strategy():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="wrong")


def test_simple_imputer_transform_before_fit():
    imputer = SimpleImputer()

    with pytest.raises(RuntimeError):
        imputer.transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_shape():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape


def test_standard_scaler_zero_variance():
    X = np.array([
        [1.0, 2.0],
        [1.0, 4.0],
        [1.0, 6.0],
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))


def test_standard_scaler_partial_fit_mean():
    scaler = StandardScaler()

    X1 = np.array([[1.0, 2.0]])
    X2 = np.array([[3.0, 4.0]])

    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    assert np.allclose(scaler.mean_, np.array([2.0, 3.0]))


def test_standard_scaler_inverse_transform():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_back = scaler.inverse_transform(X_scaled)

    assert np.allclose(X_back, X)


def test_standard_scaler_transform_before_fit():
    scaler = StandardScaler()

    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_handles_nan_in_transform():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    scaler = StandardScaler()
    scaler.fit(X)

    X_test = np.array([[np.nan, 2.0]])
    X_scaled = scaler.transform(X_test)

    assert not np.any(np.isnan(X_scaled))


def test_one_hot_encoder_basic():
    X = np.array([
        ["red"],
        ["blue"],
        ["red"],
    ], dtype=object)

    encoder = OneHotEncoder()
    X_out = encoder.fit_transform(X)

    assert X_out.shape == (3, 2)


def test_one_hot_encoder_partial_fit_new_category():
    encoder = OneHotEncoder()

    X1 = np.array([["red"], ["blue"]], dtype=object)
    X2 = np.array([["green"]], dtype=object)

    encoder.partial_fit(X1)
    encoder.partial_fit(X2)

    X_out = encoder.transform(np.array([["red"], ["green"]], dtype=object))

    assert X_out.shape == (2, 3)


def test_one_hot_encoder_unknown_ignore():
    encoder = OneHotEncoder(handle_unknown="ignore")

    X = np.array([["red"], ["blue"]], dtype=object)

    encoder.fit(X)
    X_out = encoder.transform(np.array([["yellow"]], dtype=object))

    assert np.allclose(X_out, np.zeros((1, 2)))


def test_one_hot_encoder_unknown_error():
    encoder = OneHotEncoder(handle_unknown="error")

    X = np.array([["red"], ["blue"]], dtype=object)

    encoder.fit(X)

    with pytest.raises(ValueError):
        encoder.transform(np.array([["yellow"]], dtype=object))


def test_one_hot_encoder_feature_names():
    encoder = OneHotEncoder()

    X = np.array([
        ["red"],
        ["blue"],
    ], dtype=object)

    encoder.fit(X)

    names = encoder.get_feature_names_out(["colour"])

    assert len(names) == 2


def test_one_hot_encoder_invalid_handle_unknown():
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="wrong")