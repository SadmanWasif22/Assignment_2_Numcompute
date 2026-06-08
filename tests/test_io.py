
import os
import tempfile

import numpy as np
import pytest

from numcompute_stream.io import (
    load_csv,
    train_test_split,
    make_chunks,
    save_csv,
)


def test_load_csv_basic():
    content = "f1,f2,target\n1.0,2.0,0\n3.0,4.0,1\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as file:
        file.write(content)
        file_path = file.name

    try:
        X, y = load_csv(file_path, target_column=-1, has_header=True)

        assert X.shape == (2, 2)
        assert y.shape == (2,)
        assert np.allclose(X, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert np.allclose(y, np.array([0.0, 1.0]))
    finally:
        os.remove(file_path)


def test_load_csv_handles_missing_values():
    content = "f1,f2,target\n1.0,,0\n3.0,4.0,1\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as file:
        file.write(content)
        file_path = file.name

    try:
        X, y = load_csv(file_path, target_column=-1, has_header=True)

        assert np.isnan(X[0, 1])
        assert X.shape == (2, 2)
        assert y.shape == (2,)
    finally:
        os.remove(file_path)


def test_load_csv_invalid_target_column():
    content = "f1,f2,target\n1.0,2.0,0\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as file:
        file.write(content)
        file_path = file.name

    try:
        with pytest.raises(ValueError):
            load_csv(file_path, target_column=10, has_header=True)
    finally:
        os.remove(file_path)


def test_load_csv_inconsistent_rows():
    content = "f1,f2,target\n1.0,2.0,0\n3.0,1\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as file:
        file.write(content)
        file_path = file.name

    try:
        with pytest.raises(ValueError):
            load_csv(file_path, target_column=-1, has_header=True)
    finally:
        os.remove(file_path)


def test_train_test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)


def test_train_test_split_shuffle_reproducible():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    split1 = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    split2 = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

    assert np.array_equal(split1[0], split2[0])
    assert np.array_equal(split1[2], split2[2])


def test_train_test_split_invalid_test_size():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)


def test_make_chunks_count():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    chunks = list(make_chunks(X, y, chunk_size=3))

    assert len(chunks) == 4


def test_make_chunks_last_chunk_size():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    chunks = list(make_chunks(X, y, chunk_size=4))
    X_last, y_last = chunks[-1]

    assert X_last.shape == (2, 2)
    assert y_last.shape == (2,)


def test_make_chunks_without_y():
    X = np.arange(20).reshape(10, 2)

    chunks = list(make_chunks(X, y=None, chunk_size=5))

    assert len(chunks) == 2
    assert chunks[0].shape == (5, 2)


def test_make_chunks_invalid_chunk_size():
    X = np.arange(20).reshape(10, 2)

    with pytest.raises(ValueError):
        list(make_chunks(X, chunk_size=0))


def test_save_csv_and_load_csv():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as file:
        file_path = file.name

    try:
        save_csv(
            file_path,
            X,
            y,
            header=["f1", "f2", "target"],
        )

        X_loaded, y_loaded = load_csv(file_path, target_column=-1, has_header=True)

        assert np.allclose(X_loaded, X)
        assert np.allclose(y_loaded, y)
    finally:
        os.remove(file_path)