
import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.metrics import Accuracy, Precision, Recall
from numcompute_stream.stream import StreamTrainer


def make_tree_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])


def make_ensemble_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", EnsembleClassifier(n_estimators=3, max_depth=2, random_state=42)),
    ])


def sample_data():
    X = np.array([
        [1.0, 2.0],
        [1.0, np.nan],
        [5.0, 6.0],
        [6.0, 5.0],
    ])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_stream_trainer_fit_chunk():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    log = trainer.fit_chunk(X, y)

    assert log["chunk"] == 1
    assert log["samples_in_chunk"] == 4
    assert trainer.samples_seen == 4


def test_stream_trainer_score_chunk():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    trainer.fit_chunk(X, y)
    scores = trainer.score_chunk(X, y)

    assert "accuracy" in scores
    assert 0.0 <= scores["accuracy"] <= 1.0


def test_stream_trainer_fit_score_chunk():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    log = trainer.fit_score_chunk(X, y)

    assert "accuracy" in log
    assert log["chunk"] == 1
    assert trainer.samples_seen == 4


def test_stream_trainer_fit_stream_two_chunks():
    X, y = sample_data()

    chunks = [
        (X[:2], y[:2]),
        (X[2:], y[2:]),
    ]

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    logs = trainer.fit_stream(chunks)

    assert len(logs) == 2
    assert trainer.samples_seen == 4


def test_stream_trainer_metric_history():
    X, y = sample_data()

    chunks = [
        (X[:2], y[:2]),
        (X[2:], y[2:]),
    ]

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    trainer.fit_stream(chunks)
    history = trainer.get_metric_history("accuracy")

    assert len(history) == 2


def test_stream_trainer_summary():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    trainer.fit_score_chunk(X, y)
    summary = trainer.summary()

    assert summary["model"] == "tree"
    assert summary["chunks_processed"] == 1
    assert summary["samples_seen"] == 4
    assert "accuracy" in summary["latest_metrics"]


def test_stream_trainer_reset():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    trainer.fit_score_chunk(X, y)
    trainer.reset()

    assert trainer.samples_seen == 0
    assert trainer.chunk_index == 0
    assert trainer.get_logs() == []
    assert trainer.get_metric_history("accuracy") == []


def test_stream_trainer_predict_chunk():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    trainer.fit_chunk(X, y)
    pred = trainer.predict_chunk(X)

    assert pred.shape == y.shape


def test_stream_trainer_multiple_metrics():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={
            "accuracy": Accuracy(),
            "precision": Precision(),
            "recall": Recall(),
        },
        name="tree",
    )

    log = trainer.fit_score_chunk(X, y)

    assert "accuracy" in log
    assert "precision" in log
    assert "recall" in log


def test_stream_trainer_with_ensemble_pipeline():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_ensemble_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="ensemble",
    )

    log = trainer.fit_score_chunk(X, y)

    assert "accuracy" in log
    assert trainer.samples_seen == 4


def test_stream_trainer_invalid_pipeline_error():
    class BadPipeline:
        pass

    with pytest.raises(TypeError):
        StreamTrainer(BadPipeline(), metrics={"accuracy": Accuracy()})


def test_stream_trainer_shape_mismatch_error():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    y = np.array([0])

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    with pytest.raises(ValueError):
        trainer.fit_chunk(X, y)


def test_stream_trainer_unknown_metric_history_error():
    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
    )

    with pytest.raises(KeyError):
        trainer.get_metric_history("unknown")


def test_stream_trainer_logs_include_memory():
    X, y = sample_data()

    trainer = StreamTrainer(
        pipeline=make_tree_pipeline(),
        metrics={"accuracy": Accuracy()},
        name="tree",
        log_memory=True,
    )

    log = trainer.fit_chunk(X, y)

    assert "memory_bytes" in log
    assert log["memory_bytes"] > 0