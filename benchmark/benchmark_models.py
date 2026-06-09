"""
benchmark_models.py

Benchmark single decision tree vs ensemble model under streaming conditions.


python benchmark/benchmark_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from numcompute_stream.io import load_csv, train_test_split, make_chunks
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline


def make_tree_pipeline() -> Pipeline:
    """
    Creating a single decision tree pipeline.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=2,
            criterion="gini",
            random_state=42,
        )),
    ])


def make_ensemble_pipeline() -> Pipeline:
    """
    Creating a bagging ensemble pipeline.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", EnsembleClassifier(
            n_estimators=5,
            max_depth=4,
            min_samples_split=2,
            criterion="gini",
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
        )),
    ])


def run_streaming_benchmark(
    model_name: str,
    pipeline: Pipeline,
    train_chunks: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Training a pipeline chunk by chunk and measure accuracy/time.
    """

    train_start = time.perf_counter()

    for X_chunk, y_chunk in train_chunks:
        pipeline.partial_fit(X_chunk, y_chunk)

    train_end = time.perf_counter()

    predict_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_end = time.perf_counter()

    accuracy = float(np.mean(y_pred == y_test))
    train_time = train_end - train_start
    predict_time = predict_end - predict_start

    return {
        "model": model_name,
        "accuracy": accuracy,
        "train_time": train_time,
        "predict_time": predict_time,
    }


def main() -> None:
    """
    Running benchmark.
    """

    dataset_path = PROJECT_ROOT / "demo" / "sample_data.csv"

    X, y = load_csv(
        file_path=str(dataset_path),
        target_column=-1,
        has_header=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    )

    train_chunks = list(make_chunks(
        X_train,
        y_train,
        chunk_size=6,
    ))

    tree_result = run_streaming_benchmark(
        model_name="Single Decision Tree",
        pipeline=make_tree_pipeline(),
        train_chunks=train_chunks,
        X_test=X_test,
        y_test=y_test,
    )

    ensemble_result = run_streaming_benchmark(
        model_name="Bagging Ensemble",
        pipeline=make_ensemble_pipeline(),
        train_chunks=train_chunks,
        X_test=X_test,
        y_test=y_test,
    )

    results = [tree_result, ensemble_result]

    print("\nStreaming Model Benchmark")
    print("=" * 78)
    print(f"{'Model':<25} {'Accuracy':<12} {'Train Time (s)':<18} {'Predict Time (s)':<18}")
    print("-" * 78)

    for result in results:
        print(
            f"{result['model']:<25} "
            f"{result['accuracy']:<12.4f} "
            f"{result['train_time']:<18.6f} "
            f"{result['predict_time']:<18.6f}"
        )

    print("=" * 78)

    print("\nInterpretation:")
    print("- Single Decision Tree is expected to train faster.")
    print("- Bagging Ensemble may be more stable but usually takes longer.")
    print("- These results can be copied into the report benchmark section.")


if __name__ == "__main__":
    main()