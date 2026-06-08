"""
NumCompute Stream

A lightweight streaming machine learning framework built using only
plain Python, NumPy, and matplotlib.

This package supports:
- CSV loading and chunk creation
- streaming statistics
- streaming preprocessing
- decision tree classification
- ensemble classification
- streaming metrics
- pipeline training
- stream training logs
- visualisation utilities
"""

__version__ = "0.1.0"

# I/O utilities
from .io import load_csv, train_test_split, make_chunks

# Streaming statistics
from .stats import StreamingStats

# Streaming metrics
from .metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)

# Preprocessing
from .preprocessing import (
    SimpleImputer,
    StandardScaler,
    OneHotEncoder,
)


from .tree import DecisionTreeClassifier
from .ensemble import EnsembleClassifier


from .pipeline import Pipeline
from .stream import StreamTrainer

__all__ = [
    "__version__",

    # it is for io.py
    "load_csv",
    "train_test_split",
    "make_chunks",

    #it is for stats.py
    "StreamingStats",

    # for metrics.py
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",

    # for preprocessing.py
    "SimpleImputer",
    "StandardScaler",
    "OneHotEncoder",

    # tree.py
    "DecisionTreeClassifier",

    # ensemble.py
    "EnsembleClassifier",

    # pipeline.py
    "Pipeline",

    # stream.py
    "StreamTrainer",
]