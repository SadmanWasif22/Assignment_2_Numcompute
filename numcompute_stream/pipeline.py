
"""
pipeline.py

A simple streaming pipeline for NumCompute Stream.

Allowed libraries:
- Python standard library
- NumPy

This pipeline connects preprocessing steps and a final model.

Each transformer should support:
- partial_fit(X)
- transform(X)
- fit_transform(X)

The final model should support:
- partial_fit(X, y)
- predict(X)
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class Pipeline:
    """
    Streaming machine learning pipeline.

    Parameters
    ----------
    steps : list of tuple
        List of (name, object) pairs.

        Example:
        Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier())
        ])

    Notes
    -----
    All steps except the last are treated as transformers.
    The last step is treated as the model.
    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        if not isinstance(steps, list):
            raise TypeError("steps must be a list of (name, object) tuples.")

        if len(steps) == 0:
            raise ValueError("Pipeline must contain at least one step.")

        self.steps = steps
        self._validate_steps()

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "Pipeline":
        """
        Incrementally fit transformers and model using one chunk.

        Parameters
        ----------
        X : np.ndarray
            Feature chunk with shape (n_samples, n_features).

        y : np.ndarray
            Target chunk with shape (n_samples,).

        Returns
        -------
        self : Pipeline
        """

        X, y = self._validate_X_y(X, y)

        X_transformed = X

        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(X_transformed)
            elif hasattr(transformer, "fit"):
                transformer.fit(X_transformed)
            else:
                raise TypeError(
                    f"Transformer step '{name}' must have partial_fit() or fit()."
                )

            if not hasattr(transformer, "transform"):
                raise TypeError(f"Transformer step '{name}' must have transform().")

            X_transformed = transformer.transform(X_transformed)
            X_transformed = self._validate_X(X_transformed)

        model_name, model = self.steps[-1]

        if not hasattr(model, "partial_fit"):
            raise TypeError(f"Model step '{model_name}' must have partial_fit().")

        model.partial_fit(X_transformed, y)

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Pipeline":
        """
        Fit pipeline from scratch.

        If a step has fit_transform(), it will be used.
        Otherwise fit()/transform() will be used.
        """

        X, y = self._validate_X_y(X, y)

        X_transformed = X

        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, "fit_transform"):
                X_transformed = transformer.fit_transform(X_transformed)
            else:
                if not hasattr(transformer, "fit"):
                    raise TypeError(f"Transformer step '{name}' must have fit().")

                if not hasattr(transformer, "transform"):
                    raise TypeError(f"Transformer step '{name}' must have transform().")

                transformer.fit(X_transformed)
                X_transformed = transformer.transform(X_transformed)

            X_transformed = self._validate_X(X_transformed)

        model_name, model = self.steps[-1]

        if hasattr(model, "fit"):
            model.fit(X_transformed, y)
        elif hasattr(model, "partial_fit"):
            model.partial_fit(X_transformed, y)
        else:
            raise TypeError(
                f"Model step '{model_name}' must have fit() or partial_fit()."
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X through preprocessing steps and predict using final model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predictions with shape (n_samples,).
        """

        X = self._validate_X(X)

        X_transformed = self.transform(X)

        model_name, model = self.steps[-1]

        if not hasattr(model, "predict"):
            raise TypeError(f"Model step '{model_name}' must have predict().")

        y_pred = model.predict(X_transformed)
        y_pred = np.asarray(y_pred)

        if y_pred.ndim != 1:
            raise ValueError("Model predictions must be a 1D array.")

        if y_pred.shape[0] != X.shape[0]:
            raise ValueError(
                "Number of predictions must match number of samples. "
                f"Got predictions={y_pred.shape[0]} and samples={X.shape[0]}."
            )

        return y_pred

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply all transformer steps to X.

        The final model step is not used.
        """

        X = self._validate_X(X)
        X_transformed = X

        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"Transformer step '{name}' must have transform().")

            X_transformed = transformer.transform(X_transformed)
            X_transformed = self._validate_X(X_transformed)

        return X_transformed

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the full pipeline and return predictions on X.
        """

        self.fit(X, y)
        return self.predict(X)

    def get_step(self, name: str) -> Any:
        """
        Return a pipeline step by name.
        """

        for step_name, step in self.steps:
            if step_name == name:
                return step

        raise KeyError(f"No pipeline step named '{name}' found.")

    def set_step(self, name: str, new_step: Any) -> "Pipeline":
        """
        Replace a pipeline step by name.
        """

        for index, (step_name, _) in enumerate(self.steps):
            if step_name == name:
                self.steps[index] = (name, new_step)
                self._validate_steps()
                return self

        raise KeyError(f"No pipeline step named '{name}' found.")

    def _validate_steps(self) -> None:
        """
        Validate pipeline step format and names.
        """

        names = []

        for step in self.steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise TypeError("Each pipeline step must be a (name, object) tuple.")

            name, obj = step

            if not isinstance(name, str):
                raise TypeError("Pipeline step names must be strings.")

            if name == "":
                raise ValueError("Pipeline step names cannot be empty.")

            if obj is None:
                raise ValueError(f"Pipeline step '{name}' cannot be None.")

            names.append(name)

        if len(names) != len(set(names)):
            raise ValueError("Pipeline step names must be unique.")

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        """
        Validate feature matrix X.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        return X

    @staticmethod
    def _validate_y(y: np.ndarray) -> np.ndarray:
        """
        Validate target vector y.
        """

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array with shape (n_samples,).")

        if y.shape[0] == 0:
            raise ValueError("y must contain at least one sample.")

        return y

    @classmethod
    def _validate_X_y(cls, X: np.ndarray, y: np.ndarray):
        """
        Validate X and y together.
        """

        X = cls._validate_X(X)
        y = cls._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples. "
                f"Got X={X.shape[0]} and y={y.shape[0]}."
            )

        return X, y