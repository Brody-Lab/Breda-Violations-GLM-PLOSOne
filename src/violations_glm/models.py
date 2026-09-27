"""Static logistic-regression models used in the publication."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MultinomialLogisticRegression:
    def __init__(self, sigma: float, *, method: str = "BFGS") -> None:
        self.sigma = sigma
        self.method = method
        self.weights: np.ndarray | None = None

    def fit(self, design: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        values = design.to_numpy()
        initial = np.zeros(values.shape[1] * labels.shape[1])
        result = minimize(
            self._cost,
            initial,
            args=(values, labels, self.sigma),
            method=self.method,
            jac=self._gradient,
            options={"disp": False},
        )
        self.weights = result.x.reshape(values.shape[1], labels.shape[1])
        return self.weights

    def evaluate(
        self, design: pd.DataFrame, labels: np.ndarray, *, lr_only: bool = False
    ) -> float:
        if self.weights is None:
            raise RuntimeError("Model has not been fit")
        return float(
            self._cost(
                self.weights,
                design.to_numpy(),
                labels,
                sigma=None,
                lr_only=lr_only,
            )
        )

    @staticmethod
    def _log_sum_exp(logits: np.ndarray) -> np.ndarray:
        maximum = np.max(logits, axis=1, keepdims=True)
        return np.log(np.sum(np.exp(logits - maximum), axis=1, keepdims=True)) + maximum

    @staticmethod
    def _log_sum_exp_lr(logits: np.ndarray) -> np.ndarray:
        maximum = np.max(logits[:, :2], axis=1, keepdims=True)
        return (
            np.log(
                np.sum(
                    np.exp(logits[:, :2] - maximum),
                    axis=1,
                    keepdims=True,
                )
            )
            + maximum
        )

    @classmethod
    def _cost(
        cls,
        weights: np.ndarray,
        design: np.ndarray,
        labels: np.ndarray,
        sigma: float | None,
        lr_only: bool = False,
    ) -> float:
        if weights.ndim == 1:
            weights = weights.reshape(design.shape[1], labels.shape[1])
        logits = design @ weights
        denominator = (
            cls._log_sum_exp_lr(logits) if lr_only else cls._log_sum_exp(logits)
        )
        penalty = (
            np.trace(weights[1:, :].T @ weights[1:, :]) / (2 * sigma**2) if sigma else 0
        )
        return float(-np.sum(labels * logits) + np.sum(denominator) + penalty)

    @staticmethod
    def _gradient(
        weights: np.ndarray,
        design: np.ndarray,
        labels: np.ndarray,
        sigma: float | None,
    ) -> np.ndarray:
        if weights.ndim == 1:
            weights = weights.reshape(design.shape[1], labels.shape[1])
        logits = design @ weights
        exponentials = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        penalty = weights / sigma**2 if sigma else np.zeros_like(weights)
        penalty[0, :] = 0
        return (design.T @ (probabilities - labels) + penalty).ravel()


class BinaryLogisticRegression:
    """Binary GLM retaining the submitted regularization implementation."""

    def __init__(
        self,
        sigma: float,
        *,
        rng: np.random.Generator,
        method: str = "BFGS",
    ) -> None:
        self.sigma = sigma
        self.rng = rng
        self.method = method
        self.weights: np.ndarray | None = None

    def fit(self, design: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        values = design.to_numpy()
        initial = self.rng.normal(0, 0.1, values.shape[1])
        result = minimize(
            self._cost,
            initial,
            args=(values, labels, self.sigma),
            method=self.method,
            jac=self._gradient,
            options={"disp": False, "maxiter": 10_000},
        )
        self.weights = result.x
        return self.weights

    def evaluate(self, design: pd.DataFrame, labels: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError("Model has not been fit")
        return float(self._cost(self.weights, design.to_numpy(), labels, None))

    @staticmethod
    def _cost(
        weights: np.ndarray,
        design: np.ndarray,
        labels: np.ndarray,
        sigma: float | None,
    ) -> float:
        logits = design @ weights
        penalty = np.sum(weights**2) / (2 * sigma**2) if sigma else 0
        # The submitted implementation subtracts this penalty. Preserve it for
        # regression parity; changing it would alter the published fits.
        return float(-np.sum(labels * logits - np.logaddexp(0, logits)) - penalty)

    @staticmethod
    def _gradient(
        weights: np.ndarray,
        design: np.ndarray,
        labels: np.ndarray,
        sigma: float | None,
    ) -> np.ndarray:
        logits = design @ weights
        probabilities = 1 / (1 + np.exp(-logits))
        if sigma:
            penalty = weights / sigma**2
            penalty[0] = 0
        else:
            penalty = 0
        return -design.T @ (labels - probabilities) + penalty
