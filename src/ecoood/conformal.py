from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(scores[rank])


@dataclass
class IntervalResult:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray


class ScaledConformalRegressor:
    def __init__(self, alpha: float = 0.1, eps: float = 1e-6) -> None:
        self.alpha = alpha
        self.eps = eps
        self.qhat: float | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray | None = None) -> "ScaledConformalRegressor":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if scale is None:
            scale = np.ones_like(y_true)
        scale = np.maximum(np.asarray(scale, dtype=float), self.eps)
        scores = np.abs(y_true - y_pred) / scale
        self.qhat = _conformal_quantile(scores, self.alpha)
        return self

    def predict(self, y_pred: np.ndarray, scale: np.ndarray | None = None) -> IntervalResult:
        if self.qhat is None:
            raise RuntimeError("Conformal regressor must be fit before predict().")
        y_pred = np.asarray(y_pred, dtype=float)
        if scale is None:
            scale = np.ones_like(y_pred)
        scale = np.maximum(np.asarray(scale, dtype=float), self.eps)
        radius = self.qhat * scale
        lower = y_pred - radius
        upper = y_pred + radius
        return IntervalResult(lower=lower, upper=upper, width=upper - lower)


def decision_labels(
    scores: np.ndarray,
    widths: np.ndarray,
    score_warn_threshold: float,
    score_abstain_threshold: float,
    width_warn_threshold: float,
    width_abstain_threshold: float,
) -> np.ndarray:
    labels = np.full(len(scores), "predict", dtype=object)
    warn_mask = (scores >= score_warn_threshold) | (widths >= width_warn_threshold)
    abstain_mask = (scores >= score_abstain_threshold) | (widths >= width_abstain_threshold)
    labels[warn_mask] = "warn"
    labels[abstain_mask] = "abstain"
    return labels

