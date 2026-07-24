from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("At least one calibration score is required.")
    rank = _conformal_rank(n, alpha)
    return float(scores[rank])


def _conformal_rank(n: int, alpha: float) -> int:
    if n <= 0:
        raise ValueError("Calibration size must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return rank


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
        self.n_calibration_: int = 0
        self.quantile_rank_: int | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scale: np.ndarray | None = None,
    ) -> "ScaledConformalRegressor":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if scale is None:
            scale = np.ones_like(y_true)
        scale = np.maximum(np.asarray(scale, dtype=float), self.eps)
        scores = np.abs(y_true - y_pred) / scale
        self.n_calibration_ = len(scores)
        self.quantile_rank_ = _conformal_rank(self.n_calibration_, self.alpha) + 1
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


class GroupConditionalScaledConformalRegressor:
    """Scaled split conformal intervals with pooled fallback for sparse groups.

    Group-specific quantiles are estimated only from calibration cases. Groups
    smaller than ``min_group_size`` use the pooled finite-sample quantile.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eps: float = 1e-6,
        min_group_size: int = 20,
    ) -> None:
        self.alpha = alpha
        self.eps = eps
        self.min_group_size = min_group_size
        self.pooled = ScaledConformalRegressor(alpha=alpha, eps=eps)
        self.group_qhat_: dict[str, float] = {}
        self.group_n_: dict[str, int] = {}

    @staticmethod
    def _groups(values) -> np.ndarray:
        return (
            np.asarray(values, dtype=object)
            .astype(str)
        )

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups,
        scale: np.ndarray | None = None,
    ) -> "GroupConditionalScaledConformalRegressor":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if scale is None:
            scale = np.ones_like(y_true)
        scale = np.maximum(np.asarray(scale, dtype=float), self.eps)
        group_values = self._groups(groups)
        if len(group_values) != len(y_true):
            raise ValueError("groups must have the same length as the calibration targets.")
        self.pooled.fit(y_true, y_pred, scale=scale)
        scores = np.abs(y_true - y_pred) / scale
        self.group_qhat_ = {}
        self.group_n_ = {}
        for group in sorted(set(group_values)):
            mask = group_values == group
            n_group = int(mask.sum())
            self.group_n_[group] = n_group
            if n_group >= self.min_group_size:
                self.group_qhat_[group] = _conformal_quantile(scores[mask], self.alpha)
        return self

    def predict(
        self,
        y_pred: np.ndarray,
        groups,
        scale: np.ndarray | None = None,
    ) -> IntervalResult:
        if self.pooled.qhat is None:
            raise RuntimeError("Conformal regressor must be fit before predict().")
        y_pred = np.asarray(y_pred, dtype=float)
        if scale is None:
            scale = np.ones_like(y_pred)
        scale = np.maximum(np.asarray(scale, dtype=float), self.eps)
        group_values = self._groups(groups)
        if len(group_values) != len(y_pred):
            raise ValueError("groups must have the same length as the predictions.")
        qhat = np.array(
            [
                self.group_qhat_.get(group, self.pooled.qhat)
                for group in group_values
            ],
            dtype=float,
        )
        radius = qhat * scale
        return IntervalResult(
            lower=y_pred - radius,
            upper=y_pred + radius,
            width=2.0 * radius,
        )


def decision_labels(
    scores: np.ndarray,
    widths: np.ndarray | None,
    score_warn_threshold: float,
    score_abstain_threshold: float,
    width_warn_threshold: float | None = None,
    width_abstain_threshold: float | None = None,
) -> np.ndarray:
    labels = np.full(len(scores), "predict", dtype=object)
    warn_mask = scores >= score_warn_threshold
    abstain_mask = scores >= score_abstain_threshold
    if widths is not None and width_warn_threshold is not None:
        warn_mask |= np.asarray(widths) >= width_warn_threshold
    if widths is not None and width_abstain_threshold is not None:
        abstain_mask |= np.asarray(widths) >= width_abstain_threshold
    labels[warn_mask] = "warn"
    labels[abstain_mask] = "abstain"
    return labels
