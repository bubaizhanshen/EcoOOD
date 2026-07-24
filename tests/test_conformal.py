from __future__ import annotations

import numpy as np

from ecoood.conformal import (
    GroupConditionalScaledConformalRegressor,
    ScaledConformalRegressor,
)


def test_scaled_conformal_exposes_finite_sample_rank() -> None:
    y_true = np.arange(19, dtype=float)
    model = ScaledConformalRegressor(alpha=0.1).fit(
        y_true,
        y_true + 1.0,
        scale=np.ones_like(y_true),
    )

    assert model.n_calibration_ == 19
    assert model.quantile_rank_ == 18
    assert model.qhat == 1.0


def test_group_conditional_conformal_uses_pooled_fallback() -> None:
    y_true = np.array([0.0, 1.0, 10.0, 12.0])
    y_pred = np.zeros(4)
    groups = np.array(["fish", "fish", "algae", "algae"])
    model = GroupConditionalScaledConformalRegressor(
        alpha=0.5,
        min_group_size=2,
    ).fit(y_true, y_pred, groups, scale=np.ones(4))

    intervals = model.predict(
        np.zeros(2),
        np.array(["fish", "unseen"]),
        scale=np.ones(2),
    )

    assert intervals.width[0] == 2.0
    assert intervals.width[1] == 20.0
