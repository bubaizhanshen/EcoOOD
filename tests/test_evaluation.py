from __future__ import annotations

import numpy as np
import pytest

from ecoood.evaluation import selective_risk_summary


def test_selective_risk_summary_improves_when_high_error_samples_removed() -> None:
    y_true = np.array([0.0, 0.1, 0.2, 2.0, 3.0], dtype=float)
    y_pred = np.array([0.0, 0.1, 0.2, 0.0, 0.0], dtype=float)
    score = np.array([0.0, 0.1, 0.2, 0.9, 1.0], dtype=float)

    summary = selective_risk_summary(y_true, y_pred, score, abstain_fractions=[0.0, 0.2, 0.4])

    assert list(summary["abstain_fraction"]) == [0.0, 0.2, 0.4]
    assert summary.loc[0, "retained_fraction"] == pytest.approx(1.0)
    assert summary.loc[2, "retained_fraction"] == pytest.approx(0.6)
    assert summary.loc[2, "rmse"] < summary.loc[0, "rmse"]
    assert summary.loc[2, "mean_abs_error"] < summary.loc[0, "mean_abs_error"]
    assert summary.loc[2, "top_decile_error_rate_reduction"] >= 0.0
