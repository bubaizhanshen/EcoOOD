from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_fewshot_recalibration import _aggregate_results, _sample_adaptation_indices


def test_sample_adaptation_indices_returns_disjoint_sets() -> None:
    frame = pd.DataFrame(
        {"chemical_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]}
    )
    adapt, eval_idx = _sample_adaptation_indices(
        frame,
        n_adapt=3,
        seed=7,
        group_column="chemical_id",
    )
    assert len(adapt) >= 3
    assert set(adapt).isdisjoint(set(eval_idx))
    assert sorted(np.concatenate([adapt, eval_idx]).tolist()) == list(range(10))
    selected = set(frame.iloc[adapt]["chemical_id"])
    assert all(
        frame.index[frame["chemical_id"].eq(chemical)].isin(adapt).all()
        for chemical in selected
    )


def test_sample_adaptation_indices_rejects_full_takeover() -> None:
    frame = pd.DataFrame({"chemical_id": ["a", "a", "b", "b", "c"]})
    with pytest.raises(ValueError):
        _sample_adaptation_indices(
            frame,
            n_adapt=5,
            seed=1,
            group_column="chemical_id",
        )


def test_aggregate_results_summarizes_mean_and_std() -> None:
    frame = pd.DataFrame(
        {
            "split": ["temporal", "temporal"],
            "model": ["lightgbm", "lightgbm"],
            "seed": [40, 41],
            "adapt_seed": [100, 101],
            "shots": [20, 20],
            "n_eval": [100, 100],
            "coverage_before": [0.82, 0.84],
            "coverage_after": [0.88, 0.90],
            "coverage_recovery": [0.06, 0.06],
            "mean_interval_width_before": [0.30, 0.32],
            "mean_interval_width_after": [0.35, 0.36],
            "aurc_before": [0.05, 0.06],
            "aurc_after": [0.04, 0.05],
            "aurc_improvement": [0.01, 0.01],
            "top_decile_error_capture_before": [0.4, 0.5],
            "top_decile_error_capture_after": [0.5, 0.6],
            "target_coverage_gap_before": [0.08, 0.06],
            "target_coverage_gap_after": [0.02, 0.0],
        }
    )

    summary = _aggregate_results(frame)
    row = summary.iloc[0]
    assert row["split"] == "temporal"
    assert row["shots"] == 20
    assert row["coverage_after_mean"] == pytest.approx(0.89)
    assert row["aurc_after_mean"] == pytest.approx(0.045)
