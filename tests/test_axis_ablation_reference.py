from __future__ import annotations

import pandas as pd


def test_reference_summary_columns_align_for_ablation_merge() -> None:
    ablation = pd.DataFrame(
        {
            "profile": ["chemical_only"],
            "split": ["temporal"],
            "model": ["lightgbm"],
            "rmse_mean": [1.2],
            "coverage_mean": [0.85],
        }
    )
    reference = pd.DataFrame(
        {
            "group": ["all"],
            "split": ["temporal"],
            "model": ["lightgbm"],
            "rmse_mean": [0.10],
            "coverage_mean": [0.90],
            "abstain_fraction_mean": [0.4],
        }
    )
    keep_cols = [col for col in ablation.columns if col in reference.columns or col in {"profile", "split", "model"}]
    merged = pd.concat(
        [
            ablation,
            reference.assign(profile="current_full")[keep_cols],
        ],
        ignore_index=True,
    )
    assert set(merged["profile"]) == {"chemical_only", "current_full"}
    assert "rmse_mean" in merged.columns
    assert "coverage_mean" in merged.columns
