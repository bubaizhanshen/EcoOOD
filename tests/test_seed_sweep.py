from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_seed_sweep import _aggregate_benchmark_summaries, _aggregate_ood_summaries, _safe_name


def test_safe_name_normalizes_strings() -> None:
    assert _safe_name("fish 96h/LC50") == "fish_96h_LC50"
    assert _safe_name("") == "missing"


def test_aggregate_seed_summaries_returns_mean_and_std() -> None:
    benchmark_frames = [
        pd.DataFrame([{"group": "all", "split": "random", "model": "lightgbm", "seed": 40, "rmse": 0.1, "coverage": 0.9}]),
        pd.DataFrame([{"group": "all", "split": "random", "model": "lightgbm", "seed": 41, "rmse": 0.2, "coverage": 0.8}]),
    ]
    ood_frames = [
        pd.DataFrame([{"group": "all", "split": "random", "model": "lightgbm", "method": "ecoood", "seed": 40, "aurc": 0.01}]),
        pd.DataFrame([{"group": "all", "split": "random", "model": "lightgbm", "method": "ecoood", "seed": 41, "aurc": 0.03}]),
    ]
    _, bench_agg = _aggregate_benchmark_summaries(benchmark_frames)
    _, ood_agg = _aggregate_ood_summaries(ood_frames)
    assert bench_agg.loc[0, "rmse_mean"] == pytest.approx(0.15)
    assert bench_agg.loc[0, "coverage_mean"] == pytest.approx(0.85)
    assert ood_agg.loc[0, "aurc_mean"] == pytest.approx(0.02)
