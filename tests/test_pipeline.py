from __future__ import annotations

from pathlib import Path

from scripts.build_demo_dataset import make_demo_dataset
from ecoood.pipeline import run_benchmark


def test_benchmark_smoke(tmp_path: Path) -> None:
    df = make_demo_dataset(n=240, seed=123)
    summary = run_benchmark(
        df=df,
        splits=["random", "scaffold", "chemical_random", "chemical_class", "temporal"],
        models=["random_forest"],
        output_dir=str(tmp_path / "results"),
        seed=123,
        n_members=3,
    )
    assert not summary.empty
    assert {"split", "model", "rmse", "mae", "coverage", "aurc"}.issubset(summary.columns)
    assert (tmp_path / "results" / "ood_score_summary.csv").exists()
