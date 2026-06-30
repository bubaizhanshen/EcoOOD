from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_failure_axes import aggregate, summarize_prediction_file


def test_summarize_prediction_file_reports_axis_enrichment(tmp_path: Path) -> None:
    path = tmp_path / "seeds" / "seed_40" / "all" / "species" / "lightgbm" / "predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "y_true": [0.0, 0.1, 0.2, 1.8, 2.2],
            "y_pred": [0.0, 0.1, 0.2, 0.0, 0.0],
            "d_chem": [0.1, 0.2, 0.1, 0.9, 1.0],
            "d_species": [0.0, 0.1, 0.0, 0.8, 0.9],
            "d_context": [0.1, 0.1, 0.1, 0.1, 0.1],
            "d_mech": [0.2, 0.2, 0.2, 0.3, 0.3],
        }
    )
    frame.to_csv(path, index=False)

    summary = summarize_prediction_file(path, error_quantile=0.8, tail_quantile=0.8)

    assert set(summary["axis"]) == {"chemical", "species", "context", "mechanism"}
    chem = summary.loc[summary["axis"] == "chemical"].iloc[0]
    assert chem["split"] == "species"
    assert chem["seed"] == 40
    assert chem["delta_mean"] > 0
    assert chem["enrichment_ratio"] > 1.0


def test_aggregate_failure_axes_returns_mean_columns() -> None:
    frame = pd.DataFrame(
        {
            "group": ["all", "all"],
            "split": ["species", "species"],
            "axis": ["chemical", "chemical"],
            "baseline_mean": [0.3, 0.5],
            "high_error_mean": [0.6, 1.0],
            "delta_mean": [0.3, 0.5],
            "enrichment_ratio": [2.0, 2.0],
            "high_tail_capture": [0.5, 0.6],
        }
    )

    aggregated = aggregate(frame)
    row = aggregated.iloc[0]
    assert row["group"] == "all"
    assert row["split"] == "species"
    assert row["axis"] == "chemical"
    assert row["delta_mean_mean"] == pytest.approx(0.4)
    assert row["enrichment_ratio_mean"] == pytest.approx(2.0)
