from __future__ import annotations

import pandas as pd

from ecoood.dashboard import load_dashboard_bundle, upload_ready_scores


def test_load_dashboard_bundle_smoke() -> None:
    bundle = load_dashboard_bundle(".")
    assert not bundle.benchmark_metrics.empty
    assert not bundle.decision_points.empty
    assert not bundle.screening_panel.empty
    assert not bundle.gate_summary.empty


def test_upload_ready_scores_derives_decisions() -> None:
    df = pd.DataFrame(
        {
            "chemical_name": ["A", "B", "C"],
            "y_pred": [-7.1, -5.9, -4.8],
            "ecoood_score": [0.1, 0.55, 0.93],
            "interval_width": [0.08, 0.18, 0.42],
        }
    )
    prepared = upload_ready_scores(df)
    assert "decision" in prepared.columns
    assert set(prepared["decision"]).issubset({"predict", "warn", "abstain"})
