from __future__ import annotations

import pandas as pd
import pytest

from scripts.generate_screening_gate_validation import classify_screening_actions, summarize_screening_gate


def test_classify_screening_actions_diverts_false_reassurance() -> None:
    frame = pd.DataFrame(
        {
            "model": ["random_forest"] * 4,
            "split": ["temporal"] * 4,
            "chemical_id": ["c1", "c2", "c3", "c4"],
            "chemical_name": ["a", "b", "c", "d"],
            "casrn": ["1", "2", "3", "4"],
            "chemical_class": ["PFAS", "PFAS", "PPCPs", "PPCPs"],
            "min_true_tox": [-7.0, -7.2, -5.0, -4.9],
            "min_pred_tox": [-6.8, -5.1, -4.8, -6.7],
            "max_ecoood": [0.10, 0.90, 0.95, 0.20],
            "endpoint_breadth": [1, 1, 1, 1],
            "row_count": [1, 1, 1, 1],
        }
    )

    classified = classify_screening_actions(
        frame,
        toxicity_cutoff=-6.0,
        ood_cutoff_by_model={"random_forest": 0.5},
    )

    row = classified.set_index("chemical_id")
    assert row.loc["c1", "baseline_action"] == "screen_now"
    assert row.loc["c1", "gated_action"] == "screen_now"
    assert row.loc["c2", "baseline_action"] == "lower_priority"
    assert row.loc["c2", "gated_action"] == "withhold_review"
    assert row.loc["c2", "rescued_by_gate"]
    assert row.loc["c4", "gated_action"] == "screen_now"


def test_summarize_screening_gate_reports_false_reassurance_reduction() -> None:
    classified = pd.DataFrame(
        {
            "model": ["random_forest"] * 4,
            "split": ["temporal"] * 4,
            "toxicity_cutoff": [-6.0] * 4,
            "ood_cutoff": [0.5] * 4,
            "true_high_concern": [True, True, False, False],
            "baseline_action": ["screen_now", "lower_priority", "lower_priority", "screen_now"],
            "gated_action": ["screen_now", "withhold_review", "lower_priority", "prioritize_testing"],
            "rescued_by_gate": [False, True, False, False],
        }
    )

    metrics, summary = summarize_screening_gate(classified)

    false_rows = metrics[(metrics["metric"] == "false_reassurance_rate") & (metrics["split"] == "pooled")].set_index("workflow")
    assert false_rows.loc["baseline_only", "value"] == pytest.approx(0.5)
    assert false_rows.loc["baseline_plus_gate", "value"] == pytest.approx(0.0)

    pooled = summary[summary["split"] == "pooled"].iloc[0]
    assert pooled["baseline_false_negatives"] == 1
    assert pooled["rescued_false_negatives"] == 1
    assert pooled["rescued_false_negative_fraction"] == pytest.approx(1.0)
