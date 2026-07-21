from __future__ import annotations

import pandas as pd

from scripts.summarize_multiseed_screening import summarize_fixed_workload


def test_fixed_workload_compares_methods_at_identical_review_count() -> None:
    frame = pd.DataFrame(
        {
            "seed": [40] * 8,
            "model": ["lightgbm"] * 8,
            "split": ["temporal"] * 8,
            "chemical_id": [f"c{i}" for i in range(8)],
            "chemical_name": [f"C{i}" for i in range(8)],
            "casrn": [str(i) for i in range(8)],
            "chemical_class": ["test"] * 8,
            "min_true_tox": [-8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5],
            "min_pred_tox": [-5.0, -5.1, -7.1, -6.6, -6.0, -5.5, -5.0, -4.5],
            "max_ecoood": [0.99, 0.98, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "max_input_distance": [0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "max_similarity_risk": [0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "endpoint_breadth": [1] * 8,
            "row_count": [1] * 8,
        }
    )

    _, summary = summarize_fixed_workload(frame, burdens=(0.25,))

    assert summary["review_count"].nunique() == 1
    assert summary["review_count"].iloc[0] == 2
    ecoood = summary.loc[summary["method"] == "EcoOOD"].iloc[0]
    assert ecoood["rescued_false_negatives"] == 2
