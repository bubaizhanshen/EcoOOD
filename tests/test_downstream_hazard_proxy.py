from __future__ import annotations

import pandas as pd

from scripts.generate_downstream_hazard_proxy import (
    aggregate_species_distribution,
    apply_downstream_gate,
)


def _toy_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for species_idx in range(5):
        rows.append(
            {
                "split": "temporal",
                "chemical_id": "chem_a",
                "chemical_name": "Chem A",
                "casrn": "1",
                "chemical_class": "PFAS",
                "species": f"sp_{species_idx}",
                "y_true": -7.0 - 0.1 * species_idx,
                "y_pred": -7.0 - 0.1 * species_idx,
                "ecoood_score": 0.15,
                "interval_width": 0.20,
            }
        )
        rows.append(
            {
                "split": "temporal",
                "chemical_id": "chem_b",
                "chemical_name": "Chem B",
                "casrn": "2",
                "chemical_class": "PFAS",
                "species": f"sp_{species_idx}",
                "y_true": -7.4 - 0.1 * species_idx,
                "y_pred": -6.1 - 0.05 * species_idx,
                "ecoood_score": 0.95,
                "interval_width": 0.80,
            }
        )
    return pd.DataFrame(rows)


def test_aggregate_species_distribution_builds_hc5_proxy():
    panel = aggregate_species_distribution(_toy_predictions(), min_species=5)

    assert len(panel) == 2
    assert set(panel["chemical_id"]) == {"chem_a", "chem_b"}
    assert panel["n_species"].tolist() == [5, 5]
    chem_b = panel.loc[panel["chemical_id"] == "chem_b"].iloc[0]
    assert chem_b["hc5_proxy_abs_error"] > 1.0


def test_apply_downstream_gate_withholds_high_ood_proxy():
    panel = aggregate_species_distribution(_toy_predictions(), min_species=5)
    gated, pooled_hi = apply_downstream_gate(panel)

    assert pooled_hi >= 0.0
    chem_a = gated.loc[gated["chemical_id"] == "chem_a"].iloc[0]
    chem_b = gated.loc[gated["chemical_id"] == "chem_b"].iloc[0]
    assert chem_a["gate_action"] == "propagate"
    assert chem_b["gate_action"] == "withhold"
    assert bool(chem_b["downstream_high_error"])
