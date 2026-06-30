from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_axis_ablation import _aggregate_results, _prepare_profile_frame
from ecoood.schema import DEFAULT_SCHEMA


def test_prepare_profile_frame_keeps_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "target_log_molar": [0.1],
            "smiles": ["CCO"],
            "endpoint": ["fish_96h_lc50"],
            "chemical_id": ["c1"],
            "chemical_name": ["ethanol"],
            "casrn": ["64-17-5"],
            "is_hard_ood": [False],
            "known_ood": [False],
            "species": ["Danio rerio"],
            "genus": ["Danio"],
            "medium": ["FW"],
            "source": ["ecotox"],
            "physchem_mol_wt": [46.07],
            "ctx_hardness": [50.0],
            "mech_signal": [0.2],
        }
    )

    chem = _prepare_profile_frame(df, profile="chemical_only", schema=DEFAULT_SCHEMA)
    assert "physchem_mol_wt" in chem.columns
    assert "species" not in chem.columns
    assert "ctx_hardness" not in chem.columns
    assert "mech_signal" not in chem.columns

    full = _prepare_profile_frame(df, profile="chemical_species_context_mechanism", schema=DEFAULT_SCHEMA)
    assert "species" in full.columns
    assert "ctx_hardness" in full.columns
    assert "mech_signal" in full.columns


def test_aggregate_results_computes_means() -> None:
    frame = pd.DataFrame(
        {
            "profile": ["chemical_only", "chemical_only"],
            "split": ["temporal", "temporal"],
            "model": ["lightgbm", "lightgbm"],
            "seed": [40, 41],
            "rmse": [0.1, 0.2],
            "coverage": [0.85, 0.9],
            "aurc": [0.05, 0.07],
        }
    )
    summary = _aggregate_results(frame)
    row = summary.iloc[0]
    assert row["profile"] == "chemical_only"
    assert row["rmse_mean"] == pytest.approx(0.15)
    assert row["coverage_mean"] == pytest.approx(0.875)
