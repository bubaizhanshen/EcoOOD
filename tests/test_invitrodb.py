from __future__ import annotations

import pandas as pd

from ecoood.invitrodb import (
    aggregate_cytotox_table,
    aggregate_h295r_table,
    aggregate_literature_table,
    attach_mechanistic_features,
)


def test_aggregate_cytotox_table_computes_hit_rate() -> None:
    df = pd.DataFrame(
        [
            {
                "dsstox_substance_id": "DTXSID001",
                "casn": "50-00-0",
                "cytotox_median_um": "100",
                "cytotox_lower_bound_um": "10",
                "cytotox_median_log": "2",
                "cytotox_lower_bound_log": "1",
                "ntested": "20",
                "nhit": "5",
            }
        ]
    )
    result = aggregate_cytotox_table(df).set_index("dtxsid")
    assert result.loc["DTXSID001", "mech_cytotox_hit_rate"] == 0.25
    assert result.loc["DTXSID001", "mech_cytotox_median_um"] == 100.0


def test_aggregate_literature_table_pivots_modes_with_strength_ordering() -> None:
    df = pd.DataFrame(
        [
            {"dtxsid": "DTXSID001", "casrn": "50-00-0", "literature_mode": "ar_literature_binding", "literature_score": "Weak"},
            {"dtxsid": "DTXSID001", "casrn": "50-00-0", "literature_mode": "ar_literature_binding", "literature_score": "Strong"},
            {"dtxsid": "DTXSID001", "casrn": "50-00-0", "literature_mode": "er_literature_agonist", "literature_score": "Inactive"},
            {"dtxsid": "DTXSID002", "casrn": "64-17-5", "literature_mode": "er_literature_binding", "literature_score": "Very Weak"},
        ]
    )
    result = aggregate_literature_table(df).set_index("dtxsid")
    assert result.loc["DTXSID001", "mech_lit_ar_binding"] == 1.0
    assert result.loc["DTXSID001", "mech_lit_er_agonist"] == 0.0
    assert result.loc["DTXSID002", "mech_lit_er_binding"] == 0.25


def test_aggregate_h295r_table_aggregates_max_abs_and_activity_count() -> None:
    row_a = {
        "dsstox_substance_id": "DTXSID001",
        "casn": "50-00-0",
        "OHPREG": 0.2,
        "PROG": -1.5,
        "OHPROG": 0.0,
        "DOC": 0.0,
        "CORTIC": 0.0,
        "X11DCORT": 0.0,
        "CORT": 0.0,
        "ANDR": 0.4,
        "TESTO": 0.0,
        "E1": 0.0,
        "E2": 0.0,
        "mMd": 1.0,
        "maxmMd": 3.0,
        "BMD": 2.0,
        "criticalVal": 1.64,
    }
    row_b = row_a | {"PROG": -0.2, "ANDR": 2.1, "BMD": 0.5, "mMd": 1.5, "maxmMd": 4.0}
    result = aggregate_h295r_table(pd.DataFrame([row_a, row_b])).set_index("dtxsid")
    assert result.loc["DTXSID001", "mech_h295r_prog_max_abs"] == 1.5
    assert result.loc["DTXSID001", "mech_h295r_andr_max_abs"] == 2.1
    assert result.loc["DTXSID001", "mech_h295r_bmd_min"] == 0.5
    assert result.loc["DTXSID001", "mech_h295r_active_endpoint_count"] == 2


def test_attach_mechanistic_features_prefers_dtxsid_then_falls_back_to_casrn() -> None:
    dataset = pd.DataFrame(
        [
            {"chemical_name": "A", "dtxsid": "DTXSID001", "cas_number": "50-00-0"},
            {"chemical_name": "B", "dtxsid": "", "cas_number": "64175"},
            {"chemical_name": "C", "dtxsid": "", "cas_number": "77777"},
        ]
    )
    mechanisms = pd.DataFrame(
        [
            {"dtxsid": "DTXSID001", "casrn": "50-00-0", "casrn_norm": "50000", "mech_signal_a": 1.0},
            {"dtxsid": "DTXSID002", "casrn": "64-17-5", "casrn_norm": "64175", "mech_signal_a": 2.0},
        ]
    )
    result = attach_mechanistic_features(dataset, mechanisms)
    assert result.loc[0, "mech_signal_a"] == 1.0
    assert result.loc[1, "mech_signal_a"] == 2.0
    assert pd.isna(result.loc[2, "mech_signal_a"])
