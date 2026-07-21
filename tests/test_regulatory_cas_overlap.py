from __future__ import annotations

import pandas as pd

from ecoood.dsstox import normalize_casrn
from scripts.build_pmra_regulatory_candidate_set import build_external_candidate_chemicals


def test_cas_overlap_uses_format_independent_keys() -> None:
    training = pd.Series([64175.0, "94-75-7", None]).map(normalize_casrn)
    external = pd.Series(["64-17-5", "94757", "135410-20-7"]).map(normalize_casrn)

    training_keys = {value for value in training if value}
    assert external.iloc[0] in training_keys
    assert external.iloc[1] in training_keys
    assert external.iloc[2] not in training_keys


def test_external_candidate_selection_is_independent_and_multi_endpoint() -> None:
    candidates = pd.DataFrame(
        {
            "chemical_name": ["eligible", "overlap", "single endpoint", "adopted source"],
            "representative_casrn": ["1-11-1", "2-22-2", "3-33-3", "4-44-4"],
            "casrn_normalized": ["1111", "2222", "3333", "4444"],
            "reference_codes": ["RVD2020-01"] * 4,
            "endpoint_count": [2, 3, 1, 3],
            "year_updated": [2020, 2021, 2022, 2023],
            "external_independence": [True, True, True, False],
            "has_valid_representative_casrn": [True] * 4,
            "training_overlap_by_casrn": [False, True, False, False],
        }
    )

    selected = build_external_candidate_chemicals(candidates)

    assert selected["chemical_name"].tolist() == ["eligible"]
    assert selected.loc[0, "min_year"] == 2020
