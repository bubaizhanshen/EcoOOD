from __future__ import annotations

import pytest

import pandas as pd

from scripts.build_ecotox_dataset import (
    clean_concentration_unit,
    concentration_to_molar,
    deterministic_rejection_flag,
)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("nM", 1e-9),
        ("umol/dm3", 1e-6),
        ("AI mg/L", 1e-5),
        ("mg/dm3", 1e-5),
        ("ug/mL", 1e-5),
        ("ng/mL", 1e-8),
    ],
)
def test_concentration_unit_equivalents(unit: str, expected: float) -> None:
    assert concentration_to_molar(1.0, unit, 100.0) == pytest.approx(expected)


def test_unit_normalization_handles_micro_and_spacing() -> None:
    assert clean_concentration_unit("  µg / L ") == "ug/l"


@pytest.mark.parametrize("unit", ["AE mg/L", "TOT mg/L"])
def test_qualified_mass_units_are_not_converted_as_parent_compound(unit: str) -> None:
    assert concentration_to_molar(1.0, unit, 100.0) is None


def test_unparseable_smiles_triggers_deterministic_rejection() -> None:
    row = pd.Series(
        {
            "chemical_class": "unclassified",
            "chemical_name": "Disodium hexafluorosilicate(2-)",
            "smiles": "[Na+].[Na+].F[Si--](F)(F)(F)(F)F",
        }
    )
    assert deterministic_rejection_flag(row)
