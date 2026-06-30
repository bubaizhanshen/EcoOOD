from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EcoOODSchema:
    target: str = "target_log_molar"
    smiles: str = "smiles"
    endpoint: str = "endpoint"
    chemical_id: str = "chemical_id"
    chemical_name: str = "chemical_name"
    chemical_class: str = "chemical_class"
    casrn: str = "casrn"
    species: str = "species"
    genus: str = "genus"
    family: str = "family"
    order: str = "order"
    clazz: str = "class_name"
    phylum: str = "phylum"
    trophic_group: str = "trophic_group"
    duration_h: str = "duration_h"
    medium: str = "medium"
    temperature_c: str = "temperature_c"
    ph: str = "ph"
    study_year: str = "study_year"
    source: str = "source"
    hard_ood: str = "is_hard_ood"
    known_ood: str = "known_ood"
    effect: str = "effect"
    value: str = "toxicity_value"
    unit: str = "toxicity_unit"
    numeric_feature_prefixes: tuple[str, ...] = (
        "physchem_",
        "mech_",
        "ctx_",
        "tax_",
    )
    protected_columns: set[str] = field(
        default_factory=lambda: {
            "target_log_molar",
            "smiles",
            "endpoint",
            "chemical_id",
            "chemical_name",
            "chemical_class",
            "casrn",
            "species",
            "genus",
            "family",
            "order",
            "class_name",
            "phylum",
            "trophic_group",
            "duration_h",
            "medium",
            "temperature_c",
            "ph",
            "study_year",
            "source",
            "is_hard_ood",
            "known_ood",
            "effect",
            "toxicity_value",
            "toxicity_unit",
        }
    )


DEFAULT_SCHEMA = EcoOODSchema()

