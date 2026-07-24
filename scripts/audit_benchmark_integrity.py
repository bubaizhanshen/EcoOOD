from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecoood.features import EcoFeatureBuilder
from ecoood.schema import DEFAULT_SCHEMA


EXACT_RECORD_KEY = [
    DEFAULT_SCHEMA.chemical_id,
    DEFAULT_SCHEMA.species,
    DEFAULT_SCHEMA.endpoint,
    DEFAULT_SCHEMA.effect,
    "measurement",
    DEFAULT_SCHEMA.duration_h,
    DEFAULT_SCHEMA.medium,
    DEFAULT_SCHEMA.temperature_c,
    DEFAULT_SCHEMA.ph,
    "ctx_hardness",
    DEFAULT_SCHEMA.study_year,
    DEFAULT_SCHEMA.target,
]


def _feature_block(builder: EcoFeatureBuilder, column: str) -> tuple[str, bool]:
    if column == DEFAULT_SCHEMA.smiles:
        return "chemical fingerprint", True
    if column in builder.descriptor_cols:
        return "physicochemical descriptor", True
    if column in builder.mechanism_cols:
        return "bioactivity proxy", True
    if column in builder.context_cols or column in builder.context_categorical_cols:
        return "experimental context", True
    if column in builder.species_cols or column in builder.species_categorical_cols:
        return "species/taxonomy", True
    if column in builder.categorical_cols:
        return "categorical model input", True
    return "identifier, target, or audit-only field", False


def _source_or_derivation(column: str) -> str:
    """Describe the field's provenance in the public predictor manifest."""
    schema = DEFAULT_SCHEMA
    if column.startswith("mech_"):
        return "invitrodb/ToxCast summary linkage"
    if column in {"smiles", "dtxsid", "inchikey", "structure_source"}:
        return "DSSTox/CompTox structure linkage"
    if column.startswith("physchem_"):
        return "DSSTox/CompTox physicochemical linkage or structure-derived value"
    if column in {schema.hard_ood, schema.known_ood, "chemical_class"}:
        return "curation and audit derivation"
    if column in {
        schema.target,
        "molar_concentration",
        schema.value,
        schema.unit,
        "endpoint_code",
        "effect",
        "measurement",
    }:
        return "ECOTOX value/end-point curation"
    if column in {
        schema.chemical_id,
        schema.chemical_name,
        schema.casrn,
        "doi",
        "source",
        "common_name",
    }:
        return "ECOTOX record metadata"
    return "ECOTOX study/taxon context curation"


def build_feature_manifest(df: pd.DataFrame) -> pd.DataFrame:
    builder = EcoFeatureBuilder(schema=DEFAULT_SCHEMA).fit(df)
    rows: list[dict[str, object]] = []
    for column in df.columns:
        block, included = _feature_block(builder, column)
        rows.append(
            {
                "field": column,
                "raw_dtype": str(df[column].dtype),
                "n_unique": int(df[column].nunique(dropna=True)),
                "missing_fraction": float(df[column].isna().mean()),
                "feature_block": block,
                "source_or_derivation": _source_or_derivation(column),
                "predictive_input": included,
            }
        )
    manifest = pd.DataFrame(rows)
    required_exclusions = {
        DEFAULT_SCHEMA.target,
        "molar_concentration",
        DEFAULT_SCHEMA.value,
        DEFAULT_SCHEMA.unit,
        DEFAULT_SCHEMA.chemical_id,
        DEFAULT_SCHEMA.casrn,
        DEFAULT_SCHEMA.chemical_name,
        "dtxsid",
        "inchikey",
        "doi",
    }
    leaked = manifest.loc[
        manifest["field"].isin(required_exclusions) & manifest["predictive_input"], "field"
    ].tolist()
    if leaked:
        raise RuntimeError(f"Target-derived or identifier fields entered the predictor matrix: {leaked}")
    return manifest


def duplicate_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    exact_keys = EXACT_RECORD_KEY.copy()
    exact_keys = [column for column in exact_keys if column in df.columns]
    checks = {
        "exact_record_key": exact_keys,
        "same_chemical_species_endpoint_target": [
            column
            for column in [
                DEFAULT_SCHEMA.chemical_id,
                DEFAULT_SCHEMA.species,
                DEFAULT_SCHEMA.endpoint,
                DEFAULT_SCHEMA.target,
            ]
            if column in df.columns
        ],
        "same_structure_species_endpoint": [
            column
            for column in ["inchikey", DEFAULT_SCHEMA.species, DEFAULT_SCHEMA.endpoint]
            if column in df.columns
        ],
    }
    rows: list[dict[str, object]] = []
    summary: dict[str, int] = {}
    for label, keys in checks.items():
        group_sizes = df.groupby(keys, dropna=False).size()
        duplicate_groups = group_sizes[group_sizes > 1]
        duplicate_rows = int(duplicate_groups.sum())
        rows.append(
            {
                "check": label,
                "key_fields": "; ".join(keys),
                "total_rows": int(len(df)),
                "duplicate_groups": int(len(duplicate_groups)),
                "rows_in_duplicate_groups": duplicate_rows,
                "rows_beyond_first_record": int(duplicate_rows - len(duplicate_groups)),
            }
        )
        summary[label] = duplicate_rows
    return pd.DataFrame(rows), summary


def deduplicate_exact_records(df: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in EXACT_RECORD_KEY if column in df.columns]
    return df.drop_duplicates(keys, keep="first").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit feature inclusion, target leakage exclusions, and duplicate records."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--deduplicated-output",
        type=Path,
        help="Optional path for a strict sensitivity dataset with exact duplicate records collapsed.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_feature_manifest(df)
    duplicate_frame, duplicate_summary = duplicate_audit(df)
    deduplicated = deduplicate_exact_records(df)

    manifest.to_csv(args.output_dir / "feature_manifest.csv", index=False)
    duplicate_frame.to_csv(args.output_dir / "duplicate_audit.csv", index=False)
    summary = {
        "data": str(args.data),
        "rows": int(len(df)),
        "chemicals": int(df[DEFAULT_SCHEMA.chemical_id].nunique()),
        "predictive_raw_fields": int(manifest["predictive_input"].sum()),
        "excluded_raw_fields": int((~manifest["predictive_input"]).sum()),
        "duplicate_checks": duplicate_summary,
        "deduplicated_rows": int(len(deduplicated)),
    }
    (args.output_dir / "integrity_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.deduplicated_output is not None:
        args.deduplicated_output.parent.mkdir(parents=True, exist_ok=True)
        deduplicated.to_csv(args.deduplicated_output, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
