from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoood.features import smiles_to_mol
from ecoood.schema import DEFAULT_SCHEMA


METAL_OR_INORGANIC_CLASSES = {
    "arsenic",
    "cadmium",
    "chromium",
    "copper",
    "lead",
    "mercury",
    "nickel",
    "silver",
    "zinc",
}
AMBIGUOUS_NAME_TOKENS = ("mixture", "unknown")


def _blank_or_unparseable_smiles(value: object) -> bool:
    if pd.isna(value) or not str(value).strip():
        return True
    return smiles_to_mol(str(value)) is None


def classify_deterministic_rejections(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    names = result[DEFAULT_SCHEMA.chemical_name].fillna("").astype(str).str.casefold()
    classes = result[DEFAULT_SCHEMA.chemical_class].fillna("").astype(str).str.casefold()
    missing_structure = result[DEFAULT_SCHEMA.smiles].map(_blank_or_unparseable_smiles)
    ambiguous_name = names.map(lambda name: any(token in name for token in AMBIGUOUS_NAME_TOKENS))
    inorganic_or_metal = classes.map(
        lambda label: any(token in label for token in METAL_OR_INORGANIC_CLASSES)
    ) | names.str.contains("inorganic|organomet", regex=True)
    salt_like = names.str.contains("salt", regex=False)

    result["missing_or_unparseable_structure"] = missing_structure
    result["mixture_or_unknown_identity"] = ambiguous_name
    result["inorganic_or_metal_scope"] = inorganic_or_metal
    result["salt_like_scope_flag"] = salt_like
    result["deterministic_action"] = "withhold_review"

    result["rejection_category"] = "other_scope_excluded"
    result.loc[salt_like, "rejection_category"] = "salt_like_scope_flag"
    result.loc[inorganic_or_metal, "rejection_category"] = "inorganic_or_metal_scope"
    result.loc[ambiguous_name, "rejection_category"] = "mixture_or_unknown_identity"
    result.loc[missing_structure, "rejection_category"] = "missing_or_unparseable_structure"
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("rejection_category", dropna=False, as_index=False)
        .agg(
            n_rows=(DEFAULT_SCHEMA.chemical_id, "size"),
            n_chemicals=(DEFAULT_SCHEMA.chemical_id, "nunique"),
            n_missing_or_unparseable_structure=("missing_or_unparseable_structure", "sum"),
            n_mixture_or_unknown_identity=("mixture_or_unknown_identity", "sum"),
            n_inorganic_or_metal_scope=("inorganic_or_metal_scope", "sum"),
            n_salt_like_scope_flag=("salt_like_scope_flag", "sum"),
        )
        .sort_values(["n_rows", "rejection_category"], ascending=[False, True])
        .reset_index(drop=True)
    )


def summarize_triggers(frame: pd.DataFrame) -> pd.DataFrame:
    trigger_columns = [
        "missing_or_unparseable_structure",
        "mixture_or_unknown_identity",
        "inorganic_or_metal_scope",
        "salt_like_scope_flag",
    ]
    rows = []
    for column in trigger_columns:
        mask = frame[column].astype(bool)
        rows.append(
            {
                "trigger": column,
                "n_rows": int(mask.sum()),
                "n_chemicals": int(
                    frame.loc[mask, DEFAULT_SCHEMA.chemical_id].nunique()
                ),
                "counting_rule": "overlapping trigger count",
            }
        )
    rows.append(
        {
            "trigger": "unique_deterministic_rejection_total",
            "n_rows": int(len(frame)),
            "n_chemicals": int(frame[DEFAULT_SCHEMA.chemical_id].nunique()),
            "counting_rule": "unique total",
        }
    )
    return pd.DataFrame(rows)


def data_flow_summary(data: pd.DataFrame) -> pd.DataFrame:
    rejected = data[DEFAULT_SCHEMA.hard_ood].fillna(False).astype(bool)
    target_available = data[DEFAULT_SCHEMA.target].notna()
    categories = [
        (
            "Scoreable structured benchmark",
            ~rejected & target_available,
            "Quantitative target available and no deterministic rejection trigger",
        ),
        (
            "Deterministic rejection",
            rejected,
            "Identity or molecular-representation failure",
        ),
        (
            "Other quantitative-target exclusions",
            ~rejected & ~target_available,
            "No harmonized quantitative log10(mol L-1) target",
        ),
    ]
    rows = [
        {
            "subset": label,
            "n_rows": int(mask.sum()),
            "n_chemicals": int(data.loc[mask, DEFAULT_SCHEMA.chemical_id].nunique()),
            "definition": definition,
        }
        for label, mask, definition in categories
    ]
    if sum(row["n_rows"] for row in rows) != len(data):
        raise RuntimeError("Mutually exclusive data-flow categories do not exhaust the dataset.")
    rows.append(
        {
            "subset": "Broader curated dataset",
            "n_rows": int(len(data)),
            "n_chemicals": int(data[DEFAULT_SCHEMA.chemical_id].nunique()),
            "definition": "Unique total",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit deterministic withhold/review cases excluded from the structured organic benchmark."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    rejected = data.loc[
        data[DEFAULT_SCHEMA.hard_ood].fillna(False).astype(bool)
    ].copy()
    classified = classify_deterministic_rejections(rejected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        DEFAULT_SCHEMA.chemical_id,
        DEFAULT_SCHEMA.chemical_name,
        DEFAULT_SCHEMA.casrn,
        DEFAULT_SCHEMA.chemical_class,
        DEFAULT_SCHEMA.smiles,
        "rejection_category",
        "deterministic_action",
        "missing_or_unparseable_structure",
        "mixture_or_unknown_identity",
        "inorganic_or_metal_scope",
        "salt_like_scope_flag",
    ]
    classified[columns].drop_duplicates().to_csv(
        args.output_dir / "deterministic_rejection_cases.csv", index=False
    )
    summary = summarize(classified)
    summary.to_csv(args.output_dir / "deterministic_rejection_summary.csv", index=False)
    summarize_triggers(classified).to_csv(
        args.output_dir / "deterministic_rejection_trigger_summary.csv",
        index=False,
    )
    data_flow_summary(data).to_csv(
        args.output_dir / "curated_data_flow_summary.csv",
        index=False,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
