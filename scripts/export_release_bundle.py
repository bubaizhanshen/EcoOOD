from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split, named_class_for_seed


DEFAULT_SPLITS = [
    "random",
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
]
DEFAULT_MODELS = ["lightgbm", "random_forest", "xgboost"]
DEFAULT_SEEDS = [40, 41, 42, 43, 44]
PREDICTION_COLUMNS = [
    "chemical_id",
    "chemical_name",
    "casrn",
    "inchikey",
    "endpoint",
    "species",
    "chemical_class",
    "y_true",
    "y_pred",
    "model_std",
    "interval_lower",
    "interval_upper",
    "interval_width",
    "endpoint_interval_lower",
    "endpoint_interval_upper",
    "endpoint_interval_width",
    "ecoood_score",
    "ecoood_q80",
    "ecoood_q95",
    "ecoood_endpoint_balanced",
    "d_chem",
    "d_species",
    "d_context",
    "d_mech",
    "u_model",
    "ad_similarity",
    "ad_distance_to_model",
    "ad_equal_block_distance",
    "ensemble_sd_risk",
    "input_space_knn_plus_sd_risk",
    "equal_block_knn_plus_sd_risk",
    "generic_support_plus_sd_risk",
    "decision",
    "screening_action",
    "route",
]
SUPPLEMENTARY_TABLES = {
    "integrity_sensitivity_summary.csv": (
        "integrity_sensitivity_full/integrity_sensitivity_summary.csv"
    ),
    "configuration_sensitivity_summary.csv": (
        "lgbm_config_sensitivity/benchmark_summary_agg.csv"
    ),
    "fewshot_interval_recalibration_summary.csv": (
        "fewshot_recalibration/fewshot_recalibration_summary.csv"
    ),
    "species_chemical_overlap_summary.csv": (
        "species_chemical_overlap/species_chemical_overlap_summary.csv"
    ),
    "fixed_temporal_summary.csv": "temporal_block/benchmark_summary_agg.csv",
    "reference_holdout_summary.csv": (
        "reference_holdout/benchmark_summary_agg.csv"
    ),
    "reference_holdout_split_metadata.csv": (
        "reference_holdout/split_metadata.csv"
    ),
    "external_main_case_summary.csv": (
        "external_main/external_case_summary.csv"
    ),
    "external_strict_case_summary.csv": (
        "external_strict/external_case_summary.csv"
    ),
    "deterministic_rejection_summary.csv": (
        "deterministic_rejection_audit/deterministic_rejection_summary.csv"
    ),
    "deterministic_rejection_trigger_summary.csv": (
        "deterministic_rejection_audit/deterministic_rejection_trigger_summary.csv"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_release_columns(frame: pd.DataFrame) -> pd.DataFrame:
    # Keep public release tables aligned with the terminology used by the package API.
    replacements = {
        "catastrophic_error_capture_rate": "top_decile_error_capture_rate",
        "catastrophic_capture": "top_decile_error_capture",
        "catastrophic_error_reduction": "top_decile_error_rate_reduction",
        "catastrophic_error_rate": "top_decile_error_rate",
    }
    renamed = {}
    for column in frame.columns:
        normalized = str(column)
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        renamed[column] = normalized
    return frame.rename(columns=renamed)


def export_split_assignments(
    data: pd.DataFrame,
    destination: Path,
    *,
    seeds: list[int],
    splits: list[str],
) -> None:
    schema = DEFAULT_SCHEMA
    rows: list[pd.DataFrame] = []
    working = data.loc[data[schema.target].notna()].reset_index(drop=True)
    identity_columns = [
        column
        for column in [
            schema.chemical_id,
            schema.casrn,
            "inchikey",
            schema.endpoint,
            schema.species,
        ]
        if column in working.columns
    ]
    for seed in seeds:
        for split in splits:
            indices = build_split(working, split=split, schema=schema, seed=seed)
            for partition, values in [
                ("train", indices.train),
                ("calibration", indices.calib),
                ("test", indices.test),
            ]:
                frame = working.loc[values, identity_columns].copy()
                frame.insert(0, "row_id", values)
                frame.insert(0, "partition", partition)
                frame.insert(
                    0,
                    "held_out_group",
                    named_class_for_seed(seed)
                    if split == "chemical_class"
                    else "",
                )
                frame.insert(0, "split", split)
                frame.insert(0, "seed", seed)
                rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(destination, index=False)


def export_predictions(
    benchmark_root: Path,
    destination: Path,
    *,
    seeds: list[int],
    splits: list[str],
    models: list[str],
) -> None:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        for split in splits:
            for model in models:
                path = (
                    benchmark_root
                    / f"seed_{seed}"
                    / "structured"
                    / split
                    / model
                    / "predictions.csv"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                columns = [column for column in PREDICTION_COLUMNS if column in frame]
                frame = frame.loc[:, columns].copy()
                if "decision" in frame:
                    frame = frame.rename(
                        columns={"decision": "calibration_diagnostic_state"}
                    )
                if "d_mech" in frame:
                    frame = frame.rename(
                        columns={"d_mech": "d_bioactivity_proxy"}
                    )
                frame = normalize_release_columns(frame)
                frame.insert(0, "model", model)
                frame.insert(0, "split", split)
                frame.insert(0, "seed", seed)
                frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(destination, index=False)


def copy_analysis_tables(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.csv")):
        normalize_release_columns(pd.read_csv(path)).to_csv(
            destination / path.name,
            index=False,
        )


def copy_supplementary_tables(benchmark_root: Path, destination: Path) -> None:
    for destination_name, relative_source in SUPPLEMENTARY_TABLES.items():
        source = benchmark_root / relative_source
        if not source.exists():
            raise FileNotFoundError(source)
        normalize_release_columns(pd.read_csv(source)).to_csv(
            destination / destination_name,
            index=False,
        )


def write_bundle_readme(destination: Path, release_tag: str) -> None:
    text = f"""# EcoOOD analysis release {release_tag}

This is the frozen analysis bundle for the EcoOOD aquatic ecotoxicity
screening benchmark. It contains the derived scoreable table, fixed split
assignments, compact case-level predictions, and statistical audit tables.

## Contents

- `data/EcoOOD_benchmark_snapshot_structured.csv`: frozen scoreable benchmark
- `data/feature_manifest.csv`: predictor roles, cardinalities, and missingness
- `data/curated_data_flow_summary.csv`: scoreable and rejected record counts
- `data/split_assignments.csv`: train, calibration, and test assignments
- `predictions/predictions_core.csv`: compact case-level model and reliability outputs
- `tables/`: benchmark, fixed-workload, sensitivity, reference-fold, and
  external case-level audit tables
- `manifest.json`: file sizes and SHA-256 checksums

`calibration_diagnostic_state` in the prediction table records the internal
predict/warn/abstain diagnostic state. Chemical-level screening routes are
reported in the fixed-workload tables.

The source records were derived from public ECOTOX, DSSTox/CompTox,
invitrodb/ToxCast, ECHA, and PMRA resources. Consult those providers for the
original records and provider-specific terms. The derived release is intended
to reproduce the reported analyses.
"""
    (destination / "README.md").write_text(text, encoding="utf-8")


def write_manifest(destination: Path, release_tag: str) -> None:
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(destination)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "release": release_tag,
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the versioned EcoOOD analysis-data release bundle."
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    destination = args.output_dir / f"EcoOOD_analysis_{args.release_tag}"
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "data").mkdir(parents=True)
    (destination / "predictions").mkdir(parents=True)
    (destination / "tables").mkdir(parents=True)

    snapshot_destination = (
        destination / "data" / "EcoOOD_benchmark_snapshot_structured.csv"
    )
    shutil.copy2(args.snapshot, snapshot_destination)
    feature_manifest = args.benchmark_root / "integrity" / "feature_manifest.csv"
    if not feature_manifest.exists():
        raise FileNotFoundError(feature_manifest)
    shutil.copy2(feature_manifest, destination / "data" / "feature_manifest.csv")
    data_flow = (
        args.benchmark_root
        / "deterministic_rejection_audit"
        / "curated_data_flow_summary.csv"
    )
    if not data_flow.exists():
        raise FileNotFoundError(data_flow)
    shutil.copy2(
        data_flow,
        destination / "data" / "curated_data_flow_summary.csv",
    )
    data = pd.read_csv(args.snapshot)
    export_split_assignments(
        data,
        destination / "data" / "split_assignments.csv",
        seeds=args.seeds,
        splits=args.splits,
    )
    export_predictions(
        args.benchmark_root,
        destination / "predictions" / "predictions_core.csv",
        seeds=args.seeds,
        splits=args.splits,
        models=args.models,
    )
    analysis_tables = args.benchmark_root / "analysis_tables"
    copy_analysis_tables(analysis_tables, destination / "tables")
    copy_supplementary_tables(args.benchmark_root, destination / "tables")
    write_bundle_readme(destination, args.release_tag)
    write_manifest(destination, args.release_tag)
    archive = shutil.make_archive(
        str(destination),
        "zip",
        root_dir=destination.parent,
        base_dir=destination.name,
    )
    print(archive)


if __name__ == "__main__":
    main()
