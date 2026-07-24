from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from ecoood.pipeline import ExperimentConfig, run_single_experiment


DEFAULT_STRUCTURED = Path("data/processed/ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv")
DEFAULT_SPLITS = [
    "random",
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
]
DEFAULT_SEEDS = [40, 41, 42, 43, 44]

CONFIGS = {
    "compact": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
    },
    "reference": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
    },
    "larger": {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
    },
}


def _quiet_rdkit() -> None:
    warnings.filterwarnings("ignore")
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def _aggregate(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    metric_cols = [
        col
        for col in combined.columns
        if col not in set(keys) | {"seed", "config_params_json"}
    ]
    aggregated = combined.groupby(keys, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    aggregated.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregated.columns
    ]
    return combined, aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM configuration sensitivity for EcoOOD.")
    parser.add_argument("--structured-data", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lgbm_config_sensitivity"))
    args = parser.parse_args()

    _quiet_rdkit()
    structured_df = pd.read_csv(args.structured_data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    notes: list[dict[str, object]] = []

    for config_name, params in CONFIGS.items():
        for seed in args.seeds:
            for split in args.splits:
                run_dir = args.output_dir / "runs" / config_name / f"seed_{seed}" / split
                config = ExperimentConfig(
                    split=split,
                    model_name="lightgbm",
                    alpha=args.alpha,
                    seed=seed,
                    n_members=args.members,
                    output_dir=str(run_dir),
                    estimator_params=params,
                    ensemble_n_jobs=args.ensemble_n_jobs,
                )
                metrics, _, score_summary = run_single_experiment(structured_df, config=config)
                row = {
                    **metrics,
                    "config": config_name,
                    "seed": seed,
                    "n_members": args.members,
                    "config_params_json": json.dumps(params, sort_keys=True),
                }
                benchmark_frames.append(pd.DataFrame([row]))

                score_summary = score_summary.copy()
                score_summary["config"] = config_name
                score_summary["seed"] = seed
                score_summary["n_members"] = args.members
                score_summary["config_params_json"] = json.dumps(params, sort_keys=True)
                score_frames.append(score_summary)
                notes.append({"config": config_name, "seed": seed, "split": split, "params": params})
                print(f"completed config={config_name} seed={seed} split={split}", flush=True)

    benchmark_all, benchmark_agg = _aggregate(benchmark_frames, ["config", "split", "model", "n_members"])
    score_all, score_agg = _aggregate(score_frames, ["config", "split", "model", "method", "n_members"])

    benchmark_all.to_csv(args.output_dir / "benchmark_summary_all.csv", index=False)
    benchmark_agg.to_csv(args.output_dir / "benchmark_summary_agg.csv", index=False)
    score_all.to_csv(args.output_dir / "ood_score_summary_all.csv", index=False)
    score_agg.to_csv(args.output_dir / "ood_score_summary_agg.csv", index=False)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(notes, indent=2) + "\n")

    key_cols = [
        "config",
        "split",
        "rmse_mean",
        "coverage_mean",
        "aurc_mean",
        "predict_fraction_mean",
    ]
    print(benchmark_agg[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()
