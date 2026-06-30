from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoood.evaluation import selective_risk_summary


METHOD_COLUMNS = {
    "ecoood": "ecoood_score",
    "ad_similarity": "ad_similarity",
    "ad_distance_to_model": "ad_distance_to_model",
    "uncertainty_interval_width": "interval_width",
    "ood_mahalanobis": "ood_mahalanobis",
    "ood_isolation_forest": "ood_isolation_forest",
    "ood_lof": "ood_lof",
}


def _prediction_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/predictions.csv"))


def _split_and_model(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    return relative.parts[0], relative.parts[1]


def summarize_prediction_file(
    path: Path,
    root: Path,
    methods: list[str],
    abstain_fractions: list[float],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    split, model = _split_and_model(path, root)
    rows: list[pd.DataFrame] = []
    for method in methods:
        column = METHOD_COLUMNS[method]
        if column not in frame.columns:
            continue
        summary = selective_risk_summary(
            y_true=frame["y_true"].to_numpy(),
            y_pred=frame["y_pred"].to_numpy(),
            score=frame[column].to_numpy(),
            abstain_fractions=abstain_fractions,
        )
        summary.insert(0, "method", method)
        summary.insert(0, "model", model)
        summary.insert(0, "split", split)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_levels(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "retained_fraction",
        "rmse",
        "mae",
        "mean_abs_error",
        "catastrophic_error_rate",
        "catastrophic_error_reduction",
        "error_reduction",
    ]
    grouped = (
        frame.groupby(["split", "model", "method", "abstain_fraction"], dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize selective-risk tradeoffs at fixed abstention fractions.")
    parser.add_argument(
        "--benchmark-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/benchmark_1000chem_dsstox_mech_structured_ad"),
            Path("outputs/benchmark_1000chem_dsstox_mech_hard_ood_ad"),
        ],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "ecoood",
            "ad_similarity",
            "ad_distance_to_model",
            "uncertainty_interval_width",
            "ood_mahalanobis",
            "ood_isolation_forest",
            "ood_lof",
        ],
    )
    parser.add_argument(
        "--abstain-fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selective_risk"))
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for root in args.benchmark_dirs:
        for path in _prediction_paths(root):
            summary = summarize_prediction_file(
                path=path,
                root=root,
                methods=args.methods,
                abstain_fractions=args.abstain_fractions,
            )
            if not summary.empty:
                frames.append(summary)

    if not frames:
        raise SystemExit("No prediction files found for selective-risk summary.")

    combined = pd.concat(frames, ignore_index=True)
    aggregated = aggregate_levels(combined)
    fixed_20 = combined[combined["abstain_fraction"].round(6) == 0.2].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "selective_risk_all.csv", index=False)
    aggregated.to_csv(args.output_dir / "selective_risk_agg.csv", index=False)
    fixed_20.to_csv(args.output_dir / "selective_risk_abstain20.csv", index=False)
    print(aggregated.to_string(index=False))


if __name__ == "__main__":
    main()
