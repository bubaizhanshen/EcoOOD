from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from ecoood.pipeline import run_benchmark


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_name(value: object) -> str:
    text = str(value).strip() or "missing"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:80]


def _aggregate_benchmark_summaries(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    metric_cols = [col for col in combined.columns if col not in {"seed", "group", "split", "model"}]
    aggregated = (
        combined.groupby(["group", "split", "model"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregated.columns
    ]
    return combined, aggregated


def _aggregate_ood_summaries(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    metric_cols = [col for col in combined.columns if col not in {"seed", "group", "split", "model", "method"}]
    aggregated = (
        combined.groupby(["group", "split", "model", "method"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregated.columns
    ]
    return combined, aggregated


def _run_group(
    df: pd.DataFrame,
    group_name: str,
    *,
    seeds: list[int],
    splits: list[str],
    models: list[str],
    output_dir: Path,
    alpha: float,
    members: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    benchmark_rows: list[pd.DataFrame] = []
    ood_rows: list[pd.DataFrame] = []
    for seed in seeds:
        run_dir = output_dir / "seeds" / f"seed_{seed}" / group_name
        summary = run_benchmark(
            df=df,
            splits=splits,
            models=models,
            output_dir=str(run_dir),
            alpha=alpha,
            seed=seed,
            n_members=members,
        )
        summary = summary.copy()
        summary["seed"] = seed
        summary["group"] = group_name
        benchmark_rows.append(summary)

        ood_summary = pd.read_csv(run_dir / "ood_score_summary.csv")
        ood_summary["seed"] = seed
        ood_summary["group"] = group_name
        ood_rows.append(ood_summary)
    return benchmark_rows, ood_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed EcoOOD benchmark sweeps.")
    parser.add_argument("--data", type=Path, required=True, help="CSV or Parquet dataset path.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "scaffold", "temporal", "chemical_class", "species", "hard_ood"],
    )
    parser.add_argument("--models", nargs="+", default=["lightgbm", "random_forest"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--group-col", type=str, default=None, help="Optional column for endpoint-wise sweeps.")
    parser.add_argument("--min-group-rows", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/seed_sweep"))
    args = parser.parse_args()

    df = _load_table(args.data)
    groups: list[tuple[str, pd.DataFrame]] = [("all", df)]
    if args.group_col:
        grouped: list[tuple[str, pd.DataFrame]] = []
        for value, frame in df.groupby(args.group_col, dropna=False):
            if len(frame) < args.min_group_rows:
                continue
            grouped.append((_safe_name(value), frame.reset_index(drop=True)))
        groups = grouped

    all_benchmark_frames: list[pd.DataFrame] = []
    all_ood_frames: list[pd.DataFrame] = []
    for group_name, group_df in groups:
        benchmark_frames, ood_frames = _run_group(
            group_df.reset_index(drop=True),
            group_name,
            seeds=args.seeds,
            splits=args.splits,
            models=args.models,
            output_dir=args.output_dir,
            alpha=args.alpha,
            members=args.members,
        )
        all_benchmark_frames.extend(benchmark_frames)
        all_ood_frames.extend(ood_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if all_benchmark_frames:
        benchmark_all, benchmark_agg = _aggregate_benchmark_summaries(all_benchmark_frames)
        benchmark_all.to_csv(args.output_dir / "benchmark_summary_all_seeds.csv", index=False)
        benchmark_agg.to_csv(args.output_dir / "benchmark_summary_agg.csv", index=False)
        print(benchmark_agg)
    if all_ood_frames:
        ood_all, ood_agg = _aggregate_ood_summaries(all_ood_frames)
        ood_all.to_csv(args.output_dir / "ood_score_summary_all_seeds.csv", index=False)
        ood_agg.to_csv(args.output_dir / "ood_score_summary_agg.csv", index=False)


if __name__ == "__main__":
    main()
