from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoood.pipeline import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EcoOOD benchmark experiments.")
    parser.add_argument("--data", type=Path, required=True, help="CSV or Parquet dataset path.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "scaffold", "chemical_class", "species", "temporal", "hard_ood"],
    )
    parser.add_argument("--models", nargs="+", default=["lightgbm", "random_forest"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()

    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    summary = run_benchmark(
        df=df,
        splits=args.splits,
        models=args.models,
        output_dir=str(args.output_dir),
        alpha=args.alpha,
        seed=args.seed,
        n_members=args.members,
    )
    print(summary)


if __name__ == "__main__":
    main()
