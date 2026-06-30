from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ecoood.plotting import ACS_DOUBLE_WIDTH, add_panel_label, apply_publication_style, save_figure


PROFILE_ORDER = [
    "chemical_only",
    "chemical_species",
    "chemical_species_context",
    "chemical_species_context_mechanism",
    "current_full",
]
PROFILE_LABELS = {
    "chemical_only": "Chemical",
    "chemical_species": "Chemical +\nSpecies",
    "chemical_species_context": "Chemical + Species\n+ Context",
    "chemical_species_context_mechanism": "Chemical + Species\n+ Context + Mechanism",
    "current_full": "Current\nfull model",
}
SPLIT_ORDER = ["temporal", "species", "chemical_class"]
SPLIT_LABELS = {
    "temporal": "Temporal",
    "species": "Species",
    "chemical_class": "Class Holdout",
}


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(ACS_DOUBLE_WIDTH, 5.7), constrained_layout=True)
    specs = [
        ("rmse_mean", "RMSE", "YlOrRd_r"),
        ("coverage_mean", "Coverage", "YlGn"),
        ("aurc_mean", "AURC", "YlOrRd_r"),
    ]
    for idx, (ax, (metric, title, cmap)) in enumerate(zip(axes, specs)):
        pivot = (
            summary.pivot(index="profile", columns="split", values=metric)
            .reindex(index=PROFILE_ORDER, columns=SPLIT_ORDER)
            .rename(index=PROFILE_LABELS, columns=SPLIT_LABELS)
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            linewidths=0.8,
            linecolor="white",
            cbar=True,
            ax=ax,
            annot_kws={"fontsize": 7},
            cbar_kws={"shrink": 0.82, "pad": 0.02},
        )
        ax.set_title(title, pad=6)
        ax.set_xlabel("" if idx < len(specs) - 1 else "Benchmark split")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        add_panel_label(ax, chr(ord("A") + idx), x=-0.08, y=1.03)
    save_figure(fig, output_dir, "Figure_S1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append the current full-model benchmark to axis-ablation summaries.")
    parser.add_argument("--ablation-summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    apply_publication_style()

    ablation = pd.read_csv(args.ablation_summary)
    reference = pd.read_csv(args.reference_summary)
    reference = reference[reference["split"].isin(SPLIT_ORDER) & (reference["group"] == "all")].copy()
    reference["profile"] = "current_full"
    keep_cols = [col for col in ablation.columns if col in reference.columns or col in {"profile", "split", "model"}]
    augmented = pd.concat(
        [
            ablation,
            reference[keep_cols],
        ],
        ignore_index=True,
    )
    augmented = augmented.sort_values(["split", "profile"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output_dir / "axis_ablation_with_full_summary.csv", index=False)
    _plot(augmented, args.output_dir)
    print(augmented.to_string(index=False))


if __name__ == "__main__":
    main()
