from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, add_panel_label, apply_publication_style, finish_axis, save_figure


PALETTE_SPLIT = {
    "chemical_class": PALETTE["green"],
    "species": PALETTE["red"],
    "temporal": PALETTE["blue"],
    "hard_ood": PALETTE["slate"],
}
LABELS = {
    "chemical_class": "Chemical class",
    "species": "Species",
    "temporal": "Temporal",
    "hard_ood": "Hard OOD",
}


def _plot_line_with_band(ax: plt.Axes, frame: pd.DataFrame, metric: str, color: str, label: str) -> None:
    x = frame["shots"].astype(int)
    mean = frame[f"{metric}_mean"]
    std = frame[f"{metric}_std"].fillna(0.0)
    ax.plot(x, mean, marker="o", color=color, label=label, linewidth=1.8)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style Figure 7 from few-shot recalibration summaries.")
    parser.add_argument(
        "--structured-summary",
        type=Path,
        default=Path("outputs/release_tables/fewshot_recalibration_summary.csv"),
    )
    parser.add_argument(
        "--hard-summary",
        type=Path,
        default=Path("outputs/release_tables/fewshot_recalibration_hard_ood_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/release_tables"))
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    apply_publication_style()
    structured = pd.read_csv(args.structured_summary)
    hard = pd.read_csv(args.hard_summary)

    fig = plt.figure(figsize=(ACS_DOUBLE_WIDTH, 5.35), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.34)
    ax_cov = fig.add_subplot(gs[0, 0])
    ax_risk = fig.add_subplot(gs[0, 1])
    ax_hard_cov = fig.add_subplot(gs[1, 0])
    ax_hard_risk = fig.add_subplot(gs[1, 1])

    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.08, right=0.985)

    for split in ["chemical_class", "species", "temporal"]:
        frame = structured[structured["split"] == split].copy().sort_values("shots")
        _plot_line_with_band(ax_cov, frame, "coverage_after", PALETTE_SPLIT[split], LABELS[split])
        _plot_line_with_band(ax_risk, frame, "aurc_after", PALETTE_SPLIT[split], LABELS[split])
        baseline_cov = frame.loc[frame["shots"] == 0, "coverage_before_mean"].iloc[0]
        baseline_aurc = frame.loc[frame["shots"] == 0, "aurc_before_mean"].iloc[0]
        ax_cov.scatter([0], [baseline_cov], color=PALETTE_SPLIT[split], s=22, zorder=3)
        ax_risk.scatter([0], [baseline_aurc], color=PALETTE_SPLIT[split], s=22, zorder=3)

    hard = hard.sort_values("shots")
    _plot_line_with_band(ax_cov, hard, "coverage_after", PALETTE_SPLIT["hard_ood"], LABELS["hard_ood"])
    _plot_line_with_band(ax_risk, hard, "aurc_after", PALETTE_SPLIT["hard_ood"], LABELS["hard_ood"])
    hard_base_cov = hard.loc[hard["shots"] == 0, "coverage_before_mean"].iloc[0]
    hard_base_aurc = hard.loc[hard["shots"] == 0, "aurc_before_mean"].iloc[0]
    ax_cov.scatter([0], [hard_base_cov], color=PALETTE_SPLIT["hard_ood"], s=24, zorder=3)
    ax_risk.scatter([0], [hard_base_aurc], color=PALETTE_SPLIT["hard_ood"], s=24, zorder=3)

    ax_cov.axhline(1 - args.alpha, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    for split in ["chemical_class", "species", "temporal"]:
        frame = structured[structured["split"] == split].copy().sort_values("shots")
        _plot_line_with_band(ax_hard_cov, frame, "target_coverage_gap_after", PALETTE_SPLIT[split], LABELS[split])
        _plot_line_with_band(ax_hard_risk, frame, "mean_interval_width_after", PALETTE_SPLIT[split], LABELS[split])
        base_gap = frame.loc[frame["shots"] == 0, "target_coverage_gap_before_mean"].iloc[0]
        base_width = frame.loc[frame["shots"] == 0, "mean_interval_width_before_mean"].iloc[0]
        ax_hard_cov.scatter([0], [base_gap], color=PALETTE_SPLIT[split], s=22, zorder=3)
        ax_hard_risk.scatter([0], [base_width], color=PALETTE_SPLIT[split], s=22, zorder=3)

    _plot_line_with_band(ax_hard_cov, hard, "target_coverage_gap_after", PALETTE_SPLIT["hard_ood"], LABELS["hard_ood"])
    _plot_line_with_band(ax_hard_risk, hard, "mean_interval_width_after", PALETTE_SPLIT["hard_ood"], LABELS["hard_ood"])
    hard_base_gap = hard.loc[hard["shots"] == 0, "target_coverage_gap_before_mean"].iloc[0]
    hard_base_width = hard.loc[hard["shots"] == 0, "mean_interval_width_before_mean"].iloc[0]
    ax_hard_cov.scatter([0], [hard_base_gap], color=PALETTE_SPLIT["hard_ood"], s=24, zorder=3)
    ax_hard_risk.scatter([0], [hard_base_width], color=PALETTE_SPLIT["hard_ood"], s=24, zorder=3)

    ax_hard_cov.axhline(0.0, color=PALETTE["ink"], linestyle="--", linewidth=0.9)

    ax_cov.set_title("Coverage across held-out domains", pad=6, fontsize=8.6)
    ax_cov.set_xlabel("New-domain labeled samples")
    ax_cov.set_ylabel("90% interval coverage")
    add_panel_label(ax_cov, "A", x=-0.18, y=1.08)
    finish_axis(ax_cov, grid_axis="y")

    ax_risk.set_title("Risk ordering across held-out domains", pad=6, fontsize=8.6)
    ax_risk.set_xlabel("New-domain labeled samples")
    ax_risk.set_ylabel("AURC (lower is better)")
    add_panel_label(ax_risk, "B", x=-0.18, y=1.08)
    finish_axis(ax_risk, grid_axis="y")

    ax_hard_cov.set_title("Target-coverage gap after recalibration", pad=6, fontsize=8.4)
    ax_hard_cov.set_xlabel("New-domain labeled samples")
    ax_hard_cov.set_ylabel("Coverage gap")
    add_panel_label(ax_hard_cov, "C", x=-0.15, y=1.08)
    finish_axis(ax_hard_cov, grid_axis="y")

    ax_hard_risk.set_title("Interval-width expansion after recalibration", pad=6, fontsize=8.4)
    ax_hard_risk.set_xlabel("New-domain labeled samples")
    ax_hard_risk.set_ylabel("Mean interval width")
    add_panel_label(ax_hard_risk, "D", x=-0.18, y=1.08)
    finish_axis(ax_hard_risk, grid_axis="y")

    handles, labels = ax_cov.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False, fontsize=6.8)
    save_figure(fig, args.output_dir, "Figure_6")

    supp = plt.figure(figsize=(ACS_DOUBLE_WIDTH, 3.25), constrained_layout=False)
    supp_gs = supp.add_gridspec(1, 2, wspace=0.26)
    ax_gap = supp.add_subplot(supp_gs[0, 0])
    ax_width = supp.add_subplot(supp_gs[0, 1])
    supp.subplots_adjust(top=0.84, bottom=0.16, left=0.08, right=0.985)

    for split in ["chemical_class", "species", "temporal"]:
        frame = structured[structured["split"] == split].copy().sort_values("shots")
        _plot_line_with_band(ax_gap, frame, "target_coverage_gap_after", PALETTE_SPLIT[split], LABELS[split])
        _plot_line_with_band(ax_width, frame, "mean_interval_width_after", PALETTE_SPLIT[split], LABELS[split])
        base_gap = frame.loc[frame["shots"] == 0, "target_coverage_gap_before_mean"].iloc[0]
        base_width = frame.loc[frame["shots"] == 0, "mean_interval_width_before_mean"].iloc[0]
        ax_gap.scatter([0], [base_gap], color=PALETTE_SPLIT[split], s=22, zorder=3)
        ax_width.scatter([0], [base_width], color=PALETTE_SPLIT[split], s=22, zorder=3)

    _plot_line_with_band(ax_gap, hard, "target_coverage_gap_after", PALETTE_SPLIT["hard_ood"], LABELS["hard_ood"])
    _plot_line_with_band(ax_width, hard, "mean_interval_width_after", PALETTE_SPLIT["hard_ood"], LABELS["hard_ood"])
    hard_base_gap = hard.loc[hard["shots"] == 0, "target_coverage_gap_before_mean"].iloc[0]
    hard_base_width = hard.loc[hard["shots"] == 0, "mean_interval_width_before_mean"].iloc[0]
    ax_gap.scatter([0], [hard_base_gap], color=PALETTE_SPLIT["hard_ood"], s=24, zorder=3)
    ax_width.scatter([0], [hard_base_width], color=PALETTE_SPLIT["hard_ood"], s=24, zorder=3)

    ax_gap.axhline(0.0, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    ax_gap.set_title("Target-coverage gap after recalibration", pad=6, fontsize=8.5)
    ax_gap.set_xlabel("New-domain labeled samples")
    ax_gap.set_ylabel("Coverage gap")
    add_panel_label(ax_gap, "A", x=-0.18, y=1.08)
    finish_axis(ax_gap, grid_axis="y")

    ax_width.set_title("Interval-width expansion after recalibration", pad=6, fontsize=8.5)
    ax_width.set_xlabel("New-domain labeled samples")
    ax_width.set_ylabel("Mean interval width")
    add_panel_label(ax_width, "B", x=-0.18, y=1.08)
    finish_axis(ax_width, grid_axis="y")

    supp_handles, supp_labels = ax_gap.get_legend_handles_labels()
    supp.legend(supp_handles, supp_labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False, fontsize=6.8)
    save_figure(supp, args.output_dir, "Figure_archived_recalibration_mechanics")


if __name__ == "__main__":
    main()
