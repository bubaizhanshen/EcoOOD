from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, apply_publication_style, save_figure


def _arrow(ax, start, end, color=PALETTE["ink"], lw=1.6, style="-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
        )
    )


def _text_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    *,
    fc,
    title_size=11,
    subtitle_size=7.3,
    weight="bold",
    align="center",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=PALETTE["ink"],
    )
    ax.add_patch(patch)
    tx = x + (w / 2 if align == "center" else 0.04)
    ha = "center" if align == "center" else "left"
    ax.text(tx, y + h * 0.62, title, ha=ha, va="center", fontsize=title_size, color=PALETTE["ink"], weight=weight)
    if subtitle:
        ax.text(
            tx,
            y + h * 0.28,
            textwrap.fill(subtitle, 26 if w < 0.18 else 36),
            ha=ha,
            va="center",
            fontsize=subtitle_size,
            color=PALETTE["slate"],
        )


def _section_label(ax, x, y, text) -> None:
    ax.text(x, y, text, fontsize=8.8, weight="bold", color=PALETTE["slate"], ha="left", va="bottom")


def figure1_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(ACS_DOUBLE_WIDTH, 4.55), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "EcoOOD data assembly, reliability assessment, and screening application", fontsize=10.7, weight="bold", color=PALETTE["ink"])
    ax.text(
        0.05,
        0.91,
        "Public ecotoxicity records support prediction, joint reliability scoring, and chemical-level screening actions.",
        fontsize=7.3,
        color=PALETTE["slate"],
        ha="left",
        va="top",
    )
    _section_label(ax, 0.05, 0.82, "Data fusion")

    _text_box(ax, 0.05, 0.63, 0.16, 0.14, "ECOTOX", "acute aquatic records", fc=PALETTE["mist"], title_size=12)
    _text_box(ax, 0.24, 0.63, 0.19, 0.14, "DSSTox / CompTox", "structure and physicochemical metadata", fc=PALETTE["mint"], title_size=10.6)
    _text_box(ax, 0.46, 0.63, 0.19, 0.14, "invitrodb / ToxCast", "summary-level bioactivity-proxy features", fc=PALETTE["blush"], title_size=10.5)
    _text_box(ax, 0.68, 0.63, 0.21, 0.14, "Curated EcoOOD table", "acute aquatic benchmark with structured and hard-OOD records", fc="#F5E8C8", title_size=10.8)

    _arrow(ax, (0.21, 0.70), (0.23, 0.70), color=PALETTE["ink"])
    _arrow(ax, (0.43, 0.70), (0.45, 0.70), color=PALETTE["ink"])
    _arrow(ax, (0.65, 0.70), (0.67, 0.70), color=PALETTE["ink"])

    counts = [
        ("6381", "records"),
        ("1000", "chemicals"),
        ("4964", "structured rows"),
        ("1093", "hard-OOD rows"),
    ]
    ribbon_y = 0.54
    ribbon_x = [0.05, 0.27, 0.49, 0.71]
    for (value, label), x in zip(counts, ribbon_x, strict=False):
        rect = FancyBboxPatch(
            (x, ribbon_y),
            0.18,
            0.065,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.8,
            facecolor="white",
            edgecolor=PALETTE["grid"],
        )
        ax.add_patch(rect)
        ax.text(x + 0.09, ribbon_y + 0.043, value, ha="center", va="center", fontsize=8.2, fontweight="bold", color=PALETTE["ink"])
        ax.text(x + 0.09, ribbon_y + 0.017, label, ha="center", va="center", fontsize=6.6, color=PALETTE["slate"])

    _section_label(ax, 0.05, 0.48, "Benchmark split family")
    split_fc = "#EDE8DD"
    _text_box(ax, 0.05, 0.31, 0.12, 0.10, "Random", "row-level interpolation", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.19, 0.31, 0.11, 0.10, "Scaffold", "structural novelty", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.32, 0.31, 0.11, 0.10, "Temporal", "later-year deployment", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.45, 0.31, 0.11, 0.10, "Species", "taxonomic shift", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.58, 0.31, 0.13, 0.10, "Class holdout", "new contaminant class", fc=split_fc, title_size=8.8, subtitle_size=6.3)
    _text_box(ax, 0.74, 0.31, 0.12, 0.10, "Hard OOD", "explicit reject set", fc="#F3E3DA", title_size=8.8, subtitle_size=6.4)
    split_centers = [0.11, 0.245, 0.375, 0.505, 0.645, 0.80]
    ax.plot([0.11, 0.93], [0.455, 0.455], color=PALETTE["ink"], linewidth=1.15)
    _arrow(ax, (0.89, 0.68), (0.93, 0.455), color=PALETTE["ink"], lw=1.15)
    for x in split_centers:
        _arrow(ax, (x, 0.455), (x, 0.415), color=PALETTE["ink"], lw=1.15)

    _section_label(ax, 0.05, 0.12, "Deployment outputs")
    _text_box(ax, 0.05, 0.02, 0.22, 0.08, "Toxicity predictor", "point prediction plus conformal interval", fc=PALETTE["mint"], title_size=8.8, subtitle_size=6.4)
    _text_box(ax, 0.36, 0.02, 0.20, 0.08, "EcoOOD reliability", "joint prediction support and risk ordering", fc=PALETTE["sand"], title_size=8.8, subtitle_size=6.4)
    _text_box(ax, 0.65, 0.02, 0.24, 0.08, "Screening actions", "screen now, lower priority, withhold/review, prioritize testing", fc=PALETTE["blush"], title_size=8.8, subtitle_size=6.0)
    _arrow(ax, (0.27, 0.06), (0.35, 0.06), color=PALETTE["blue"], lw=1.5)
    _arrow(ax, (0.56, 0.06), (0.64, 0.06), color=PALETTE["red"], lw=1.5)

    save_figure(fig, output_dir, "Figure_1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EcoOOD workflow schematic figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/release_tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    figure1_pipeline(args.output_dir)


if __name__ == "__main__":
    main()
