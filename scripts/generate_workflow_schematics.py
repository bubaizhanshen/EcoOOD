from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, apply_publication_style, save_figure, sync_saved_figure


def _box(ax, x, y, w, h, text, fc, ec=PALETTE["ink"], fontsize=12, weight="normal") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=PALETTE["ink"], weight=weight)


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


def figure1_concept(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(ACS_DOUBLE_WIDTH, 4.05), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "Reliability shifts from static chemical AD to joint deployment novelty", fontsize=10.7, weight="bold", color=PALETTE["ink"])
    ax.text(
        0.05,
        0.91,
        "The deployment question is no longer only whether the chemistry looks familiar, but whether the prediction remains trustworthy in joint chemical-species-context-mechanism space.",
        fontsize=7.3,
        color=PALETTE["slate"],
        ha="left",
        va="top",
    )
    _section_label(ax, 0.05, 0.80, "Legacy framing")
    _section_label(ax, 0.37, 0.80, "Deployment novelty axes")
    _section_label(ax, 0.78, 0.80, "Decision output")

    _text_box(
        ax,
        0.05,
        0.51,
        0.23,
        0.22,
        "Traditional AD",
        "asks whether the chemical resembles the training chemistry",
        fc=PALETTE["mist"],
        title_size=12,
    )
    _text_box(
        ax,
        0.05,
        0.20,
        0.23,
        0.18,
        "Trust decision",
        "can this prediction be used, warned, or rejected at deployment time?",
        fc="#F0ECE4",
        title_size=10.8,
    )

    _text_box(ax, 0.37, 0.58, 0.16, 0.15, "Chemical OOD", "new scaffold or contaminant class", fc=PALETTE["blush"], title_size=10.5)
    _text_box(ax, 0.56, 0.58, 0.16, 0.15, "Biological OOD", "unseen species, genus, or trophic group", fc=PALETTE["mint"], title_size=10.5)
    _text_box(ax, 0.37, 0.37, 0.16, 0.15, "Context OOD", "exposure duration, medium, year, or study shift", fc=PALETTE["sand"], title_size=10.5)
    _text_box(ax, 0.56, 0.37, 0.16, 0.15, "Mechanistic OOD", "bioactivity profile outside the training mechanism space", fc="#ECE7F5", title_size=10.3)

    _text_box(
        ax,
        0.78,
        0.49,
        0.17,
        0.25,
        "EcoOOD",
        "joint novelty score integrating chemistry, species, context, mechanism, and model uncertainty",
        fc="#F5E8C8",
        title_size=12,
    )

    _text_box(ax, 0.37, 0.12, 0.16, 0.14, "Predict", "low novelty and narrow interval", fc=PALETTE["mint"], title_size=10.5)
    _text_box(ax, 0.56, 0.12, 0.16, 0.14, "Warn", "moderate novelty or widening uncertainty", fc=PALETTE["sand"], title_size=10.5)
    _text_box(ax, 0.75, 0.12, 0.20, 0.14, "Diagnostic hold / prioritize testing", "high novelty or operationally out-of-scope chemistry", fc=PALETTE["blush"], title_size=10.3)

    _arrow(ax, (0.28, 0.62), (0.36, 0.65), color=PALETTE["blue"], lw=1.7)
    _arrow(ax, (0.28, 0.62), (0.55, 0.65), color=PALETTE["green"], lw=1.7)
    _arrow(ax, (0.28, 0.60), (0.36, 0.44), color=PALETTE["orange"], lw=1.7)
    _arrow(ax, (0.28, 0.60), (0.55, 0.44), color=PALETTE["purple"], lw=1.7)
    for start in [(0.53, 0.65), (0.72, 0.65), (0.53, 0.44), (0.72, 0.44)]:
        _arrow(ax, start, (0.78, 0.61), color=PALETTE["ink"], lw=1.25)
    _arrow(ax, (0.87, 0.48), (0.45, 0.27), color=PALETTE["green"], lw=1.5)
    _arrow(ax, (0.89, 0.45), (0.64, 0.27), color=PALETTE["orange"], lw=1.5)
    _arrow(ax, (0.91, 0.42), (0.84, 0.27), color=PALETTE["red"], lw=1.5)

    note = FancyBboxPatch(
        (0.74, 0.80),
        0.21,
        0.08,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.8,
        facecolor="white",
        edgecolor=PALETTE["grid"],
    )
    ax.add_patch(note)
    ax.text(
        0.845,
        0.84,
        "Low toxicity + high OOD =\nwithhold/review, not low concern",
        ha="center",
        va="center",
        fontsize=7.0,
        color=PALETTE["ink"],
    )

    save_figure(fig, output_dir, "Figure_archived_concept")


def figure2_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(ACS_DOUBLE_WIDTH, 4.55), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "EcoOOD benchmark assembly and deployment workflow", fontsize=10.7, weight="bold", color=PALETTE["ink"])
    ax.text(
        0.05,
        0.91,
        "The benchmark is defined by real ecotoxicity records, realistic held-out deployment domains, and an explicit operational reject set.",
        fontsize=7.3,
        color=PALETTE["slate"],
        ha="left",
        va="top",
    )
    _section_label(ax, 0.05, 0.82, "Data fusion")

    _text_box(ax, 0.05, 0.63, 0.16, 0.14, "ECOTOX", "acute aquatic records", fc=PALETTE["mist"], title_size=12)
    _text_box(ax, 0.24, 0.63, 0.19, 0.14, "DSSTox / CompTox", "structure and physicochemical metadata", fc=PALETTE["mint"], title_size=10.6)
    _text_box(ax, 0.46, 0.63, 0.19, 0.14, "invitrodb / ToxCast", "summary-level mechanism features", fc=PALETTE["blush"], title_size=10.5)
    _text_box(ax, 0.68, 0.63, 0.21, 0.14, "Curated EcoOOD table", "acute aquatic benchmark with structured and hard-OOD records", fc="#F5E8C8", title_size=10.8)

    _arrow(ax, (0.21, 0.70), (0.23, 0.70), color=PALETTE["ink"])
    _arrow(ax, (0.43, 0.70), (0.45, 0.70), color=PALETTE["ink"])
    _arrow(ax, (0.65, 0.70), (0.67, 0.70), color=PALETTE["ink"])

    counts = [
        ("6381", "records"),
        ("1000", "chemicals"),
        ("4942", "structured rows"),
        ("1088", "hard-OOD rows"),
    ]
    ribbon_y = 0.55
    ribbon_x = [0.08, 0.27, 0.46, 0.67]
    for (value, label), x in zip(counts, ribbon_x, strict=False):
        rect = FancyBboxPatch(
            (x, ribbon_y),
            0.14,
            0.055,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.8,
            facecolor="white",
            edgecolor=PALETTE["grid"],
        )
        ax.add_patch(rect)
        ax.text(x + 0.04, ribbon_y + 0.028, value, ha="left", va="center", fontsize=8.2, fontweight="bold", color=PALETTE["ink"])
        ax.text(x + 0.085, ribbon_y + 0.028, label, ha="left", va="center", fontsize=6.8, color=PALETTE["slate"])

    _section_label(ax, 0.05, 0.48, "Benchmark split family")
    split_fc = "#EDE8DD"
    _text_box(ax, 0.05, 0.31, 0.12, 0.10, "Random", "row-level interpolation", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.19, 0.31, 0.11, 0.10, "Scaffold", "structural novelty", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.32, 0.31, 0.11, 0.10, "Temporal", "later-year deployment", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.45, 0.31, 0.11, 0.10, "Species", "taxonomic shift", fc=split_fc, title_size=9.2, subtitle_size=6.5)
    _text_box(ax, 0.58, 0.31, 0.13, 0.10, "Class holdout", "new contaminant class", fc=split_fc, title_size=8.8, subtitle_size=6.3)
    _text_box(ax, 0.74, 0.31, 0.12, 0.10, "Hard OOD", "explicit reject set", fc="#F3E3DA", title_size=8.8, subtitle_size=6.4)
    for x in [0.11, 0.24, 0.37, 0.50, 0.64, 0.80]:
        _arrow(ax, (0.785, 0.63), (x, 0.42), color=PALETTE["ink"], lw=1.15)

    _section_label(ax, 0.05, 0.12, "Deployment outputs")
    _text_box(ax, 0.05, 0.02, 0.22, 0.08, "Multimodal predictor", "point estimate plus conformal interval", fc=PALETTE["mint"], title_size=8.8, subtitle_size=6.4)
    _text_box(ax, 0.36, 0.02, 0.20, 0.08, "EcoOOD score", "multi-axis novelty and risk ordering", fc=PALETTE["sand"], title_size=8.8, subtitle_size=6.4)
    _text_box(ax, 0.65, 0.02, 0.24, 0.08, "Decision layer", "screen now, lower priority, withhold/review, prioritize testing", fc=PALETTE["blush"], title_size=8.8, subtitle_size=6.0)
    _arrow(ax, (0.27, 0.06), (0.35, 0.06), color=PALETTE["blue"], lw=1.5)
    _arrow(ax, (0.56, 0.06), (0.64, 0.06), color=PALETTE["red"], lw=1.5)

    save_figure(fig, output_dir, "Figure_archived_workflow")


def copy_user_supplied_figure1(output_dir: Path) -> None:
    source_candidates = [
        output_dir / "_preview_figure1_fixed.png",
        output_dir / "4211_画板 1.tif",
    ]
    source_path = next((path for path in source_candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError("Unable to locate the user-supplied Figure 1 asset.")

    image = Image.open(source_path).convert("RGB")
    png_path = output_dir / "Figure_1.png"
    pdf_path = output_dir / "Figure_1.pdf"
    image.save(png_path)
    image.save(pdf_path, resolution=300.0)
    sync_saved_figure(output_dir, "Figure_1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EcoOOD workflow schematic figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/release_tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    copy_user_supplied_figure1(args.output_dir)
    figure2_pipeline(args.output_dir)


if __name__ == "__main__":
    main()
