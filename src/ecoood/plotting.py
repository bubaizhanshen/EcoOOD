from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


ACS_SINGLE_WIDTH = 3.35
ACS_DOUBLE_WIDTH = 7.08
ACS_MAX_HEIGHT = 9.25

PALETTE = {
    "ink": "#1F1F1F",
    "paper": "#FCFBF8",
    "grid": "#DDD6CB",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#7A68A6",
    "rose": "#CC79A7",
    "slate": "#5E6C84",
    "sand": "#F3E8CC",
    "mist": "#E8F0F7",
    "mint": "#E3F2EB",
    "blush": "#F7E3DD",
}


def _register_local_arial() -> None:
    font_dir = Path.home() / ".local" / "share" / "fonts" / "arial"
    for font_name in ["Arial.TTF", "Arialbd.TTF", "Arialbi.TTF", "Ariali.TTF"]:
        font_path = font_dir / font_name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def apply_publication_style() -> None:
    _register_local_arial()
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.linewidth": 0.8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 7.5,
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis=grid_axis)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.16, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def _figure_format_dirs(output_dir: Path) -> tuple[Path, Path]:
    vector_dir = output_dir / "vector"
    raster_dir = output_dir / "raster"
    vector_dir.mkdir(parents=True, exist_ok=True)
    raster_dir.mkdir(parents=True, exist_ok=True)
    return vector_dir, raster_dir


def sync_saved_figure(output_dir: Path, stem: str, *, include_svg: bool = False) -> None:
    vector_dir, raster_dir = _figure_format_dirs(output_dir)
    sync_plan = [
        (output_dir / f"{stem}.pdf", vector_dir / f"{stem}.pdf"),
        (output_dir / f"{stem}.png", raster_dir / f"{stem}.png"),
    ]
    if include_svg:
        sync_plan.append((output_dir / f"{stem}.svg", vector_dir / f"{stem}.svg"))

    for src, dst in sync_plan:
        if src.exists():
            shutil.copy2(src, dst)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir, raster_dir = _figure_format_dirs(output_dir)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(raster_dir / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(vector_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
