from __future__ import annotations

import argparse
import hashlib
import textwrap
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle

from generate_gate_method_comparison import (
    aggregate_chemical_panel as aggregate_gate_panel,
    load_prediction_panel as load_gate_panel,
    representative_examples as select_gate_examples,
    summarize_gate_methods,
    summarize_gate_methods_by_split,
)
from generate_screening_gate_validation import (
    aggregate_chemical_panel as aggregate_screening_panel,
    classify_screening_actions,
    load_prediction_panel as load_screening_panel,
    summarize_screening_gate,
)
from ecoood.plotting import (
    ACS_DOUBLE_WIDTH,
    PALETTE,
    add_panel_label,
    apply_publication_style,
    finish_axis,
    save_figure,
    sync_saved_figure,
)


SPLIT_ORDER = ["random", "scaffold", "temporal", "species", "chemical_class", "hard_ood"]
SPLIT_LABELS = {
    "random": "Random",
    "scaffold": "Scaffold",
    "temporal": "Temporal",
    "species": "Species",
    "chemical_class": "Class Holdout",
    "hard_ood": "Hard OOD",
}
SPLIT_COLORS = {
    "scaffold": PALETTE["blue"],
    "temporal": PALETTE["green"],
    "species": PALETTE["slate"],
    "chemical_class": PALETTE["orange"],
    "hard_ood": PALETTE["red"],
}
METHOD_ORDER = ["ecoood", "ad_distance_to_model", "ad_similarity", "ad_leverage", "ad_range"]
METHOD_LABELS = {
    "ecoood": "EcoOOD",
    "ad_distance_to_model": "Input-space kNN",
    "ad_similarity": "Similarity AD",
    "ad_leverage": "Leverage AD",
    "ad_range": "Range AD",
}
GATE_METHOD_ORDER = ["ecoood", "distance_to_model", "similarity_ad"]
GATE_METHOD_LABELS = {
    "ecoood": "EcoOOD",
    "distance_to_model": "Input-space kNN",
    "similarity_ad": "Similarity AD",
}
MODEL_ORDER = ["random_forest", "lightgbm", "xgboost"]
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
PRIORITY_CASE_SPECS = [
    {
        "casrn_digits": "2439103",
        "label": "Dodine",
        "anchor": "Repeated daphnia/algae flags",
    },
    {
        "casrn_digits": "3380345",
        "label": "Triclosan",
        "anchor": "Wastewater antimicrobial",
    },
    {
        "casrn_digits": "119446683",
        "label": "Held-out conazole",
        "anchor": "Held-out conazole class",
    },
]
WORKFLOW_ORDER = ["baseline_only", "baseline_plus_gate"]
WORKFLOW_LABELS = {
    "baseline_only": "Baseline only",
    "baseline_plus_gate": "Baseline + EcoOOD rule",
}
WORKFLOW_COLORS = {
    "baseline_only": PALETTE["slate"],
    "baseline_plus_gate": PALETTE["blue"],
}
GATE_METHOD_COLORS = {
    "ecoood": PALETTE["blue"],
    "distance_to_model": PALETTE["green"],
    "similarity_ad": PALETTE["orange"],
}
ACTION_LABELS = {
    "screen_now": "Screen now",
    "prioritize_testing": "Prioritize testing",
    "withhold_review": "Withhold/review",
    "lower_priority": "Lower priority",
}
ACTION_COLORS = {
    "screen_now": PALETTE["green"],
    "prioritize_testing": PALETTE["orange"],
    "withhold_review": PALETTE["red"],
    "lower_priority": "#D8D1C5",
}
PRIORITY_ORDER = ["actionable_high_tox", "lower_priority", "withhold_review_low_tox", "prioritize_testing"]
PRIORITY_LABELS = {
    "actionable_high_tox": "Screen now",
    "lower_priority": "Lower priority",
    "low_priority": "Lower priority",
    "withhold_review_low_tox": "Withhold/review",
    "prioritize_testing": "Prioritize testing",
}
PRIORITY_COLORS = {
    "actionable_high_tox": PALETTE["green"],
    "lower_priority": "#5F94B3",
    "low_priority": "#5F94B3",
    "withhold_review_low_tox": PALETTE["red"],
    "prioritize_testing": PALETTE["orange"],
}
ROUTE_VALIDATION_ORDER = [
    "low_priority",
    "withhold_review_low_tox",
    "actionable_high_tox",
    "prioritize_testing",
]


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def priority_threshold_sensitivity(
    frame: pd.DataFrame,
    tox_quantiles: tuple[float, ...] = (0.20, 0.25, 0.30),
    ood_quantiles: tuple[float, ...] = (0.70, 0.75, 0.80),
) -> pd.DataFrame:
    baseline_tox = float(frame["y_pred"].quantile(0.25))
    baseline_ood = float(frame["ecoood_score"].quantile(0.75))
    baseline_mask = (frame["y_pred"] <= baseline_tox) & (frame["ecoood_score"] > baseline_ood)
    baseline_priority = set(frame.index[baseline_mask])
    baseline_top30 = set(
        frame.loc[baseline_mask]
        .sort_values(["ecoood_score", "y_pred"], ascending=[False, True])
        .head(30)
        .index
    )

    rows: list[dict[str, float | int]] = []
    for tox_q in tox_quantiles:
        tox_cut = float(frame["y_pred"].quantile(tox_q))
        for ood_q in ood_quantiles:
            ood_cut = float(frame["ecoood_score"].quantile(ood_q))
            mask = (frame["y_pred"] <= tox_cut) & (frame["ecoood_score"] > ood_cut)
            priority = set(frame.index[mask])
            top30 = set(
                frame.loc[mask]
                .sort_values(["ecoood_score", "y_pred"], ascending=[False, True])
                .head(30)
                .index
            )
            rows.append(
                {
                    "tox_quantile": tox_q,
                    "ood_quantile": ood_q,
                    "priority_count": int(mask.sum()),
                    "priority_jaccard_vs_baseline": _jaccard(priority, baseline_priority),
                    "top30_jaccard_vs_baseline": _jaccard(top30, baseline_top30),
                }
            )
    return pd.DataFrame(rows)


def route_threshold_profile(frame: pd.DataFrame) -> pd.DataFrame:
    profiles = {
        "relaxed": {"tox_q": 0.20, "ood_q": 0.70},
        "baseline": {"tox_q": 0.25, "ood_q": 0.75},
        "strict": {"tox_q": 0.30, "ood_q": 0.80},
    }
    rows: list[dict[str, float | str]] = []
    for profile_name, profile in profiles.items():
        tox_thresh = float(frame["y_pred"].quantile(profile["tox_q"]))
        ood_thresh = float(frame["ecoood_score"].quantile(profile["ood_q"]))
        routed = frame.copy()
        routed["priority_bucket"] = "low_priority"
        routed.loc[(routed["y_pred"] <= tox_thresh) & (routed["ecoood_score"] <= ood_thresh), "priority_bucket"] = "actionable_high_tox"
        routed.loc[(routed["y_pred"] <= tox_thresh) & (routed["ecoood_score"] > ood_thresh), "priority_bucket"] = "prioritize_testing"
        routed.loc[(routed["y_pred"] > tox_thresh) & (routed["ecoood_score"] > ood_thresh), "priority_bucket"] = "withhold_review_low_tox"
        rows.append(
            {
                "profile": profile_name,
                "toxicity_quantile": profile["tox_q"],
                "ood_quantile": profile["ood_q"],
                "screen_now_fraction": float((routed["priority_bucket"] == "actionable_high_tox").mean()),
                "lower_priority_fraction": float((routed["priority_bucket"] == "low_priority").mean()),
                "withhold_review_fraction": float((routed["priority_bucket"] == "withhold_review_low_tox").mean()),
                "prioritize_testing_fraction": float((routed["priority_bucket"] == "prioritize_testing").mean()),
            }
        )
    return pd.DataFrame(rows)


def figure_s5_threshold_sensitivity(
    route_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(ACS_DOUBLE_WIDTH, 2.8),
        gridspec_kw={"width_ratios": [0.95, 1.05]},
        constrained_layout=True,
    )

    route_mix = route_summary.set_index("profile").loc[
        ["relaxed", "baseline", "strict"],
        [
            "screen_now_fraction",
            "lower_priority_fraction",
            "withhold_review_fraction",
            "prioritize_testing_fraction",
        ],
    ]
    route_keys = [
        "actionable_high_tox",
        "lower_priority",
        "withhold_review_low_tox",
        "prioritize_testing",
    ]
    bottom = np.zeros(len(route_mix), dtype=float)
    for column, route in zip(route_mix.columns, route_keys, strict=True):
        values = route_mix[column].to_numpy(dtype=float)
        axes[0].bar(
            route_mix.index,
            values,
            bottom=bottom,
            color=PRIORITY_COLORS[route],
            edgecolor="white",
            linewidth=0.5,
            width=0.68,
            label=PRIORITY_LABELS[route],
        )
        bottom += values
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Fraction of evaluated predictions")
    axes[0].set_xlabel("Threshold profile")
    axes[0].set_title("Screening-action composition", pad=6)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, fontsize=6.6)
    finish_axis(axes[0], grid_axis="y")
    add_panel_label(axes[0], "A", x=-0.17, y=1.08)

    top30 = sensitivity.pivot(
        index="tox_quantile",
        columns="ood_quantile",
        values="top30_jaccard_vs_baseline",
    )
    sns.heatmap(
        top30,
        ax=axes[1],
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Jaccard overlap", "shrink": 0.82},
    )
    axes[1].set_xlabel("EcoOOD-score cutoff quantile")
    axes[1].set_ylabel("Predicted-toxicity cutoff quantile")
    axes[1].set_title("Top-30 priority-list stability", pad=6)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="y", rotation=0)
    add_panel_label(axes[1], "B", x=-0.18, y=1.08)

    save_figure(fig, output_dir, "Figure_S5")


def load_priority_case_studies(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(",", 8)
            if len(parts) != 9:
                raise ValueError(f"Unable to parse case-study row: {line}")
            row = dict(zip(header, parts))
            row["candidate_rows"] = int(row["candidate_rows"])
            row["unique_species"] = int(row["unique_species"])
            row["min_predicted_log_molar"] = float(row["min_predicted_log_molar"])
            row["max_ecoood_score"] = float(row["max_ecoood_score"])
            rows.append(row)
    return pd.DataFrame(rows)


def molecule_image(
    smiles: str,
    size: tuple[int, int] = (230, 145),
    cache_dir: Path | None = None,
) -> np.ndarray | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return np.asarray(Draw.MolToImage(mol, size=size))
    except ImportError:
        pass
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        data = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha1(f"{smiles}|{size[0]}|{size[1]}".encode("utf-8")).hexdigest()[:16]
            cache_path = cache_dir / f"{key}.png"
            if cache_path.exists():
                data = cache_path.read_bytes()
        if data is None:
            data = fetch_pubchem_png(smiles, size[0], size[1])
            if cache_dir is not None:
                cache_path.write_bytes(data)
    except Exception:
        return None
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"))


@lru_cache(maxsize=64)
def fetch_pubchem_png(smiles: str, width: int, height: int) -> bytes:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/PNG"
        f"?smiles={quote(smiles, safe='')}&image_size={width}x{height}"
    )
    with urlopen(url, timeout=20) as response:
        return response.read()


@lru_cache(maxsize=64)
def molecule_svg(smiles: str, width: int = 230, height: int = 145) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_action_chip(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    action: str,
    label: str,
    highlight: bool = False,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.2 if highlight else 0.8,
            edgecolor=PALETTE["ink"] if highlight else "white",
            facecolor=ACTION_COLORS[action],
        )
    )
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6.3,
        color=PALETTE["ink"],
        fontweight="bold" if highlight else "normal",
    )


def _svg_group_from_text(svg_text: str, x: float, y: float, width: float, height: float) -> str:
    root = ET.fromstring(svg_text)
    view_box = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(view_box) == 4:
        min_x, min_y, vb_w, vb_h = map(float, view_box)
    else:
        vb_w = float(str(root.attrib.get("width", 1)).replace("px", ""))
        vb_h = float(str(root.attrib.get("height", 1)).replace("px", ""))
        min_x = 0.0
        min_y = 0.0
    scale = min(width / vb_w, height / vb_h)
    tx = x + (width - vb_w * scale) / 2.0 - min_x * scale
    ty = y + (height - vb_h * scale) / 2.0 - min_y * scale
    inner = "".join(ET.tostring(child, encoding="unicode") for child in root)
    return f'<g transform="translate({tx:.3f},{ty:.3f}) scale({scale:.6f})">{inner}</g>'


def compose_figure4_vector_overlay(
    fig: plt.Figure,
    ax_cases: plt.Axes,
    examples: pd.DataFrame,
    output_dir: Path,
    base_svg_path: Path,
    tight_bbox,
) -> None:
    if examples.empty:
        return
    try:
        from svgutils.compose import Figure as SvgFigure, SVG
        import cairosvg
    except ImportError:
        return

    fig_w_in, fig_h_in = fig.get_size_inches()
    bbox = ax_cases.get_position()
    width_pt = tight_bbox.width * 72.0
    height_pt = tight_bbox.height * 72.0
    row_centers = np.linspace(0.78, 0.18, len(examples))
    overlay_elements: list[str] = []
    for y, (_, row) in zip(row_centers, examples.iterrows()):
        fx0 = bbox.x0 + bbox.width * 0.04
        fx1 = bbox.x0 + bbox.width * 0.28
        fy0 = bbox.y0 + bbox.height * (y - 0.095)
        fy1 = bbox.y0 + bbox.height * (y + 0.095)
        x_pt = (fx0 * fig_w_in - tight_bbox.x0) * 72.0
        y_pt = (tight_bbox.y1 - fy1 * fig_h_in) * 72.0
        w_pt = (fx1 - fx0) * fig_w_in * 72.0
        h_pt = (fy1 - fy0) * fig_h_in * 72.0
        overlay_elements.append(
            f'<rect x="{x_pt:.3f}" y="{y_pt:.3f}" width="{w_pt:.3f}" height="{h_pt:.3f}" '
            f'rx="2.4" ry="2.4" fill="{PALETTE["paper"]}" />'
        )
        mol_svg = molecule_svg(str(row.get("smiles", "")))
        if mol_svg:
            overlay_elements.append(_svg_group_from_text(mol_svg, x_pt, y_pt, w_pt, h_pt))

    overlay_text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_pt:.3f}pt" height="{height_pt:.3f}pt" '
        f'viewBox="0 0 {width_pt:.3f} {height_pt:.3f}">{"".join(overlay_elements)}</svg>'
    )
    overlay_path = output_dir / "_Figure_3_molecule_overlay.svg"
    overlay_path.write_text(overlay_text, encoding="utf-8")

    final_svg = output_dir / "Figure_3.svg"
    SvgFigure(
        f"{width_pt:.3f}pt",
        f"{height_pt:.3f}pt",
        SVG(str(base_svg_path)),
        SVG(str(overlay_path)),
    ).save(str(final_svg))
    cairosvg.svg2pdf(url=str(final_svg), write_to=str(output_dir / "Figure_3.pdf"))
    cairosvg.svg2png(url=str(final_svg), write_to=str(output_dir / "Figure_3.png"), dpi=600)
    overlay_path.unlink(missing_ok=True)
    base_svg_path.unlink(missing_ok=True)


def load_prediction_pool(structured_dir: Path, hard_dir: Path) -> pd.DataFrame:
    paths = [
        structured_dir / "temporal" / "lightgbm" / "predictions.csv",
        structured_dir / "species" / "lightgbm" / "predictions.csv",
        structured_dir / "chemical_class" / "lightgbm" / "predictions.csv",
        hard_dir / "hard_ood" / "lightgbm" / "predictions.csv",
    ]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["split"] = path.parts[-3]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _draw_best_cell_boxes(ax: plt.Axes, pivot: pd.DataFrame, *, maximize: bool = False) -> None:
    row_labels = list(pivot.index)
    col_labels = list(pivot.columns)
    for col_idx, column in enumerate(col_labels):
        series = pivot[column].dropna()
        if series.empty:
            continue
        target = series.max() if maximize else series.min()
        winners = series[series.eq(target)].index.tolist()
        for row in winners:
            row_idx = row_labels.index(row)
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    linewidth=1.15,
                    edgecolor=PALETTE["ink"],
                )
            )


def figure3_performance_collapse(
    summary: pd.DataFrame,
    random_sensitivity: pd.DataFrame,
    random_reference: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = summary.copy()
    frame = frame[(frame["split"].isin(SPLIT_ORDER)) & (frame["model"] == "lightgbm")].copy()
    frame["split_label"] = frame["split"].map(SPLIT_LABELS)
    split_order = [SPLIT_LABELS[s] for s in SPLIT_ORDER]
    split_colors = [
        PALETTE["mint"] if s == "random" else PALETTE["blue"]
        for s in SPLIT_ORDER
    ]

    fig = plt.figure(figsize=(ACS_DOUBLE_WIDTH, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0], hspace=0.34, wspace=0.24)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
    ]
    subgs = gs[1, 1].subgridspec(1, 3, wspace=0.36)
    ax_delta = [fig.add_subplot(subgs[0, i]) for i in range(3)]
    fig.text(
        0.5,
        0.985,
        "Deployment collapse relative to random interpolation",
        ha="center",
        va="top",
        fontsize=7.6,
        color=PALETTE["slate"],
    )
    panel_specs = [
        ("rmse", "RMSE on log toxicity", "Deployment error"),
        ("coverage", "90% interval coverage", "Coverage under shift"),
        ("abstain_fraction", "Diagnostic abstention fraction", "Selective abstention"),
    ]
    for idx, (ax, (metric, ylabel, title)) in enumerate(zip(axes, panel_specs)):
        values = (
            frame.set_index("split_label")
            .reindex(split_order)[metric]
            .to_numpy(dtype=float)
        )
        ax.bar(split_order, values, color=split_colors, edgecolor="white", linewidth=0.8, width=0.72)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=22)
        ax.set_title(title, pad=6)
        add_panel_label(ax, chr(ord("A") + idx), x=-0.2, y=1.1)
        if metric == "coverage":
            ax.axhline(0.9, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
        finish_axis(ax)
    fig.text(0.135, 0.928, "Current full model: LightGBM", ha="left", va="center", fontsize=6.8, color=PALETTE["slate"])

    rand = random_sensitivity[(random_sensitivity["group"] == "all") & (random_sensitivity["model"] == "lightgbm")].iloc[0]
    random_ref = random_reference[(random_reference["group"] == "all") & (random_reference["split"] == "random") & (random_reference["model"] == "lightgbm")].iloc[0]
    metric_specs = [
        ("rmse", "RMSE", True),
        ("coverage", "Coverage", False),
        ("aurc", "AURC", True),
    ]
    add_panel_label(ax_delta[0], "D", x=-0.38, y=1.12)
    for idx, (ax, (metric, title, lower_is_better)) in enumerate(zip(ax_delta, metric_specs)):
        base_mean = random_ref[f"{metric}_mean"]
        base_std = random_ref[f"{metric}_std"]
        grouped_mean = rand[f"{metric}_mean"]
        grouped_std = rand[f"{metric}_std"]
        sns.barplot(
            x=["Row-random", "Grouped"],
            y=[base_mean, grouped_mean],
            hue=["Row-random", "Grouped"],
            palette=[PALETTE["blue"], PALETTE["purple"]],
            ax=ax,
            width=0.58,
            dodge=False,
            legend=False,
        )
        ax.errorbar(
            [0, 1],
            [base_mean, grouped_mean],
            yerr=[base_std, grouped_std],
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=0.8,
            capsize=2.8,
            capthick=0.8,
            zorder=3,
        )
        ax.set_title(title, pad=5)
        ax.set_xlabel("")
        ax.set_ylabel("" if idx else "Random-split sensitivity")
        ax.tick_params(axis="x", rotation=18, labelsize=6.7)
        if metric == "coverage":
            ax.axhline(0.90, color=PALETTE["ink"], linestyle="--", linewidth=0.8)
        finish_axis(ax)
        if idx:
            ax.set_ylabel("")
        arrow = "worse" if (grouped_mean > base_mean and lower_is_better) or (grouped_mean < base_mean and not lower_is_better) else "better"
        ax.text(
            0.50,
            0.98,
            arrow,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.8,
            color=PALETTE["red"] if arrow == "worse" else PALETTE["green"],
            fontweight="bold",
        )
    save_figure(fig, output_dir, "Figure_2")


def figure4_operational_gate(
    screening_metrics: pd.DataFrame,
    screening_summary: pd.DataFrame,
    burden_summary: pd.DataFrame,
    gate_split_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    short_split_labels = {
        "Class Holdout": "Class",
        "Hard OOD": "Hard",
        "Scaffold": "Scaf",
        "Species": "Sp.",
        "Temporal": "Temp",
    }
    short_workflow_labels = {
        "Baseline only": "Baseline",
        "Baseline + EcoOOD rule": "+ EcoOOD",
    }
    short_method_labels = {
        "EcoOOD": "EcoOOD",
        "Input-space kNN": "Input kNN",
        "Similarity AD": "Sim. AD",
    }
    short_model_labels = {
        "Random Forest": "RF",
        "LightGBM": "LGBM",
        "XGBoost": "XGB",
    }
    fig = plt.figure(figsize=(ACS_DOUBLE_WIDTH, 5.35), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.0, 1.05],
        hspace=0.68,
        wspace=0.34,
    )
    ax_false = fig.add_subplot(outer[0, 0:2])
    ax_burden_false = fig.add_subplot(outer[0, 2:4])
    ax_burden_rescue = fig.add_subplot(outer[0, 4:6])
    split_false_gs = outer[1, 0:3].subgridspec(1, 3, wspace=0.28)
    split_rescue_gs = outer[1, 3:6].subgridspec(1, 3, wspace=0.28)
    axes_split_false = [fig.add_subplot(split_false_gs[0, idx]) for idx in range(3)]
    axes_split_rescue = [fig.add_subplot(split_rescue_gs[0, idx]) for idx in range(3)]

    false_df = screening_metrics[
        (screening_metrics["metric"] == "false_reassurance_rate") & (screening_metrics["split"] != "pooled")
    ].copy()
    false_df["split_label"] = false_df["split"].map(SPLIT_LABELS)
    false_df["workflow_label"] = false_df["workflow"].map(WORKFLOW_LABELS)
    false_df["model_label"] = false_df["model"].map(MODEL_LABELS)
    sns.lineplot(
        data=false_df,
        x="split_label",
        y="value",
        hue="workflow_label",
        style="model_label",
        markers=True,
        dashes=True,
        palette={WORKFLOW_LABELS[key]: WORKFLOW_COLORS[key] for key in WORKFLOW_ORDER},
        estimator=None,
        legend=False,
        ax=ax_false,
    )
    ax_false.set_ylabel("HC in low-priority")
    ax_false.set_xlabel("")
    ax_false.set_title("Rule validation", pad=5, fontsize=8.1)
    ax_false.set_xticks(ax_false.get_xticks())
    ax_false.set_xticklabels(
        [short_split_labels.get(label.get_text(), label.get_text()) for label in ax_false.get_xticklabels()]
    )
    ax_false.tick_params(axis="x", rotation=20)
    finish_axis(ax_false, grid_axis="y")
    add_panel_label(ax_false, "A", x=-0.15, y=1.08)
    pooled = screening_summary[screening_summary["split"] == "pooled"].set_index("model")
    pooled_lines = []
    for model in MODEL_ORDER:
        if model not in pooled.index:
            continue
        pooled_lines.append(
            f"{ {'random_forest': 'RF', 'lightgbm': 'LGBM', 'xgboost': 'XGB'}[model] }: {pooled.loc[model, 'baseline_false_reassurance_rate']:.3f}"
            f" -> {pooled.loc[model, 'gated_false_reassurance_rate']:.3f}"
        )
    if pooled_lines:
        ax_false.text(
            0.03,
            0.03,
            "\n".join(pooled_lines),
            transform=ax_false.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE["grid"], alpha=0.94),
        )
    workflow_handles = [
        plt.Line2D(
            [0],
            [0],
            color=WORKFLOW_COLORS[key],
            lw=1.8,
            marker="o",
            markersize=3.4,
            label=short_workflow_labels[WORKFLOW_LABELS[key]],
        )
        for key in WORKFLOW_ORDER
    ]
    figure4_model_handles = [
        plt.Line2D(
            [0],
            [0],
            color=PALETTE["ink"],
            lw=1.6,
            linestyle={"Random Forest": "-", "LightGBM": (0, (4, 2)), "XGBoost": (0, (1, 1))}[MODEL_LABELS[model]],
            marker={"Random Forest": "o", "LightGBM": "s", "XGBoost": "^"}[MODEL_LABELS[model]],
            markersize=3.4,
            label=short_model_labels[MODEL_LABELS[model]],
        )
        for model in MODEL_ORDER
    ]
    ax_false.legend(
        handles=workflow_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        fontsize=5.5,
        handlelength=2.2,
        columnspacing=0.8,
        handletextpad=0.5,
        bbox_to_anchor=(0.0, 1.01),
    )

    burden = burden_summary.copy()
    burden["model_label"] = burden["model"].map(MODEL_LABELS)
    burden["method_label"] = burden["method"].map(GATE_METHOD_LABELS)
    line_styles = {"Random Forest": "-", "LightGBM": (0, (4, 2)), "XGBoost": (0, (1, 1))}
    marker_map = {"Random Forest": "o", "LightGBM": "s", "XGBoost": "^"}
    for method in GATE_METHOD_ORDER:
        method_frame = burden[burden["method"] == method].copy()
        for model_label in [MODEL_LABELS[key] for key in MODEL_ORDER]:
            subset = method_frame[method_frame["model_label"] == model_label].copy()
            ax_burden_false.plot(
                subset["review_burden"],
                subset["false_reassurance_rate"],
                color=GATE_METHOD_COLORS[method],
                linestyle=line_styles[model_label],
                marker=marker_map[model_label],
                markersize=3.2,
                linewidth=1.5,
            )
            ax_burden_rescue.plot(
                subset["review_burden"],
                subset["rescued_false_negative_fraction"],
                color=GATE_METHOD_COLORS[method],
                linestyle=line_styles[model_label],
                marker=marker_map[model_label],
                markersize=3.2,
                linewidth=1.5,
            )

    for ax in [ax_burden_false, ax_burden_rescue]:
        ax.axvline(0.25, color=PALETTE["slate"], linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_xlim(0.145, 0.355)
        ax.set_xticks([0.15, 0.20, 0.25, 0.30, 0.35])
        ax.set_xlabel("Review burden")
        finish_axis(ax, grid_axis="both")

    ax_burden_false.set_title("False reassurance", pad=5, fontsize=8.1)
    ax_burden_false.set_ylabel("Lower-priority false reassurance")
    add_panel_label(ax_burden_false, "B", x=-0.15, y=1.08)

    ax_burden_rescue.set_title("Rescued misses", pad=5, fontsize=8.1)
    ax_burden_rescue.set_ylabel("Rescued false-negative fraction")
    add_panel_label(ax_burden_rescue, "C", x=-0.15, y=1.08)

    gate_handles = [
        plt.Line2D(
            [0],
            [0],
            color=GATE_METHOD_COLORS[method],
            lw=1.8,
            marker="o",
            markersize=3.5,
            label=short_method_labels[GATE_METHOD_LABELS[method]],
        )
        for method in GATE_METHOD_ORDER
    ]
    fig.legend(
        handles=gate_handles + figure4_model_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.72, 1.005),
        frameon=False,
        fontsize=5.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    split_frame = gate_split_summary.copy()
    split_frame["split_label"] = split_frame["split"].map(SPLIT_LABELS)
    split_frame["model_label"] = split_frame["model"].map(MODEL_LABELS)
    split_frame["method_label"] = split_frame["method"].map(GATE_METHOD_LABELS)
    split_order = [SPLIT_LABELS[key] for key in ["scaffold", "temporal", "species", "chemical_class", "hard_ood"]]
    split_order_short = [short_split_labels[label] for label in split_order]
    for idx, model in enumerate(MODEL_ORDER):
        model_label = MODEL_LABELS[model]
        model_frame = split_frame[split_frame["model"] == model].copy()
        sns.barplot(
            data=model_frame,
            x="split_label",
            y="false_reassurance_rate",
            hue="method_label",
            hue_order=[GATE_METHOD_LABELS[key] for key in GATE_METHOD_ORDER],
            order=split_order,
            palette={GATE_METHOD_LABELS[key]: GATE_METHOD_COLORS[key] for key in GATE_METHOD_ORDER},
            errorbar=None,
            ax=axes_split_false[idx],
        )
        sns.barplot(
            data=model_frame,
            x="split_label",
            y="rescued_false_negative_fraction",
            hue="method_label",
            hue_order=[GATE_METHOD_LABELS[key] for key in GATE_METHOD_ORDER],
            order=split_order,
            palette={GATE_METHOD_LABELS[key]: GATE_METHOD_COLORS[key] for key in GATE_METHOD_ORDER},
            errorbar=None,
        ax=axes_split_rescue[idx],
        )
        short_model_label = {"Random Forest": "RF", "LightGBM": "LGBM", "XGBoost": "XGB"}[model_label]
        axes_split_false[idx].set_title(short_model_label, pad=4, fontsize=7.4)
        axes_split_rescue[idx].set_title(short_model_label, pad=4, fontsize=7.4)
        axes_split_false[idx].set_xticks(axes_split_false[idx].get_xticks())
        axes_split_rescue[idx].set_xticks(axes_split_rescue[idx].get_xticks())
        axes_split_false[idx].set_xticklabels(split_order_short)
        axes_split_rescue[idx].set_xticklabels(split_order_short)
        axes_split_false[idx].tick_params(axis="x", rotation=24, labelsize=6.1)
        axes_split_rescue[idx].tick_params(axis="x", rotation=24, labelsize=6.1)
        axes_split_false[idx].set_xlabel("")
        axes_split_rescue[idx].set_xlabel("")
        axes_split_false[idx].legend_.remove()
        axes_split_rescue[idx].legend_.remove()
        finish_axis(axes_split_false[idx], grid_axis="y")
        finish_axis(axes_split_rescue[idx], grid_axis="y")
        if idx == 0:
            axes_split_false[idx].set_ylabel("False reassurance")
            axes_split_rescue[idx].set_ylabel("Rescued misses")
            add_panel_label(axes_split_false[idx], "D", x=-0.22, y=1.08)
            add_panel_label(axes_split_rescue[idx], "E", x=-0.22, y=1.08)
        else:
            axes_split_false[idx].set_ylabel("")
            axes_split_rescue[idx].set_ylabel("")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "Figure_3.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "Figure_3.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "Figure_3.svg", bbox_inches="tight", facecolor="white")
    sync_saved_figure(output_dir, "Figure_3", include_svg=True)
    plt.close(fig)


def figure6_decision_map(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    def normalize_casrn(value: object) -> str:
        return "".join(ch for ch in str(value) if ch.isdigit())

    def build_priority_case_studies(priority_rows: pd.DataFrame) -> pd.DataFrame:
        if priority_rows.empty:
            return pd.DataFrame(
                columns=[
                    "label",
                    "chemical_name",
                    "casrn",
                    "chemical_class",
                    "candidate_rows",
                    "unique_species",
                    "min_predicted_log_molar",
                    "max_ecoood_score",
                    "decision_pattern",
                    "environmental_anchor",
                ]
            )
        frame = priority_rows.copy()
        frame["casrn_digits"] = frame["casrn"].map(normalize_casrn)
        records: list[dict[str, object]] = []
        for spec in PRIORITY_CASE_SPECS:
            subset = frame[frame["casrn_digits"] == spec["casrn_digits"]].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(["ecoood_score", "y_pred"], ascending=[False, True])
            records.append(
                {
                    "label": spec["label"],
                    "chemical_name": subset.iloc[0]["chemical_name"],
                    "casrn": subset.iloc[0]["casrn"],
                    "smiles": subset.iloc[0].get("smiles", ""),
                    "chemical_class": subset.iloc[0]["chemical_class"],
                    "candidate_rows": int(len(subset)),
                    "unique_species": int(subset["species"].nunique()),
                    "min_predicted_log_molar": float(subset["y_pred"].min()),
                    "max_ecoood_score": float(subset["ecoood_score"].max()),
                    "decision_pattern": "; ".join(sorted(set(subset["decision"].astype(str)))),
                    "environmental_anchor": spec["anchor"],
                }
            )
        return pd.DataFrame.from_records(records)

    frame = predictions.copy()
    frame["toxicity_priority"] = -frame["y_pred"]
    tox_thresh = frame["y_pred"].quantile(0.25)
    ood_thresh = frame["ecoood_score"].quantile(0.75)
    frame["priority_bucket"] = "low_priority"
    frame.loc[(frame["y_pred"] <= tox_thresh) & (frame["ecoood_score"] <= ood_thresh), "priority_bucket"] = "actionable_high_tox"
    frame.loc[(frame["y_pred"] <= tox_thresh) & (frame["ecoood_score"] > ood_thresh), "priority_bucket"] = "prioritize_testing"
    frame.loc[(frame["y_pred"] > tox_thresh) & (frame["ecoood_score"] > ood_thresh), "priority_bucket"] = "withhold_review_low_tox"
    frame["priority_label"] = frame["priority_bucket"].map(PRIORITY_LABELS)
    frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
    frame["covered"] = (frame["y_true"] >= frame["interval_lower"]) & (frame["y_true"] <= frame["interval_upper"])
    catastrophic_cutoff = float(frame["abs_error"].quantile(0.90))
    frame["catastrophic_error"] = frame["abs_error"] >= catastrophic_cutoff
    frame["true_high_concern"] = frame["y_true"] <= tox_thresh
    frame.to_csv(output_dir / "figure6_source_predictions.csv", index=False)

    decision_order = ["predict", "warn", "abstain"]
    decision_summary = (
        frame.groupby("decision", as_index=False)
        .agg(
            n_rows=("decision", "size"),
            rmse=("abs_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("abs_error", "mean"),
            coverage=("covered", "mean"),
            catastrophic_error_rate=("catastrophic_error", "mean"),
            catastrophic_error_count=("catastrophic_error", "sum"),
        )
    )
    decision_summary["decision"] = pd.Categorical(decision_summary["decision"], categories=decision_order, ordered=True)
    decision_summary = decision_summary.sort_values("decision").reset_index(drop=True)
    total_catastrophic = float(decision_summary["catastrophic_error_count"].sum())
    decision_summary["catastrophic_error_share"] = decision_summary["catastrophic_error_count"] / total_catastrophic if total_catastrophic else 0.0
    decision_summary.to_csv(output_dir / "decision_label_outcome_summary.csv", index=False)
    route_summary = (
        frame.groupby("priority_bucket", as_index=False)
        .agg(
            n_rows=("priority_bucket", "size"),
            rmse=("abs_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("abs_error", "mean"),
            coverage=("covered", "mean"),
            catastrophic_error_rate=("catastrophic_error", "mean"),
            catastrophic_error_count=("catastrophic_error", "sum"),
            true_high_concern_fraction=("true_high_concern", "mean"),
        )
    )
    route_summary["priority_bucket"] = pd.Categorical(
        route_summary["priority_bucket"],
        categories=ROUTE_VALIDATION_ORDER,
        ordered=True,
    )
    route_summary = route_summary.sort_values("priority_bucket").reset_index(drop=True)
    route_summary.to_csv(output_dir / "route_action_validation_summary.csv", index=False)
    route_threshold_summary = route_threshold_profile(frame)
    sensitivity = priority_threshold_sensitivity(frame)
    route_threshold_summary.to_csv(output_dir / "route_action_threshold_sensitivity.csv", index=False)
    sensitivity.to_csv(output_dir / "priority_map_threshold_sensitivity.csv", index=False)

    priority_full = frame[frame["priority_bucket"] == "prioritize_testing"].copy()
    priority_full = priority_full.sort_values(["ecoood_score", "y_pred"], ascending=[False, True])
    priority = priority_full[
        [
            "chemical_name",
            "casrn",
            "split",
            "endpoint",
            "chemical_class",
            "species",
            "y_pred",
            "ecoood_score",
            "decision",
            "priority_bucket",
            "known_ood",
        ]
    ].head(30)
    priority.to_csv(output_dir / "priority_testing_candidates.csv", index=False)
    case_studies = build_priority_case_studies(priority_full)
    case_studies.to_csv(output_dir / "priority_testing_case_studies.csv", index=False)

    fig = plt.figure(figsize=(ACS_DOUBLE_WIDTH, 6.05), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.86, 3.3, 1.72],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.80],
        hspace=0.46,
        wspace=0.28,
    )
    ax_top = fig.add_subplot(gs[0, :5])
    ax_main = fig.add_subplot(gs[1, :5], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 5], sharey=ax_main)
    ax_validation = fig.add_subplot(gs[2, 0:3])
    ax_route_mix = fig.add_subplot(gs[2, 3:4])
    ax_priority_stability = fig.add_subplot(gs[2, 4:6])

    x_min = float(frame["y_pred"].min())
    x_max = float(frame["y_pred"].max())
    y_max = float(frame["ecoood_score"].max() * 1.04)

    fig.text(
        0.5,
        0.985,
        f"Deployment output is a routing decision across {len(frame):,} rows and {frame['chemical_id'].nunique():,} chemicals",
        ha="center",
        va="top",
        fontsize=7.6,
        color=PALETTE["slate"],
    )

    hist_priority_order = ["actionable_high_tox", "withhold_review_low_tox", "prioritize_testing", "lower_priority"]

    sns.histplot(
        data=frame,
        x="y_pred",
        hue="priority_label",
        hue_order=[PRIORITY_LABELS[key] for key in hist_priority_order],
        palette={PRIORITY_LABELS[key]: PRIORITY_COLORS[key] for key in PRIORITY_ORDER},
        bins=26,
        multiple="stack",
        alpha=0.95,
        ax=ax_top,
        legend=False,
        edgecolor="white",
        linewidth=0.25,
    )
    ax_top.axvline(tox_thresh, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("Rows")
    ax_top.tick_params(axis="x", labelbottom=False)
    finish_axis(ax_top, grid_axis="y")

    ax_main.axvspan(x_min, tox_thresh, ymin=0, ymax=ood_thresh / y_max, facecolor=PALETTE["mint"], alpha=0.5, zorder=0)
    ax_main.axvspan(tox_thresh, x_max, ymin=0, ymax=ood_thresh / y_max, facecolor="#E6F2F7", alpha=1.0, zorder=0)
    ax_main.axvspan(x_min, tox_thresh, ymin=ood_thresh / y_max, ymax=1, facecolor=PALETTE["sand"], alpha=0.45, zorder=0)
    ax_main.axvspan(tox_thresh, x_max, ymin=ood_thresh / y_max, ymax=1, facecolor=PALETTE["blush"], alpha=0.45, zorder=0)
    non_low = frame[frame["priority_bucket"] != "low_priority"]
    sns.scatterplot(
        data=non_low,
        x="y_pred",
        y="ecoood_score",
        hue="priority_label",
        hue_order=[PRIORITY_LABELS[key] for key in PRIORITY_ORDER],
        palette={PRIORITY_LABELS[key]: PRIORITY_COLORS[key] for key in PRIORITY_ORDER},
        alpha=0.74,
        s=30,
        linewidth=0.35,
        edgecolor="white",
        ax=ax_main,
    )
    # Re-draw lower-priority points with a soft blue overlay so the dense bottom-right quadrant remains visible
    # without reverting to the harsher black/gray palette used in earlier drafts.
    low_priority = frame[frame["priority_bucket"] == "low_priority"]
    ax_main.scatter(
        low_priority["y_pred"],
        low_priority["ecoood_score"],
        s=16,
        color=PRIORITY_COLORS["lower_priority"],
        alpha=0.34,
        linewidth=0.15,
        edgecolors="white",
        zorder=2.8,
    )
    ax_main.axvline(tox_thresh, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    ax_main.axhline(ood_thresh, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    ax_main.set_xlabel("Predicted toxicity (log molar, left = more toxic)")
    ax_main.set_ylabel("EcoOOD score")
    add_panel_label(ax_main, "A", x=-0.08, y=1.03)
    bucket_counts = frame["priority_bucket"].value_counts().to_dict()
    ax_main.text(
        frame["y_pred"].quantile(0.08),
        frame["ecoood_score"].quantile(0.95),
        f"High tox\nPrioritize testing\nn={bucket_counts.get('prioritize_testing', 0)}",
        ha="left",
        va="top",
        fontsize=7.6,
        weight="bold",
        color=PALETTE["ink"],
    )
    ax_main.text(
        frame["y_pred"].quantile(0.93),
        frame["ecoood_score"].quantile(0.95),
        f"Low tox\nWithhold/review\nn={bucket_counts.get('withhold_review_low_tox', 0)}",
        ha="right",
        va="top",
        fontsize=7.6,
        color=PALETTE["ink"],
    )
    ax_main.text(
        frame["y_pred"].quantile(0.08),
        frame["ecoood_score"].quantile(0.06),
        f"High tox\nScreen now\nn={bucket_counts.get('actionable_high_tox', 0)}",
        ha="left",
        va="bottom",
        fontsize=7.6,
        weight="bold",
        color=PALETTE["ink"],
    )
    ax_main.text(
        frame["y_pred"].quantile(0.93),
        frame["ecoood_score"].quantile(0.06),
        f"Lower priority\nn={bucket_counts.get('low_priority', 0)}",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=PALETTE["ink"],
    )
    finish_axis(ax_main, grid_axis="both")

    sns.histplot(
        data=frame,
        y="ecoood_score",
        hue="priority_label",
        hue_order=[PRIORITY_LABELS[key] for key in hist_priority_order],
        palette={PRIORITY_LABELS[key]: PRIORITY_COLORS[key] for key in PRIORITY_ORDER},
        bins=26,
        multiple="stack",
        alpha=0.95,
        ax=ax_right,
        legend=False,
        edgecolor="white",
        linewidth=0.25,
    )
    ax_right.axhline(ood_thresh, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    ax_right.set_xlabel("Rows")
    ax_right.set_ylabel("")
    ax_right.tick_params(axis="y", labelleft=False)
    finish_axis(ax_right, grid_axis="x")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.0,
            markerfacecolor=PRIORITY_COLORS[key],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=PRIORITY_LABELS[key],
        )
        for key in PRIORITY_ORDER
    ]
    labels = [PRIORITY_LABELS[key] for key in PRIORITY_ORDER]
    ax_main.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=2,
        title="",
        frameon=False,
    )

    add_panel_label(ax_validation, "B", x=-0.07, y=1.03)
    route_labels = {
        "low_priority": "Lower priority",
        "withhold_review_low_tox": "Withhold/review",
        "actionable_high_tox": "Screen now",
        "prioritize_testing": "Prioritize testing",
    }
    route_positions = [3.0, 2.0, 0.6, -0.4]
    route_plot = route_summary.set_index("priority_bucket").loc[ROUTE_VALIDATION_ORDER].reset_index()
    route_plot["severe_rate_pct"] = route_plot["catastrophic_error_rate"] * 100.0
    route_plot["true_high_pct"] = route_plot["true_high_concern_fraction"] * 100.0
    ax_validation.barh(
        route_positions,
        route_plot["severe_rate_pct"],
        color=[PRIORITY_COLORS[key] for key in ROUTE_VALIDATION_ORDER],
        edgecolor=PALETTE["ink"],
        linewidth=0.35,
        height=0.56,
        zorder=2,
    )
    ax_validation.axhline(1.45, color=PALETTE["grid"], linewidth=0.9, zorder=1)
    for ypos, (_, row) in zip(route_positions, route_plot.iterrows()):
        ax_validation.text(
            float(row["severe_rate_pct"]) + 0.9,
            ypos,
            f"RMSE {row['rmse']:.02f} | HC {row['true_high_pct']:.1f}%",
            va="center",
            ha="left",
            fontsize=6.6,
            color=PALETTE["ink"],
        )
    ax_validation.text(
        0.0,
        3.62,
        "Predicted lower-tox rows",
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=PALETTE["slate"],
        fontweight="bold",
    )
    ax_validation.text(
        0.0,
        1.22,
        "Predicted higher-tox rows",
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=PALETTE["slate"],
        fontweight="bold",
    )
    ax_validation.set_title("Failure risk by route", pad=2, fontsize=7.8)
    ax_validation.set_yticks(route_positions, [route_labels[key] for key in ROUTE_VALIDATION_ORDER])
    ax_validation.set_ylim(-0.8, 4.0)
    ax_validation.set_xlim(0, max(30, float(route_plot["severe_rate_pct"].max()) + 8.5))
    ax_validation.set_xticks([0, 5, 10, 15, 20, 25])
    ax_validation.set_xlabel("Top-decile absolute error rate (%)")
    ax_validation.set_ylabel("")
    finish_axis(ax_validation, grid_axis="x")

    add_panel_label(ax_route_mix, "C", x=-0.22, y=1.03)
    route_mix = route_threshold_summary.set_index("profile").loc[
        ["relaxed", "baseline", "strict"],
        [
            "screen_now_fraction",
            "lower_priority_fraction",
            "withhold_review_fraction",
            "prioritize_testing_fraction",
        ],
    ]
    bottom = np.zeros(len(route_mix))
    for column, route in zip(route_mix.columns, ["actionable_high_tox", "lower_priority", "withhold_review_low_tox", "prioritize_testing"], strict=False):
        ax_route_mix.bar(
            route_mix.index,
            route_mix[column],
            bottom=bottom,
            color=PRIORITY_COLORS[route],
            width=0.66,
        )
        bottom += route_mix[column].to_numpy()
    ax_route_mix.set_ylim(0, 1.02)
    ax_route_mix.set_title("Route mix", pad=2, fontsize=7.8)
    ax_route_mix.set_ylabel("Fraction")
    ax_route_mix.set_xlabel("")
    finish_axis(ax_route_mix, grid_axis="y")

    add_panel_label(ax_priority_stability, "D", x=-0.15, y=1.03)
    heat = sensitivity.pivot(index="tox_quantile", columns="ood_quantile", values="priority_jaccard_vs_baseline")
    sns.heatmap(
        heat,
        ax=ax_priority_stability,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        linewidths=0.6,
        linecolor="white",
    )
    ax_priority_stability.set_title("Priority stability", pad=2, fontsize=7.8)
    ax_priority_stability.set_xlabel("OOD cutoff")
    ax_priority_stability.set_ylabel("Toxicity cutoff")
    ax_priority_stability.tick_params(axis="x", rotation=0)
    ax_priority_stability.tick_params(axis="y", rotation=0)
    save_figure(fig, output_dir, "Figure_5")
    figure_s5_threshold_sensitivity(route_threshold_summary, sensitivity, output_dir)
    return priority


def write_manifest(output_dir: Path, priority: pd.DataFrame) -> None:
    lines = [
        "Publication figure bundle generated from benchmark outputs.",
        "",
        "Files:",
        "- Figure_1.png/pdf",
        "- Figure_2.png/pdf",
        "- Figure_3.png/pdf",
        "- Figure_4.png/pdf",
        "- Figure_5.png/pdf",
        "- Figure_6.png/pdf",
        "- Figure_S1.png/pdf",
        "- Figure_S2.png/pdf",
        "- Figure_S3.png/pdf",
        "- figure_redesign_blueprint_v1.md",
        "- figure3_source_metrics.csv",
        "- figure4_source_scores.csv",
        "- figure6_source_predictions.csv",
        "- decision_label_outcome_summary.csv",
        "- route_action_validation_summary.csv",
        "- screening_gate_validation_metrics.csv",
        "- screening_gate_validation_summary.csv",
        "- gate_method_comparison_burden_summary.csv",
        "- gate_method_comparison_summary.csv",
        "- gate_method_comparison_examples.csv",
        "- priority_testing_case_studies.csv",
        "- priority_map_threshold_sensitivity.csv",
        "- route_action_threshold_sensitivity.csv",
        "- priority_testing_candidates.csv",
        "- si_outline.md",
        "",
        f"Priority testing candidates exported: {len(priority)}",
    ]
    (output_dir / "figure_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EcoOOD analysis figures.")
    parser.add_argument(
        "--structured-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_structured_ad"),
    )
    parser.add_argument(
        "--structured-scaffold-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_structured_ad_scaffold"),
    )
    parser.add_argument(
        "--hard-ood-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_hard_ood_ad"),
    )
    parser.add_argument(
        "--structured-xgb-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_structured_ad_xgboost"),
    )
    parser.add_argument(
        "--structured-scaffold-xgb-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_structured_ad_scaffold_xgboost"),
    )
    parser.add_argument(
        "--hard-ood-xgb-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_hard_ood_ad_xgboost"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/release_tables"))
    args = parser.parse_args()

    apply_publication_style()
    structured_summary = pd.read_csv(args.structured_dir / "benchmark_summary.csv")
    scaffold_summary = pd.read_csv(args.structured_scaffold_dir / "benchmark_summary.csv")
    hard_summary = pd.read_csv(args.hard_ood_dir / "benchmark_summary.csv")
    structured_xgb_summary = pd.read_csv(args.structured_xgb_dir / "benchmark_summary.csv")
    scaffold_xgb_summary = pd.read_csv(args.structured_scaffold_xgb_dir / "benchmark_summary.csv")
    hard_xgb_summary = pd.read_csv(args.hard_ood_xgb_dir / "benchmark_summary.csv")
    structured_summary = structured_summary[structured_summary["split"] != "scaffold"].copy()
    scaffold_summary = scaffold_summary[scaffold_summary["split"] == "scaffold"].copy()
    structured_xgb_summary = structured_xgb_summary[structured_xgb_summary["split"] != "scaffold"].copy()
    scaffold_xgb_summary = scaffold_xgb_summary[scaffold_xgb_summary["split"] == "scaffold"].copy()
    combined_summary = pd.concat(
        [
            structured_summary,
            scaffold_summary,
            hard_summary,
            structured_xgb_summary,
            scaffold_xgb_summary,
            hard_xgb_summary,
        ],
        ignore_index=True,
    )
    structured_ood = pd.read_csv(args.structured_dir / "ood_score_summary.csv")
    scaffold_ood = pd.read_csv(args.structured_scaffold_dir / "ood_score_summary.csv")
    hard_ood = pd.read_csv(args.hard_ood_dir / "ood_score_summary.csv")
    structured_ood = structured_ood[structured_ood["split"] != "scaffold"].copy()
    scaffold_ood = scaffold_ood[scaffold_ood["split"] == "scaffold"].copy()
    combined_ood = pd.concat([structured_ood, scaffold_ood, hard_ood], ignore_index=True)
    prediction_pool = load_prediction_pool(args.structured_dir, args.hard_ood_dir)
    screening_predictions = load_screening_panel(
        args.structured_dir,
        args.structured_scaffold_dir,
        args.hard_ood_dir,
        args.structured_xgb_dir,
        args.structured_scaffold_xgb_dir,
        args.hard_ood_xgb_dir,
    )
    screening_panel = aggregate_screening_panel(screening_predictions)
    toxicity_cutoff = float(screening_panel["min_true_tox"].quantile(0.25))
    ood_cutoff_by_model = screening_panel.groupby("model")["max_ecoood"].quantile(0.75).to_dict()
    screening_classified = classify_screening_actions(screening_panel, toxicity_cutoff, ood_cutoff_by_model)
    screening_metrics, screening_summary = summarize_screening_gate(screening_classified)
    gate_predictions = load_gate_panel(
        args.structured_dir,
        args.structured_scaffold_dir,
        args.hard_ood_dir,
        args.structured_xgb_dir,
        args.structured_scaffold_xgb_dir,
        args.hard_ood_xgb_dir,
    )
    gate_panel = aggregate_gate_panel(gate_predictions)
    gate_burden_summary, gate_summary_25 = summarize_gate_methods(gate_panel, burdens=[0.15, 0.20, 0.25, 0.30, 0.35])
    gate_split_summary = summarize_gate_methods_by_split(gate_panel, burden=0.25)
    chemical_random = pd.read_csv(Path("outputs/release_tables/chemical_random_sensitivity_summary.csv"))
    random_reference = pd.read_csv(Path("outputs/seed_sweep_1000chem_dsstox_mech_structured_scaffold_lightgbm/benchmark_summary_agg.csv"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(args.output_dir / "figure3_source_metrics.csv", index=False)
    combined_ood.to_csv(args.output_dir / "figure4_source_scores.csv", index=False)
    prediction_pool.to_csv(args.output_dir / "figure6_source_predictions.csv", index=False)
    screening_metrics.to_csv(args.output_dir / "screening_gate_validation_metrics.csv", index=False)
    screening_summary.to_csv(args.output_dir / "screening_gate_validation_summary.csv", index=False)
    gate_burden_summary.to_csv(args.output_dir / "gate_method_comparison_burden_summary.csv", index=False)
    gate_summary_25.to_csv(args.output_dir / "gate_method_comparison_summary.csv", index=False)
    gate_split_summary.to_csv(args.output_dir / "gate_method_comparison_split_summary.csv", index=False)
    select_gate_examples(gate_panel, review_burden=0.25).to_csv(args.output_dir / "gate_method_comparison_examples.csv", index=False)

    figure3_performance_collapse(combined_summary, chemical_random, random_reference, args.output_dir)
    figure4_operational_gate(screening_metrics, screening_summary, gate_burden_summary, gate_split_summary, args.output_dir)
    priority = figure6_decision_map(prediction_pool, args.output_dir)
    write_manifest(args.output_dir, priority)


if __name__ == "__main__":
    main()
