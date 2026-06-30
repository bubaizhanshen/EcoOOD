from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, add_panel_label, apply_publication_style, finish_axis, save_figure


AXIS_COLUMNS = {
    "chemical": "d_chem",
    "species": "d_species",
    "context": "d_context",
    "mechanism": "d_mech",
}
AXIS_ORDER = ["chemical", "species", "context", "mechanism"]
AXIS_LABELS = {
    "chemical": "Chemical",
    "species": "Species",
    "context": "Context",
    "mechanism": "Mechanism",
}
AXIS_COLORS = {
    "chemical": PALETTE["blue"],
    "species": PALETTE["green"],
    "context": PALETTE["orange"],
    "mechanism": PALETTE["purple"],
}
SPLIT_ORDER = ["random", "temporal", "species", "chemical_class", "hard_ood"]
SPLIT_LABELS = {
    "random": "Random",
    "temporal": "Temporal",
    "species": "Species",
    "chemical_class": "Class Holdout",
    "hard_ood": "Hard OOD",
}
REFERENCE_METHOD_ORDER = ["ecoood", "ad_similarity"]
REFERENCE_METHOD_LABELS = {
    "ecoood": "EcoOOD",
    "ad_similarity": "Similarity AD",
}
REFERENCE_METHOD_COLORS = {
    "ecoood": PALETTE["blue"],
    "ad_similarity": PALETTE["orange"],
}
REFERENCE_METRICS = [
    ("aurc", "AURC"),
    ("reference_fpr95", "Reference FPR95"),
]


def _prediction_paths(root: Path, group: str, model: str) -> list[Path]:
    return sorted(root.glob(f"seeds/seed_*/{group}/*/{model}/predictions.csv"))


def _metadata_from_path(path: Path) -> tuple[int, str, str]:
    parts = path.parts
    seed = int(parts[-5].removeprefix("seed_"))
    group = parts[-4]
    split = parts[-3]
    return seed, group, split


def summarize_prediction_file(path: Path, error_quantile: float = 0.9, tail_quantile: float = 0.8) -> pd.DataFrame:
    frame = pd.read_csv(path)
    seed, group, split = _metadata_from_path(path)
    abs_error = np.abs(frame["y_true"] - frame["y_pred"])
    error_cutoff = float(np.quantile(abs_error, error_quantile))
    high_error = abs_error >= error_cutoff

    rows: list[dict[str, float | int | str]] = []
    for axis_name, column in AXIS_COLUMNS.items():
        values = frame[column].to_numpy(dtype=float)
        baseline_mean = float(np.mean(values))
        high_error_mean = float(np.mean(values[high_error])) if np.any(high_error) else float("nan")
        cutoff = float(np.quantile(values, tail_quantile))
        high_tail_capture = float(np.mean(values[high_error] >= cutoff)) if np.any(high_error) else float("nan")
        rows.append(
            {
                "seed": seed,
                "group": group,
                "split": split,
                "axis": axis_name,
                "n_samples": int(len(frame)),
                "high_error_fraction": float(np.mean(high_error)),
                "baseline_mean": baseline_mean,
                "high_error_mean": high_error_mean,
                "delta_mean": float(high_error_mean - baseline_mean),
                "enrichment_ratio": float(high_error_mean / baseline_mean) if baseline_mean > 0 else float("nan"),
                "high_tail_capture": high_tail_capture,
            }
        )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "baseline_mean",
        "high_error_mean",
        "delta_mean",
        "enrichment_ratio",
        "high_tail_capture",
    ]
    grouped = (
        frame.groupby(["group", "split", "axis"], dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def load_reference_summary(
    source_scores_path: Path,
    *,
    split: str = "scaffold",
    model: str = "lightgbm",
) -> pd.DataFrame:
    source_scores = pd.read_csv(source_scores_path)
    subset = source_scores[
        (source_scores["split"] == split)
        & (source_scores["model"] == model)
        & (source_scores["method"].isin(REFERENCE_METHOD_ORDER))
    ].copy()
    missing = [method for method in REFERENCE_METHOD_ORDER if method not in set(subset["method"])]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Missing scaffold reference rows in {source_scores_path}: {missing_text}")

    rows: list[dict[str, float | str]] = []
    for metric, metric_label in REFERENCE_METRICS:
        for method in REFERENCE_METHOD_ORDER:
            value = float(subset.loc[subset["method"] == method, metric].iloc[0])
            rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "method": method,
                    "method_label": REFERENCE_METHOD_LABELS[method],
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def plot_reference_panel(ax: plt.Axes, reference_summary: pd.DataFrame) -> None:
    y_positions = np.arange(len(REFERENCE_METRICS))[::-1]
    metric_labels = [metric_label for _, metric_label in REFERENCE_METRICS]
    x_max = max(0.12, float(reference_summary["value"].max()) * 1.35)

    for y_pos, (metric, _) in zip(y_positions, REFERENCE_METRICS):
        subset = reference_summary[reference_summary["metric"] == metric].set_index("method")
        values = subset.loc[REFERENCE_METHOD_ORDER, "value"].to_numpy(dtype=float)
        ax.hlines(y_pos, values.min(), values.max(), color=PALETTE["grid"], linewidth=1.3, zorder=1)
        for idx, method in enumerate(REFERENCE_METHOD_ORDER):
            value = float(subset.loc[method, "value"])
            ax.scatter(
                value,
                y_pos,
                s=36,
                color=REFERENCE_METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            dy = 0.16 if idx == 0 else -0.16
            ax.text(
                value,
                y_pos + dy,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.6,
                color=REFERENCE_METHOD_COLORS[method],
            )

    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-0.55, len(REFERENCE_METRICS) - 0.45)
    ax.set_yticks(y_positions, labels=metric_labels)
    ax.set_xlabel("Metric value (lower is better)")
    ax.set_ylabel("")
    ax.set_title("Scaffold reference separates ranking from structural detection", pad=6)
    finish_axis(ax, grid_axis="x")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=REFERENCE_METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=5.5,
            label=REFERENCE_METHOD_LABELS[method],
        )
        for method in REFERENCE_METHOD_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=6.4,
        handletextpad=0.5,
        borderaxespad=0.3,
    )


def make_figure(
    aggregated: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
) -> None:
    frame = aggregated.copy()
    frame = frame[frame["split"].isin(["temporal", "species", "chemical_class", "hard_ood"])].copy()
    frame["split_label"] = frame["split"].map(SPLIT_LABELS)

    fig = plt.figure(figsize=(ACS_DOUBLE_WIDTH, 5.75), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.08, 0.92], hspace=0.48, wspace=0.34)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_stack = fig.add_subplot(gs[0, 1])
    ax_endpoint = fig.add_subplot(gs[1, 0])
    ax_reference = fig.add_subplot(gs[1, 1])
    fig.text(
        0.5,
        0.985,
        "Different deployment failures concentrate along different novelty axes",
        ha="center",
        va="top",
        fontsize=7.6,
        color=PALETTE["slate"],
    )

    pivot = (
        frame.pivot(index="axis", columns="split_label", values="delta_mean_mean")
        .reindex(
            index=AXIS_ORDER,
            columns=[SPLIT_LABELS[s] for s in ["temporal", "species", "chemical_class", "hard_ood"]],
        )
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrBr",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "High-error minus baseline axis score", "shrink": 0.82, "pad": 0.02},
        ax=ax_heat,
        annot_kws={"fontsize": 6.8},
    )
    ax_heat.set_title("Axis enrichment among high-error rows", pad=6)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_yticklabels([AXIS_LABELS[a] for a in AXIS_ORDER], rotation=0)
    ax_heat.tick_params(axis="x", rotation=0)
    add_panel_label(ax_heat, "A", x=-0.12, y=1.05)

    comp = frame.copy()
    comp["positive_delta"] = comp["delta_mean_mean"].clip(lower=0.0)
    comp["split_label"] = comp["split"].map(SPLIT_LABELS)
    split_display = [SPLIT_LABELS[s] for s in ["temporal", "species", "chemical_class", "hard_ood"]]
    bottom = np.zeros(len(split_display))
    for axis in AXIS_ORDER:
        vals = (
            comp[comp["axis"] == axis]
            .set_index("split_label")
            .reindex(split_display)["positive_delta"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        totals = (
            comp.groupby("split_label")["positive_delta"].sum()
            .reindex(split_display)
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
        frac = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0)
        ax_stack.bar(split_display, frac, bottom=bottom, color=AXIS_COLORS[axis], label=AXIS_LABELS[axis], width=0.62)
        bottom += frac
    ax_stack.set_ylim(0, 1.02)
    ax_stack.set_ylabel("Share of positive enrichment", labelpad=2)
    ax_stack.set_title("Additional enrichment among failures", pad=6)
    ax_stack.tick_params(axis="x", rotation=16)
    add_panel_label(ax_stack, "B", x=-0.16, y=1.08)
    finish_axis(ax_stack, grid_axis="y")
    ax_stack.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=6.2)

    endpoint = endpoint_summary.copy()
    endpoint = endpoint[(endpoint["split"] == "species") & (endpoint["model"] == "lightgbm")].copy()
    endpoint["endpoint_label"] = endpoint["group"].map(
        {
            "fish_96h_lc50": "Fish 96 h LC50",
            "daphnia_48h_ec50": "Daphnia 48 h EC50",
            "algae_72_96h_ec50": "Algae 72-96 h EC50",
        }
    )
    endpoint = endpoint.sort_values("rmse_mean", ascending=True)
    colors = []
    for group in endpoint["group"]:
        if group == "fish_96h_lc50":
            colors.append(PALETTE["blue"])
        elif group == "daphnia_48h_ec50":
            colors.append(PALETTE["red"])
        else:
            colors.append(PALETTE["green"])
    ax_endpoint.barh(endpoint["endpoint_label"], endpoint["rmse_mean"], color=colors, alpha=0.88)
    ax_endpoint.errorbar(
        endpoint["rmse_mean"],
        endpoint["endpoint_label"],
        xerr=endpoint["rmse_std"],
        fmt="none",
        ecolor=PALETTE["ink"],
        elinewidth=0.8,
        capsize=2.8,
        capthick=0.8,
        zorder=3,
    )
    for _, row in endpoint.iterrows():
        ax_endpoint.text(
            row["rmse_mean"] + row["rmse_std"] + 0.03,
            row["endpoint_label"],
            f"cov {row['coverage_mean']:.2f}",
            va="center",
            ha="left",
            fontsize=6.9,
            color=PALETTE["slate"],
        )
    ax_endpoint.set_title("Species-holdout instability peaks in daphnia endpoints", pad=6)
    ax_endpoint.set_xlabel("RMSE on log toxicity")
    ax_endpoint.set_ylabel("")
    add_panel_label(ax_endpoint, "C", x=-0.16, y=1.08)
    finish_axis(ax_endpoint, grid_axis="x")

    plot_reference_panel(ax_reference, reference_summary)
    add_panel_label(ax_reference, "D", x=-0.16, y=1.08)

    save_figure(fig, output_dir, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze axis-level drivers of high-error EcoOOD failures.")
    parser.add_argument(
        "--sweep-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/seed_sweep_1000chem_dsstox_mech_structured_lightgbm"),
            Path("outputs/seed_sweep_1000chem_dsstox_mech_hard_ood_lightgbm"),
        ],
    )
    parser.add_argument("--group", default="all")
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--error-quantile", type=float, default=0.9)
    parser.add_argument("--tail-quantile", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/failure_axes"))
    parser.add_argument(
        "--endpoint-summary",
        type=Path,
        default=Path("outputs/release_tables/endpoint_split_summary.csv"),
    )
    parser.add_argument(
        "--source-scores",
        type=Path,
        default=Path("outputs/release_tables/figure4_source_scores.csv"),
    )
    parser.add_argument("--output-stem", default="Figure_4")
    args = parser.parse_args()

    apply_publication_style()
    frames: list[pd.DataFrame] = []
    for root in args.sweep_dirs:
        for path in _prediction_paths(root, group=args.group, model=args.model):
            frames.append(
                summarize_prediction_file(
                    path,
                    error_quantile=args.error_quantile,
                    tail_quantile=args.tail_quantile,
                )
            )
    if not frames:
        raise SystemExit("No prediction files found for failure-axis analysis.")

    combined = pd.concat(frames, ignore_index=True)
    aggregated = aggregate(combined)
    driver_rows = (
        aggregated.sort_values(["split", "delta_mean_mean"], ascending=[True, False])
        .groupby(["group", "split"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "failure_axis_all.csv", index=False)
    aggregated.to_csv(args.output_dir / "failure_axis_summary.csv", index=False)
    driver_rows.to_csv(args.output_dir / "failure_axis_top_driver.csv", index=False)
    endpoint_summary = pd.read_csv(args.endpoint_summary)
    reference_summary = load_reference_summary(args.source_scores)
    make_figure(
        aggregated,
        endpoint_summary,
        reference_summary,
        args.output_dir,
        output_stem=args.output_stem,
    )
    print(aggregated.to_string(index=False))


if __name__ == "__main__":
    main()
