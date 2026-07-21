from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, add_panel_label, apply_publication_style, finish_axis, save_figure


MODEL_NAME = "lightgbm"
DEPLOYMENT_PROXY_SPLITS = ["temporal", "species", "chemical_class"]
REFERENCE_SPLITS = ["scaffold"]
ALL_SPLITS = REFERENCE_SPLITS + DEPLOYMENT_PROXY_SPLITS
SPLIT_LABELS = {
    "scaffold": "Scaffold ref.",
    "temporal": "Temporal",
    "species": "Species",
    "chemical_class": "Class holdout",
    "pooled_deployment": "Pooled deployment",
}
ACTION_ORDER = ["propagate", "withhold"]
ACTION_LABELS = {
    "propagate": "Propagate downstream",
    "withhold": "Withhold for review",
}
ACTION_COLORS = {
    "propagate": PALETTE["green"],
    "withhold": PALETTE["orange"],
}


def load_prediction_panel(structured_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ALL_SPLITS:
        path = structured_dir / split / MODEL_NAME / "predictions.csv"
        frame = pd.read_csv(
            path,
            usecols=[
                "chemical_id",
                "chemical_name",
                "casrn",
                "chemical_class",
                "species",
                "y_true",
                "y_pred",
                "ecoood_score",
                "interval_width",
            ],
        )
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_species_distribution(predictions: pd.DataFrame, min_species: int = 5) -> pd.DataFrame:
    species_panel = (
        predictions.groupby(
            ["split", "chemical_id", "chemical_name", "casrn", "chemical_class", "species"],
            dropna=False,
            as_index=False,
        )
        .agg(
            species_true_tox=("y_true", "median"),
            species_pred_tox=("y_pred", "median"),
            species_max_ecoood=("ecoood_score", "max"),
            species_median_interval=("interval_width", "median"),
        )
    )

    chemical_panel = (
        species_panel.groupby(
            ["split", "chemical_id", "chemical_name", "casrn", "chemical_class"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_species=("species", "nunique"),
            species_q05_true=("species_true_tox", lambda s: float(pd.Series(s).quantile(0.05))),
            species_q05_pred=("species_pred_tox", lambda s: float(pd.Series(s).quantile(0.05))),
            max_ecoood=("species_max_ecoood", "max"),
            median_interval_width=("species_median_interval", "median"),
        )
    )

    chemical_panel = chemical_panel[chemical_panel["n_species"] >= min_species].copy()
    chemical_panel["species_q05_abs_error"] = (
        chemical_panel["species_q05_pred"] - chemical_panel["species_q05_true"]
    ).abs()
    return chemical_panel


def apply_downstream_gate(chemical_panel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    frames: list[pd.DataFrame] = []
    for split_name, split_frame in chemical_panel.groupby("split", sort=False):
        split_frame = split_frame.copy()
        split_frame["ecoood_cutoff_q75"] = float(split_frame["max_ecoood"].quantile(0.75))
        split_frame["interval_cutoff_q75"] = float(split_frame["median_interval_width"].quantile(0.75))
        split_frame["withhold"] = (
            (split_frame["max_ecoood"] > split_frame["ecoood_cutoff_q75"])
            | (split_frame["median_interval_width"] > split_frame["interval_cutoff_q75"])
        )
        split_frame["gate_action"] = split_frame["withhold"].map(
            {False: "propagate", True: "withhold"}
        )
        frames.append(split_frame)

    gated = pd.concat(frames, ignore_index=True)
    gated["panel_family"] = gated["split"].map(
        lambda x: "deployment_like" if x in DEPLOYMENT_PROXY_SPLITS else "reference"
    )
    pooled_high_error_cutoff = float(
        gated.loc[gated["panel_family"] == "deployment_like", "species_q05_abs_error"].quantile(0.75)
    )
    gated["downstream_high_error"] = gated["species_q05_abs_error"] >= pooled_high_error_cutoff
    return gated, pooled_high_error_cutoff


def summarize_downstream_proxy(
    gated: pd.DataFrame, pooled_high_error_cutoff: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | int | str]] = []
    metric_rows: list[dict[str, float | int | str]] = []

    split_iter = list(ALL_SPLITS) + ["pooled_deployment"]
    for split_name in split_iter:
        if split_name == "pooled_deployment":
            split_frame = gated[gated["panel_family"] == "deployment_like"].copy()
        else:
            split_frame = gated[gated["split"] == split_name].copy()
        if split_frame.empty:
            continue

        summary_rows.append(
            {
                "split": split_name,
                "panel_family": "deployment_like" if split_name == "pooled_deployment" else split_frame["panel_family"].iloc[0],
                "n_chemicals": int(len(split_frame)),
                "n_propagate": int((split_frame["gate_action"] == "propagate").sum()),
                "n_withhold": int((split_frame["gate_action"] == "withhold").sum()),
                "mean_species_per_chemical": float(split_frame["n_species"].mean()),
                "pooled_high_error_cutoff": pooled_high_error_cutoff,
                "mean_abs_error_all": float(split_frame["species_q05_abs_error"].mean()),
                "median_abs_error_all": float(split_frame["species_q05_abs_error"].median()),
                "mean_abs_error_propagate": float(
                    split_frame.loc[split_frame["gate_action"] == "propagate", "species_q05_abs_error"].mean()
                ),
                "mean_abs_error_withhold": float(
                    split_frame.loc[split_frame["gate_action"] == "withhold", "species_q05_abs_error"].mean()
                ),
                "high_error_rate_all": float(split_frame["downstream_high_error"].mean()),
                "high_error_rate_propagate": float(
                    split_frame.loc[split_frame["gate_action"] == "propagate", "downstream_high_error"].mean()
                ),
                "high_error_rate_withhold": float(
                    split_frame.loc[split_frame["gate_action"] == "withhold", "downstream_high_error"].mean()
                ),
            }
        )

        for action_name in ACTION_ORDER:
            action_frame = split_frame[split_frame["gate_action"] == action_name]
            if action_frame.empty:
                continue
            metric_rows.extend(
                [
                    {
                        "split": split_name,
                        "panel_family": "deployment_like" if split_name == "pooled_deployment" else split_frame["panel_family"].iloc[0],
                        "gate_action": action_name,
                        "metric": "mean_abs_error",
                        "value": float(action_frame["species_q05_abs_error"].mean()),
                        "n_chemicals": int(len(action_frame)),
                    },
                    {
                        "split": split_name,
                        "panel_family": "deployment_like" if split_name == "pooled_deployment" else split_frame["panel_family"].iloc[0],
                        "gate_action": action_name,
                        "metric": "high_error_rate",
                        "value": float(action_frame["downstream_high_error"].mean()),
                        "n_chemicals": int(len(action_frame)),
                    },
                ]
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(metric_rows)


def representative_examples(gated: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    deploy = gated[gated["panel_family"] == "deployment_like"].copy()
    examples = deploy[deploy["gate_action"] == "withhold"].copy()
    examples = examples.sort_values(
        ["downstream_high_error", "species_q05_abs_error", "max_ecoood"],
        ascending=[False, False, False],
    ).head(top_n)
    return examples[
        [
            "split",
            "chemical_name",
            "casrn",
            "chemical_class",
            "n_species",
            "species_q05_true",
            "species_q05_pred",
            "species_q05_abs_error",
            "max_ecoood",
            "median_interval_width",
            "gate_action",
            "downstream_high_error",
        ]
    ].reset_index(drop=True)


def plot_downstream_proxy(gated: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    deploy = gated[gated["panel_family"] == "deployment_like"].copy()
    plot_metrics = metrics[
        (metrics["panel_family"] == "deployment_like") & (metrics["split"] != "pooled_deployment")
    ].copy()
    plot_metrics["split_label"] = plot_metrics["split"].map(SPLIT_LABELS)
    deploy["split_label"] = deploy["split"].map(SPLIT_LABELS)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(ACS_DOUBLE_WIDTH, 3.2),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    sns.scatterplot(
        data=deploy,
        x="species_q05_true",
        y="species_q05_pred",
        hue="gate_action",
        style="split_label",
        palette=ACTION_COLORS,
        hue_order=ACTION_ORDER,
        s=44,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    lo = float(min(deploy["species_q05_true"].min(), deploy["species_q05_pred"].min()))
    hi = float(max(deploy["species_q05_true"].max(), deploy["species_q05_pred"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9, color=PALETTE["ink"])
    ax.set_xlabel("Measured species-level 5th percentile (log molar)")
    ax.set_ylabel("Predicted species-level 5th percentile")
    ax.set_title("A  Lower-tail cross-species summary", pad=6, loc="left")
    add_panel_label(ax, "A")
    finish_axis(ax, grid_axis="both")
    ax.legend(loc="lower right", fontsize=6.6, title="")

    ax = axes[1]
    err_df = plot_metrics[plot_metrics["metric"] == "mean_abs_error"].copy()
    sns.barplot(
        data=err_df,
        x="split_label",
        y="value",
        hue="gate_action",
        hue_order=ACTION_ORDER,
        palette=ACTION_COLORS,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean absolute lower-tail error")
    ax.set_title("B  Propagate vs withhold error", pad=6, loc="left")
    ax.tick_params(axis="x", rotation=20)
    add_panel_label(ax, "B")
    finish_axis(ax, grid_axis="y")
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax = axes[2]
    hi_df = plot_metrics[plot_metrics["metric"] == "high_error_rate"].copy()
    sns.barplot(
        data=hi_df,
        x="split_label",
        y="value",
        hue="gate_action",
        hue_order=ACTION_ORDER,
        palette=ACTION_COLORS,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("High downstream-error fraction")
    ax.set_title("C  Failure enrichment in withheld queue", pad=6, loc="left")
    ax.tick_params(axis="x", rotation=20)
    add_panel_label(ax, "C")
    finish_axis(ax, grid_axis="y")
    ax.set_ylim(0, max(0.62, float(hi_df["value"].max()) * 1.15))
    if ax.legend_ is not None:
        ax.legend_.remove()

    save_figure(fig, output_dir, "Figure_archived_downstream_proxy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a retrospective lower-tail cross-species summary for EcoOOD."
    )
    parser.add_argument(
        "--structured-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_structured_ad_scaffold"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/release_tables"))
    args = parser.parse_args()

    apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_prediction_panel(args.structured_dir)
    chemical_panel = aggregate_species_distribution(predictions, min_species=5)
    gated, pooled_high_error_cutoff = apply_downstream_gate(chemical_panel)
    summary, metrics = summarize_downstream_proxy(gated, pooled_high_error_cutoff)
    examples = representative_examples(gated)

    chemical_panel.to_csv(args.output_dir / "cross_species_lower_tail_panel.csv", index=False)
    gated.to_csv(args.output_dir / "cross_species_lower_tail_classified.csv", index=False)
    summary.to_csv(args.output_dir / "cross_species_lower_tail_summary.csv", index=False)
    metrics.to_csv(args.output_dir / "cross_species_lower_tail_metrics.csv", index=False)
    examples.to_csv(args.output_dir / "cross_species_lower_tail_examples.csv", index=False)

    plot_downstream_proxy(gated, metrics, args.output_dir)


if __name__ == "__main__":
    main()
