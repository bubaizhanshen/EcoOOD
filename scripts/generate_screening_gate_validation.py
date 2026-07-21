from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, add_panel_label, apply_publication_style, finish_axis, save_figure


DEPLOYMENT_SPLITS = ["scaffold", "temporal", "species", "chemical_class", "hard_ood"]
MODEL_ORDER = ["random_forest", "lightgbm", "xgboost"]
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "random_forest": PALETTE["blue"],
    "lightgbm": PALETTE["green"],
    "xgboost": PALETTE["orange"],
}
WORKFLOW_ORDER = ["baseline_only", "baseline_plus_gate"]
WORKFLOW_LABELS = {
    "baseline_only": "Baseline only",
    "baseline_plus_gate": "Baseline + EcoOOD rule",
}
WORKFLOW_COLORS = {
    "baseline_only": PALETTE["slate"],
    "baseline_plus_gate": PALETTE["blue"],
}
ACTION_ORDER = ["screen_now", "prioritize_testing", "withhold_review", "lower_priority"]
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
    "lower_priority": "#BFB8AE",
}
SPLIT_LABELS = {
    "scaffold": "Scaffold",
    "temporal": "Temporal",
    "species": "Species",
    "chemical_class": "Class Holdout",
    "hard_ood": "Hard OOD",
}


def load_prediction_panel(
    structured_dir: Path,
    scaffold_dir: Path,
    hard_ood_dir: Path,
    structured_xgb_dir: Path | None = None,
    scaffold_xgb_dir: Path | None = None,
    hard_ood_xgb_dir: Path | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in DEPLOYMENT_SPLITS:
        for model in MODEL_ORDER:
            if model == "xgboost":
                if split == "hard_ood":
                    root = hard_ood_xgb_dir or hard_ood_dir
                elif split == "scaffold":
                    root = scaffold_xgb_dir or scaffold_dir
                else:
                    root = structured_xgb_dir or structured_dir
            else:
                if split == "hard_ood":
                    root = hard_ood_dir
                elif split == "scaffold":
                    root = scaffold_dir
                else:
                    root = structured_dir
            path = root / split / model / "predictions.csv"
            frame = pd.read_csv(
                path,
                usecols=[
                    "chemical_id",
                    "chemical_name",
                    "casrn",
                    "chemical_class",
                    "endpoint",
                    "y_true",
                    "y_pred",
                    "ecoood_score",
                ],
            )
            frame["split"] = split
            frame["model"] = model
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_chemical_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    chemical_panel = (
        predictions.groupby(
            ["model", "split", "chemical_id", "chemical_name", "casrn", "chemical_class"],
            dropna=False,
            as_index=False,
        )
        .agg(
            min_true_tox=("y_true", "min"),
            min_pred_tox=("y_pred", "min"),
            max_ecoood=("ecoood_score", "max"),
            endpoint_breadth=("endpoint", "nunique"),
            row_count=("chemical_id", "size"),
        )
    )
    return chemical_panel


def classify_screening_actions(
    chemical_panel: pd.DataFrame,
    toxicity_cutoff: float,
    ood_cutoff_by_model: dict[str, float],
) -> pd.DataFrame:
    frame = chemical_panel.copy()
    frame["toxicity_cutoff"] = toxicity_cutoff
    frame["ood_cutoff"] = frame["model"].map(ood_cutoff_by_model)
    frame["true_high_concern"] = frame["min_true_tox"] <= toxicity_cutoff
    frame["pred_high_concern"] = frame["min_pred_tox"] <= toxicity_cutoff
    frame["high_ood"] = frame["max_ecoood"] >= frame["ood_cutoff"]

    frame["baseline_action"] = "lower_priority"
    frame.loc[frame["pred_high_concern"], "baseline_action"] = "screen_now"

    frame["gated_action"] = "lower_priority"
    frame.loc[frame["pred_high_concern"] & ~frame["high_ood"], "gated_action"] = "screen_now"
    frame.loc[frame["pred_high_concern"] & frame["high_ood"], "gated_action"] = "prioritize_testing"
    frame.loc[~frame["pred_high_concern"] & frame["high_ood"], "gated_action"] = "withhold_review"

    frame["baseline_false_reassurance"] = frame["true_high_concern"] & (frame["baseline_action"] == "lower_priority")
    frame["rescued_by_gate"] = frame["baseline_false_reassurance"] & (frame["gated_action"] != "lower_priority")
    return frame


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def summarize_screening_gate(classified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for model in MODEL_ORDER:
        model_frame = classified[classified["model"] == model]
        if model_frame.empty:
            continue
        tox_cutoff = float(model_frame["toxicity_cutoff"].iloc[0])
        ood_cutoff = float(model_frame["ood_cutoff"].iloc[0])

        for split_name, split_frame in list(model_frame.groupby("split")) + [("pooled", model_frame)]:
            true_high = split_frame["true_high_concern"]
            baseline_screen = split_frame["baseline_action"] == "screen_now"
            gated_screen = split_frame["gated_action"] == "screen_now"
            baseline_low = split_frame["baseline_action"] == "lower_priority"
            gated_low = split_frame["gated_action"] == "lower_priority"

            base_false_neg = true_high & baseline_low
            rescued = split_frame["rescued_by_gate"]

            metric_rows.extend(
                [
                    {
                        "model": model,
                        "split": split_name,
                        "workflow": "baseline_only",
                        "metric": "screen_now_precision",
                        "value": _safe_rate((true_high & baseline_screen).sum(), baseline_screen.sum()),
                        "n_chemicals": int(len(split_frame)),
                    },
                    {
                        "model": model,
                        "split": split_name,
                        "workflow": "baseline_plus_gate",
                        "metric": "screen_now_precision",
                        "value": _safe_rate((true_high & gated_screen).sum(), gated_screen.sum()),
                        "n_chemicals": int(len(split_frame)),
                    },
                    {
                        "model": model,
                        "split": split_name,
                        "workflow": "baseline_only",
                        "metric": "false_reassurance_rate",
                        "value": _safe_rate(base_false_neg.sum(), baseline_low.sum()),
                        "n_chemicals": int(len(split_frame)),
                    },
                    {
                        "model": model,
                        "split": split_name,
                        "workflow": "baseline_plus_gate",
                        "metric": "false_reassurance_rate",
                        "value": _safe_rate((true_high & gated_low).sum(), gated_low.sum()),
                        "n_chemicals": int(len(split_frame)),
                    },
                ]
            )

            summary_rows.append(
                {
                    "model": model,
                    "split": split_name,
                    "toxicity_cutoff": tox_cutoff,
                    "ood_cutoff": ood_cutoff,
                    "n_chemicals": int(len(split_frame)),
                    "true_high_concern": int(true_high.sum()),
                    "baseline_screen_now": int(baseline_screen.sum()),
                    "baseline_lower_priority": int(baseline_low.sum()),
                    "gated_screen_now": int(gated_screen.sum()),
                    "gated_prioritize_testing": int((split_frame["gated_action"] == "prioritize_testing").sum()),
                    "gated_withhold_review": int((split_frame["gated_action"] == "withhold_review").sum()),
                    "gated_lower_priority": int(gated_low.sum()),
                    "baseline_screen_now_precision": _safe_rate((true_high & baseline_screen).sum(), baseline_screen.sum()),
                    "gated_screen_now_precision": _safe_rate((true_high & gated_screen).sum(), gated_screen.sum()),
                    "baseline_false_reassurance_rate": _safe_rate(base_false_neg.sum(), baseline_low.sum()),
                    "gated_false_reassurance_rate": _safe_rate((true_high & gated_low).sum(), gated_low.sum()),
                    "baseline_false_negatives": int(base_false_neg.sum()),
                    "rescued_false_negatives": int(rescued.sum()),
                    "rescued_false_negative_fraction": _safe_rate(rescued.sum(), base_false_neg.sum()),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    return metrics, summary


def representative_examples(classified: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    candidates = classified[classified["rescued_by_gate"]].copy()
    candidates = candidates.sort_values(
        ["model", "split", "max_ecoood", "min_true_tox"],
        ascending=[True, True, False, True],
    )
    examples = (
        candidates.groupby("model", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return examples[
        [
            "model",
            "split",
            "chemical_name",
            "casrn",
            "chemical_class",
            "row_count",
            "endpoint_breadth",
            "min_true_tox",
            "min_pred_tox",
            "max_ecoood",
            "baseline_action",
            "gated_action",
        ]
    ]


def summarize_diverted_destinations(classified: pd.DataFrame) -> pd.DataFrame:
    diverted = classified[classified["rescued_by_gate"]].copy()
    if diverted.empty:
        return pd.DataFrame(
            columns=["model", "split", "diverted_action", "rescued_count", "share_within_model_rescued"]
        )

    totals = diverted.groupby("model").size().rename("model_total")
    summary = (
        diverted.groupby(["model", "split", "gated_action"], as_index=False)
        .size()
        .rename(columns={"gated_action": "diverted_action", "size": "rescued_count"})
    )
    summary["model_total"] = summary["model"].map(totals)
    summary["share_within_model_rescued"] = summary["rescued_count"] / summary["model_total"]
    return summary.drop(columns=["model_total"]).sort_values(
        ["model", "rescued_count", "split", "diverted_action"], ascending=[True, False, True, True]
    )


def summarize_diverted_classes(classified: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    diverted = classified[classified["rescued_by_gate"]].copy()
    if diverted.empty:
        return pd.DataFrame(
            columns=["model", "chemical_class", "rescued_count", "n_splits", "n_unique_chemicals"]
        )

    diverted["chemical_class"] = diverted["chemical_class"].fillna("unclassified")
    class_summary = (
        diverted.groupby(["model", "chemical_class"], as_index=False)
        .agg(
            rescued_count=("chemical_id", "size"),
            n_splits=("split", "nunique"),
            n_unique_chemicals=("chemical_id", "nunique"),
        )
        .sort_values(["model", "rescued_count", "n_unique_chemicals"], ascending=[True, False, False])
    )
    trimmed = class_summary.groupby("model", group_keys=False).head(top_n).reset_index(drop=True)
    return trimmed


def summarize_named_diverted_classes(classified: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    diverted = classified[classified["rescued_by_gate"]].copy()
    if diverted.empty:
        return pd.DataFrame(
            columns=["model", "chemical_class", "rescued_count", "n_splits", "n_unique_chemicals"]
        )

    diverted["chemical_class"] = diverted["chemical_class"].fillna("unclassified")
    named = diverted[diverted["chemical_class"].str.lower() != "unclassified"].copy()
    if named.empty:
        return pd.DataFrame(
            columns=["model", "chemical_class", "rescued_count", "n_splits", "n_unique_chemicals"]
        )
    class_summary = (
        named.groupby(["model", "chemical_class"], as_index=False)
        .agg(
            rescued_count=("chemical_id", "size"),
            n_splits=("split", "nunique"),
            n_unique_chemicals=("chemical_id", "nunique"),
        )
        .sort_values(["model", "rescued_count", "n_unique_chemicals"], ascending=[True, False, False])
    )
    return class_summary.groupby("model", group_keys=False).head(top_n).reset_index(drop=True)


def action_mix(summary: pd.DataFrame) -> pd.DataFrame:
    pooled = summary[summary["split"] == "pooled"].copy()
    rows: list[dict[str, str | int]] = []
    for _, row in pooled.iterrows():
        rows.extend(
            [
                {
                    "model": row["model"],
                    "workflow": "baseline_only",
                    "action": "screen_now",
                    "n_chemicals": int(row["baseline_screen_now"]),
                },
                {
                    "model": row["model"],
                    "workflow": "baseline_only",
                    "action": "lower_priority",
                    "n_chemicals": int(row["baseline_lower_priority"]),
                },
                {
                    "model": row["model"],
                    "workflow": "baseline_plus_gate",
                    "action": "screen_now",
                    "n_chemicals": int(row["gated_screen_now"]),
                },
                {
                    "model": row["model"],
                    "workflow": "baseline_plus_gate",
                    "action": "prioritize_testing",
                    "n_chemicals": int(row["gated_prioritize_testing"]),
                },
                {
                    "model": row["model"],
                    "workflow": "baseline_plus_gate",
                    "action": "withhold_review",
                    "n_chemicals": int(row["gated_withhold_review"]),
                },
                {
                    "model": row["model"],
                    "workflow": "baseline_plus_gate",
                    "action": "lower_priority",
                    "n_chemicals": int(row["gated_lower_priority"]),
                },
            ]
        )
    mix = pd.DataFrame(rows)
    mix["workflow"] = pd.Categorical(mix["workflow"], categories=WORKFLOW_ORDER, ordered=True)
    mix["action"] = pd.Categorical(mix["action"], categories=ACTION_ORDER, ordered=True)
    mix["model"] = pd.Categorical(mix["model"], categories=MODEL_ORDER, ordered=True)
    return mix.sort_values(["model", "workflow", "action"]).reset_index(drop=True)


def plot_screening_gate_validation(metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(ACS_DOUBLE_WIDTH, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.0, 1.0])

    ax_false = fig.add_subplot(gs[0, 0])
    false_df = metrics[metrics["metric"] == "false_reassurance_rate"].copy()
    false_df = false_df[false_df["split"] != "pooled"].copy()
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
        palette={WORKFLOW_LABELS[k]: WORKFLOW_COLORS[k] for k in WORKFLOW_ORDER},
        estimator=None,
        ax=ax_false,
    )
    ax_false.set_ylabel("High-concern fraction within lower-priority outputs")
    ax_false.set_xlabel("")
    ax_false.set_title("Reliability assessment lowers false reassurance", pad=6)
    ax_false.tick_params(axis="x", rotation=20)
    finish_axis(ax_false, grid_axis="y")
    add_panel_label(ax_false, "A", x=-0.15, y=1.08)
    ax_false.legend(loc="upper right", ncol=1, frameon=False, fontsize=6.7, title="")

    ax_rescue = fig.add_subplot(gs[0, 1])
    rescue_df = summary[summary["split"] != "pooled"].copy()
    rescue_df["split_label"] = rescue_df["split"].map(SPLIT_LABELS)
    rescue_df["model_label"] = rescue_df["model"].map(MODEL_LABELS)
    sns.barplot(
        data=rescue_df,
        x="split_label",
        y="rescued_false_negative_fraction",
        hue="model_label",
        palette={MODEL_LABELS[key]: MODEL_COLORS[key] for key in MODEL_ORDER},
        ax=ax_rescue,
    )
    ax_rescue.set_ylabel("Baseline false negatives reassigned by EcoOOD")
    ax_rescue.set_xlabel("")
    ax_rescue.set_title("EcoOOD reassigns baseline misses", pad=6)
    ax_rescue.tick_params(axis="x", rotation=20)
    finish_axis(ax_rescue, grid_axis="y")
    add_panel_label(ax_rescue, "B", x=-0.15, y=1.08)
    ax_rescue.legend(loc="upper right", frameon=False, fontsize=6.7, title="")

    ax_mix = fig.add_subplot(gs[1, :])
    mix = action_mix(summary)
    x_positions = []
    xticklabels = []
    pos = 0
    gap = 0.45
    for model in MODEL_ORDER:
        for workflow in WORKFLOW_ORDER:
            x_positions.append(pos)
            xticklabels.append(f"{MODEL_LABELS[model]}\n{WORKFLOW_LABELS[workflow]}")
            pos += 1
        pos += gap
    bottoms = [0.0] * len(x_positions)
    combo_order = [(m, w) for m in MODEL_ORDER for w in WORKFLOW_ORDER]
    for action in ACTION_ORDER:
        heights = []
        for model, workflow in combo_order:
            row = mix[(mix["model"] == model) & (mix["workflow"] == workflow) & (mix["action"] == action)]
            heights.append(float(row["n_chemicals"].iloc[0]) if not row.empty else 0.0)
        ax_mix.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=ACTION_COLORS[action],
            edgecolor="white",
            linewidth=0.7,
            label=ACTION_LABELS[action],
            width=0.74,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    pooled = summary[summary["split"] == "pooled"].set_index("model")
    for idx, model in enumerate(MODEL_ORDER):
        base_x = idx * 2 + idx * gap
        gate_x = base_x + 1
        if model in pooled.index:
            base_false = pooled.loc[model, "baseline_false_reassurance_rate"]
            gate_false = pooled.loc[model, "gated_false_reassurance_rate"]
            ax_mix.text(base_x, bottoms[idx * 2] + 7, f"FR {base_false:.2f}", ha="center", va="bottom", fontsize=6.7, color=PALETTE["ink"])
            ax_mix.text(gate_x, bottoms[idx * 2 + 1] + 7, f"FR {gate_false:.2f}", ha="center", va="bottom", fontsize=6.7, color=PALETTE["ink"])

    ax_mix.set_xticks(x_positions, xticklabels)
    ax_mix.set_ylabel("Unique chemicals in pooled deployment panel")
    ax_mix.set_xlabel("")
    ax_mix.set_title("Baseline propagation versus EcoOOD-supported actions", pad=6)
    finish_axis(ax_mix, grid_axis="y")
    add_panel_label(ax_mix, "C", x=-0.04, y=1.06)
    ax_mix.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=4, frameon=False, fontsize=6.8)

    save_figure(fig, output_dir, "Figure_S3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrospective screening-gate validation for EcoOOD.")
    parser.add_argument(
        "--structured-dir",
        type=Path,
        default=Path("outputs/benchmark_1000chem_dsstox_mech_structured_ad"),
    )
    parser.add_argument(
        "--scaffold-dir",
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
        "--scaffold-xgb-dir",
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_prediction_panel(
        args.structured_dir,
        args.scaffold_dir,
        args.hard_ood_dir,
        args.structured_xgb_dir,
        args.scaffold_xgb_dir,
        args.hard_ood_xgb_dir,
    )
    chemical_panel = aggregate_chemical_panel(predictions)
    tox_cutoff = float(chemical_panel["min_true_tox"].quantile(0.25))
    ood_cutoff_by_model = (
        chemical_panel.groupby("model")["max_ecoood"].quantile(0.75).to_dict()
    )
    classified = classify_screening_actions(chemical_panel, tox_cutoff, ood_cutoff_by_model)
    metrics, summary = summarize_screening_gate(classified)
    examples = representative_examples(classified)
    diverted_destinations = summarize_diverted_destinations(classified)
    diverted_classes = summarize_diverted_classes(classified)
    named_diverted_classes = summarize_named_diverted_classes(classified)

    chemical_panel.to_csv(args.output_dir / "screening_gate_validation_panel.csv", index=False)
    classified.to_csv(args.output_dir / "screening_gate_validation_classified.csv", index=False)
    metrics.to_csv(args.output_dir / "screening_gate_validation_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "screening_gate_validation_summary.csv", index=False)
    examples.to_csv(args.output_dir / "screening_gate_validation_examples.csv", index=False)
    diverted_destinations.to_csv(args.output_dir / "screening_gate_validation_diverted_destinations.csv", index=False)
    diverted_classes.to_csv(args.output_dir / "screening_gate_validation_diverted_classes.csv", index=False)
    named_diverted_classes.to_csv(args.output_dir / "screening_gate_validation_named_diverted_classes.csv", index=False)

    plot_screening_gate_validation(metrics, summary, args.output_dir)


if __name__ == "__main__":
    main()
