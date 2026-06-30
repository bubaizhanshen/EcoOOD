from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, add_panel_label, apply_publication_style, finish_axis, save_figure


DEPLOYMENT_SPLITS = ["scaffold", "temporal", "species", "chemical_class", "hard_ood"]
POLICY_CLASS_ORDER = [
    "Conazoles",
    "Per- and Polyfluoroalkyl Substances (PFAS)",
    "Neonicotinoids",
    "Endocrine Disrupting Chemicals (EDCs)",
    "Pharmaceutical Personal Care Products (PPCPs)",
]
ACTION_ORDER = [
    "testing_required",
    "false_reassurance_warning",
    "reliable_screening_concern",
    "lower_priority",
]
ACTION_LABELS = {
    "testing_required": "Testing required",
    "false_reassurance_warning": "False-reassurance warning",
    "reliable_screening_concern": "Reliable screening concern",
    "lower_priority": "Lower priority",
}
ACTION_COLORS = {
    "testing_required": PALETTE["red"],
    "false_reassurance_warning": PALETTE["orange"],
    "reliable_screening_concern": PALETTE["green"],
    "lower_priority": "#BFB8AE",
}


def _primary_class(label: str) -> str | None:
    label = "" if pd.isna(label) else str(label)
    for class_name in [
        "Neonicotinoids",
        "Conazoles",
        "Per- and Polyfluoroalkyl Substances (PFAS)",
        "Pharmaceutical Personal Care Products (PPCPs)",
        "Endocrine Disrupting Chemicals (EDCs)",
    ]:
        if class_name in label:
            return class_name
    return None


def _aggregate_panel(predictions: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    panel = predictions[predictions["split"].isin(DEPLOYMENT_SPLITS)].copy()
    panel["primary_class"] = panel["chemical_class"].apply(_primary_class)
    panel = panel[panel["primary_class"].notna()].copy()

    chemical_panel = (
        panel.groupby(["chemical_id", "chemical_name", "casrn"], dropna=False)
        .agg(
            class_labels=("chemical_class", lambda x: "; ".join(sorted(set(str(v) for v in x if pd.notna(v))))),
            primary_class=("primary_class", lambda x: x.mode().iat[0]),
            n_rows=("chemical_id", "size"),
            split_breadth=("split", "nunique"),
            endpoint_breadth=("endpoint", "nunique"),
            abstain_fraction=("decision", lambda x: float((x == "abstain").mean())),
            warn_fraction=("decision", lambda x: float((x == "warn").mean())),
            max_ecoood=("ecoood_score", "max"),
            median_ecoood=("ecoood_score", "median"),
            min_pred_tox=("y_pred", "min"),
            median_pred_tox=("y_pred", "median"),
            median_interval=("interval_width", "median"),
            temporal_rows=("split", lambda x: int((x == "temporal").sum())),
            later_year_rows=("study_year", lambda x: int((pd.to_numeric(x, errors="coerce") >= 2016).sum())),
            year_min=("study_year", "min"),
            year_max=("study_year", "max"),
        )
        .reset_index()
    )

    tox_cutoff = float(chemical_panel["min_pred_tox"].quantile(0.25))
    ood_cutoff = float(chemical_panel["max_ecoood"].quantile(0.75))

    def classify(row: pd.Series) -> str:
        if row["min_pred_tox"] <= tox_cutoff and row["max_ecoood"] >= ood_cutoff:
            return "testing_required"
        if row["min_pred_tox"] > tox_cutoff and row["max_ecoood"] >= ood_cutoff:
            return "false_reassurance_warning"
        if row["min_pred_tox"] <= tox_cutoff and row["max_ecoood"] < ood_cutoff:
            return "reliable_screening_concern"
        return "lower_priority"

    chemical_panel["screening_action"] = chemical_panel.apply(classify, axis=1)
    chemical_panel["screening_action_label"] = chemical_panel["screening_action"].map(ACTION_LABELS)
    return chemical_panel, tox_cutoff, ood_cutoff


def _representative_examples(chemical_panel: pd.DataFrame) -> pd.DataFrame:
    top_testing = chemical_panel[chemical_panel["screening_action"] == "testing_required"].copy()
    top_testing = top_testing.sort_values(["max_ecoood", "min_pred_tox"], ascending=[False, True]).head(4)
    top_testing["label"] = [
        "Conazole A",
        "PFAS amide",
        "Neonicotinoid",
        "Conazole B",
    ][: len(top_testing)]

    top_false = chemical_panel[chemical_panel["screening_action"] == "false_reassurance_warning"].copy()
    false_examples = []
    for class_name, label in [
        ("Per- and Polyfluoroalkyl Substances (PFAS)", "PFAS temporal proxy"),
        ("Neonicotinoids", "Neonicotinoid warning"),
    ]:
        subset = top_false[top_false["primary_class"] == class_name].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["later_year_rows", "max_ecoood"], ascending=[False, False]).head(1)
        subset["label"] = label
        false_examples.append(subset)
    top_false = pd.concat(false_examples, ignore_index=True) if false_examples else top_false.head(0)

    examples = pd.concat([top_testing, top_false], ignore_index=True)
    return examples


def _action_counts(chemical_panel: pd.DataFrame) -> pd.DataFrame:
    counts = (
        chemical_panel.groupby(["primary_class", "screening_action"], dropna=False)
        .size()
        .reset_index(name="n_chemicals")
    )
    counts["primary_class"] = pd.Categorical(counts["primary_class"], categories=POLICY_CLASS_ORDER, ordered=True)
    counts["screening_action"] = pd.Categorical(counts["screening_action"], categories=ACTION_ORDER, ordered=True)
    counts = counts.sort_values(["primary_class", "screening_action"]).reset_index(drop=True)
    return counts


def _plot_screening_panel(
    chemical_panel: pd.DataFrame,
    examples: pd.DataFrame,
    counts: pd.DataFrame,
    tox_cutoff: float,
    ood_cutoff: float,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(ACS_DOUBLE_WIDTH, 3.55),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    size_min = 36
    size_scale = 22
    for action in ACTION_ORDER[::-1]:
        frame = chemical_panel[chemical_panel["screening_action"] == action]
        if frame.empty:
            continue
        ax.scatter(
            frame["min_pred_tox"],
            frame["max_ecoood"],
            s=size_min + size_scale * frame["split_breadth"] * frame["endpoint_breadth"],
            c=ACTION_COLORS[action],
            alpha=0.82,
            linewidth=0.55,
            edgecolor="white",
            label=ACTION_LABELS[action],
            zorder=3,
        )

    ax.axvline(tox_cutoff, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    ax.axhline(ood_cutoff, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    ax.text(0.02, 0.10, "Reliable screening\nconcern", transform=ax.transAxes, color=PALETTE["green"], fontsize=7.1)
    ax.text(0.60, 0.10, "Lower priority", transform=ax.transAxes, color=PALETTE["slate"], fontsize=7.1)
    ax.text(0.60, 0.86, "False-reassurance\nwarning", transform=ax.transAxes, color=PALETTE["orange"], fontsize=7.1)
    ax.text(0.02, 0.86, "Testing required", transform=ax.transAxes, color=PALETTE["red"], fontsize=7.1)

    for _, row in examples.iterrows():
        ax.text(
            row["min_pred_tox"] + 0.06,
            row["max_ecoood"] + 0.008,
            row["label"],
            fontsize=6.8,
            color=PALETTE["ink"],
            ha="left",
            va="bottom",
            zorder=5,
        )

    ax.set_xlabel("Most toxic deployment prediction (log molar; left = more toxic)")
    ax.set_ylabel("Maximum EcoOOD score across deployment splits")
    ax.set_title("Policy-relevant chemical panel", pad=6)
    add_panel_label(ax, "A", x=-0.16, y=1.07)
    finish_axis(ax, grid_axis="both")
    ax.legend(loc="lower left", frameon=False, fontsize=6.8, ncol=2)

    ax = axes[1]
    pivot = (
        counts.pivot(index="primary_class", columns="screening_action", values="n_chemicals")
        .reindex(POLICY_CLASS_ORDER)
        .fillna(0)
    )
    left = pd.Series(0, index=pivot.index, dtype=float)
    for action in ACTION_ORDER:
        vals = pivot[action]
        ax.barh(
            pivot.index,
            vals,
            left=left,
            color=ACTION_COLORS[action],
            edgecolor="white",
            linewidth=0.6,
            label=ACTION_LABELS[action],
        )
        left = left + vals
    ax.set_xlabel("Unique chemicals")
    ax.set_ylabel("")
    ax.set_title("Action mix across policy-relevant classes", pad=6)
    add_panel_label(ax, "B", x=-0.16, y=1.07)
    finish_axis(ax, grid_axis="x")

    save_figure(fig, output_dir, "Figure_archived_policy_panel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a policy-relevant screening-panel add-on for EcoOOD.")
    parser.add_argument(
        "--prediction-pool",
        type=Path,
        default=Path("outputs/release_tables/figure6_source_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/release_tables"))
    args = parser.parse_args()

    apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.prediction_pool)
    chemical_panel, tox_cutoff, ood_cutoff = _aggregate_panel(predictions)
    examples = _representative_examples(chemical_panel)
    counts = _action_counts(chemical_panel)

    chemical_panel.to_csv(args.output_dir / "policy_relevant_screening_panel.csv", index=False)
    counts.to_csv(args.output_dir / "policy_relevant_screening_action_counts.csv", index=False)
    examples.to_csv(args.output_dir / "policy_relevant_screening_examples.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_chemicals": int(len(chemical_panel)),
                "toxicity_cutoff_q25": tox_cutoff,
                "ood_cutoff_q75": ood_cutoff,
                "testing_required": int((chemical_panel["screening_action"] == "testing_required").sum()),
                "false_reassurance_warning": int((chemical_panel["screening_action"] == "false_reassurance_warning").sum()),
                "reliable_screening_concern": int((chemical_panel["screening_action"] == "reliable_screening_concern").sum()),
                "lower_priority": int((chemical_panel["screening_action"] == "lower_priority").sum()),
            }
        ]
    )
    summary.to_csv(args.output_dir / "policy_relevant_screening_summary.csv", index=False)

    _plot_screening_panel(chemical_panel, examples, counts, tox_cutoff, ood_cutoff, args.output_dir)


if __name__ == "__main__":
    main()
