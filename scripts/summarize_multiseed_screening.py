from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEPLOYMENT_SPLITS = ["scaffold", "temporal", "species", "chemical_class", "hard_ood"]
DECISION_MAP_SPLITS = ["temporal", "species", "chemical_class", "hard_ood"]
MODELS = ["lightgbm", "random_forest", "xgboost"]
METHOD_COLUMNS = {
    "EcoOOD": "max_ecoood",
    "Input-space kNN distance": "max_input_distance",
    "Similarity AD": "max_similarity_risk",
}
ACTION_ORDER = ["screen_now", "lower_priority", "withhold_review", "prioritize_testing"]
CLASS_ORDER = [
    "Conazoles",
    "Per- and Polyfluoroalkyl Substances (PFAS)",
    "Neonicotinoids",
    "Endocrine Disrupting Chemicals (EDCs)",
    "Pharmaceutical Personal Care Products (PPCPs)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed EcoOOD benchmark and fixed-workload screening results."
    )
    parser.add_argument("--input-root", type=Path, default=Path("outputs/corrected"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/corrected/aggregate"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--models", nargs="+", default=MODELS)
    return parser.parse_args()


def _group_summary(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    summary = frame.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns.to_flat_index()
    ]
    return summary


def _run_dir(root: Path, seed: int, split: str) -> Path:
    group = "hard_ood" if split == "hard_ood" else "structured"
    return root / f"seed_{seed}" / group


def load_core_outputs(
    root: Path,
    seeds: list[int],
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        for group in ["structured", "hard_ood"]:
            run_dir = root / f"seed_{seed}" / group
            metrics = pd.read_csv(run_dir / "benchmark_summary.csv")
            metrics = metrics[metrics["model"].isin(models)].copy()
            metrics["seed"] = seed
            metric_frames.append(metrics)

            scores = pd.read_csv(run_dir / "ood_score_summary.csv")
            scores = scores[scores["model"].isin(models)].copy()
            scores["seed"] = seed
            score_frames.append(scores)

        for split in DEPLOYMENT_SPLITS:
            run_dir = _run_dir(root, seed, split)
            for model in models:
                path = run_dir / split / model / "predictions.csv"
                frame = pd.read_csv(path)
                frame["seed"] = seed
                frame["split"] = split
                frame["model"] = model
                prediction_frames.append(frame)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(score_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def aggregate_chemical_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identifier_cols = [
        "seed",
        "model",
        "split",
        "chemical_id",
        "chemical_name",
        "casrn",
        "chemical_class",
    ]
    return (
        predictions.groupby(identifier_cols, dropna=False, as_index=False)
        .agg(
            min_true_tox=("y_true", "min"),
            min_pred_tox=("y_pred", "min"),
            max_ecoood=("ecoood_score", "max"),
            max_input_distance=("ad_distance_to_model", "max"),
            max_similarity_risk=("ad_similarity", "max"),
            endpoint_breadth=("endpoint", "nunique"),
            row_count=("chemical_id", "size"),
        )
    )


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _assign_actions(frame: pd.DataFrame, review_mask: pd.Series) -> pd.Series:
    action = pd.Series("lower_priority", index=frame.index, dtype=object)
    action.loc[frame["pred_high_concern"] & ~review_mask] = "screen_now"
    action.loc[~frame["pred_high_concern"] & review_mask] = "withhold_review"
    action.loc[frame["pred_high_concern"] & review_mask] = "prioritize_testing"
    return action


def summarize_direct_rule(chemical_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for (seed, model), group in chemical_panel.groupby(["seed", "model"], sort=False):
        frame = group.copy()
        toxicity_cutoff = float(frame["min_true_tox"].quantile(0.25))
        score_cutoff = float(frame["max_ecoood"].quantile(0.75))
        frame["toxicity_cutoff"] = toxicity_cutoff
        frame["score_cutoff"] = score_cutoff
        frame["true_high_concern"] = frame["min_true_tox"] <= toxicity_cutoff
        frame["pred_high_concern"] = frame["min_pred_tox"] <= toxicity_cutoff
        frame["reviewed"] = frame["max_ecoood"] >= score_cutoff
        frame["baseline_action"] = np.where(frame["pred_high_concern"], "screen_now", "lower_priority")
        frame["screening_action"] = _assign_actions(frame, frame["reviewed"])
        frame["baseline_false_negative"] = frame["true_high_concern"] & ~frame["pred_high_concern"]
        frame["rescued_false_negative"] = frame["baseline_false_negative"] & frame["reviewed"]
        classified_frames.append(frame)

        split_groups = list(frame.groupby("split", sort=False)) + [("pooled", frame)]
        for split, subset in split_groups:
            baseline_low = ~subset["pred_high_concern"]
            routed_low = subset["screening_action"].eq("lower_priority")
            baseline_fn = subset["baseline_false_negative"]
            rescued = subset["rescued_false_negative"]
            row: dict[str, object] = {
                "seed": seed,
                "model": model,
                "split": split,
                "n_chemical_split_cases": int(len(subset)),
                "toxicity_cutoff": toxicity_cutoff,
                "ecoood_cutoff": score_cutoff,
                "baseline_false_reassurance": _safe_rate(
                    int((subset["true_high_concern"] & baseline_low).sum()), int(baseline_low.sum())
                ),
                "routed_false_reassurance": _safe_rate(
                    int((subset["true_high_concern"] & routed_low).sum()), int(routed_low.sum())
                ),
                "baseline_false_negatives": int(baseline_fn.sum()),
                "rescued_false_negatives": int(rescued.sum()),
                "rescued_false_negative_fraction": _safe_rate(int(rescued.sum()), int(baseline_fn.sum())),
            }
            for action in ACTION_ORDER:
                row[f"{action}_count"] = int(subset["screening_action"].eq(action).sum())
            rows.append(row)
    return pd.concat(classified_frames, ignore_index=True), pd.DataFrame(rows)


def summarize_fixed_workload(
    chemical_panel: pd.DataFrame,
    burdens: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for (seed, model), group in chemical_panel.groupby(["seed", "model"], sort=False):
        frame = group.copy()
        toxicity_cutoff = float(frame["min_true_tox"].quantile(0.25))
        frame["true_high_concern"] = frame["min_true_tox"] <= toxicity_cutoff
        frame["pred_high_concern"] = frame["min_pred_tox"] <= toxicity_cutoff
        frame["baseline_false_negative"] = frame["true_high_concern"] & ~frame["pred_high_concern"]
        for burden in burdens:
            review_n = max(1, int(round(len(frame) * burden)))
            for method, score_col in METHOD_COLUMNS.items():
                ordered = frame.sort_values([score_col, "split", "chemical_id"], ascending=[False, True, True])
                reviewed_idx = ordered.head(review_n).index
                reviewed = frame.index.to_series().isin(reviewed_idx)
                classified = frame.copy()
                classified["method"] = method
                classified["review_burden"] = burden
                classified["reviewed"] = reviewed.to_numpy()
                classified["screening_action"] = _assign_actions(classified, classified["reviewed"])
                classified["rescued_false_negative"] = (
                    classified["baseline_false_negative"] & classified["reviewed"]
                )
                classification_frames.append(classified)

                lower = classified["screening_action"].eq("lower_priority")
                baseline_fn = classified["baseline_false_negative"]
                rescued = classified["rescued_false_negative"]
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "method": method,
                        "review_burden": burden,
                        "toxicity_cutoff": toxicity_cutoff,
                        "review_count": int(classified["reviewed"].sum()),
                        "lower_priority_false_reassurance": _safe_rate(
                            int((classified["true_high_concern"] & lower).sum()), int(lower.sum())
                        ),
                        "baseline_false_negatives": int(baseline_fn.sum()),
                        "rescued_false_negatives": int(rescued.sum()),
                        "rescued_false_negative_fraction": _safe_rate(
                            int(rescued.sum()), int(baseline_fn.sum())
                        ),
                        "rescued_per_100_reviews": _safe_rate(int(rescued.sum()) * 100, review_n),
                        "prioritize_testing_count": int(
                            classified["screening_action"].eq("prioritize_testing").sum()
                        ),
                        "withhold_review_count": int(
                            classified["screening_action"].eq("withhold_review").sum()
                        ),
                    }
                )
    return pd.concat(classification_frames, ignore_index=True), pd.DataFrame(rows)


def _primary_class(value: object) -> str | None:
    text = "" if pd.isna(value) else str(value)
    for class_name in CLASS_ORDER:
        if class_name in text:
            return class_name
    return None


def build_class_focused_panel(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions[
        predictions["model"].eq("lightgbm") & predictions["split"].isin(DECISION_MAP_SPLITS)
    ].copy()
    frame["primary_class"] = frame["chemical_class"].map(_primary_class)
    frame = frame[frame["primary_class"].notna()].copy()
    panel = (
        frame.groupby(["chemical_id", "chemical_name", "casrn"], dropna=False, as_index=False)
        .agg(
            primary_class=("primary_class", lambda values: values.mode().iat[0]),
            class_labels=("chemical_class", lambda values: "; ".join(sorted(set(values.astype(str))))),
            n_seeds=("seed", "nunique"),
            split_breadth=("split", "nunique"),
            endpoint_breadth=("endpoint", "nunique"),
            min_pred_tox=("y_pred", "min"),
            min_true_tox=("y_true", "min"),
            max_ecoood=("ecoood_score", "max"),
            median_ecoood=("ecoood_score", "median"),
        )
    )
    toxicity_cutoff = float(panel["min_pred_tox"].quantile(0.25))
    score_cutoff = float(panel["max_ecoood"].quantile(0.75))
    panel["toxicity_cutoff"] = toxicity_cutoff
    panel["ecoood_cutoff"] = score_cutoff
    panel["pred_high_concern"] = panel["min_pred_tox"] <= toxicity_cutoff
    panel["reviewed"] = panel["max_ecoood"] >= score_cutoff
    panel["screening_action"] = _assign_actions(panel, panel["reviewed"])
    counts = (
        panel.groupby(["primary_class", "screening_action"], as_index=False)
        .size()
        .rename(columns={"size": "n_chemicals"})
    )
    return panel, counts


def build_decision_map_summary(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions[
        predictions["model"].eq("lightgbm") & predictions["split"].isin(DECISION_MAP_SPLITS)
    ].copy()
    toxicity_cutoff = float(frame["y_pred"].quantile(0.25))
    score_cutoff = float(frame["ecoood_score"].quantile(0.75))
    frame["pred_high_concern"] = frame["y_pred"] <= toxicity_cutoff
    frame["reviewed"] = frame["ecoood_score"] >= score_cutoff
    frame["screening_action"] = _assign_actions(frame, frame["reviewed"])
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    severe_cutoff = float(frame["abs_error"].quantile(0.90))
    frame["top_decile_error"] = frame["abs_error"] >= severe_cutoff
    frame["true_high_concern"] = frame["y_true"] <= toxicity_cutoff
    frame["covered"] = (frame["y_true"] >= frame["interval_lower"]) & (
        frame["y_true"] <= frame["interval_upper"]
    )
    summary = (
        frame.groupby("screening_action", as_index=False)
        .agg(
            n_rows=("screening_action", "size"),
            rmse=("abs_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            coverage=("covered", "mean"),
            top_decile_error_rate=("top_decile_error", "mean"),
            measured_high_concern_fraction=("true_high_concern", "mean"),
        )
    )
    summary["toxicity_cutoff"] = toxicity_cutoff
    summary["ecoood_cutoff"] = score_cutoff
    return frame, summary


def build_threshold_sensitivity(decision_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for toxicity_q in [0.20, 0.25, 0.30]:
        for ecoood_q in [0.70, 0.75, 0.80]:
            frame = decision_rows.copy()
            toxicity_cutoff = float(frame["y_pred"].quantile(toxicity_q))
            score_cutoff = float(frame["ecoood_score"].quantile(ecoood_q))
            frame["pred_high_concern"] = frame["y_pred"] <= toxicity_cutoff
            frame["reviewed"] = frame["ecoood_score"] >= score_cutoff
            frame["action_tmp"] = _assign_actions(frame, frame["reviewed"])
            row: dict[str, object] = {
                "toxicity_quantile": toxicity_q,
                "ecoood_quantile": ecoood_q,
                "toxicity_cutoff": toxicity_cutoff,
                "ecoood_cutoff": score_cutoff,
            }
            for action in ACTION_ORDER:
                row[f"{action}_fraction"] = float(frame["action_tmp"].eq(action).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_values = [
        col for col in metrics.columns if col not in {"seed", "split", "model"} and pd.api.types.is_numeric_dtype(metrics[col])
    ]
    score_values = [
        col
        for col in scores.columns
        if col not in {"seed", "split", "model", "method"} and pd.api.types.is_numeric_dtype(scores[col])
    ]
    metrics.to_csv(output_dir / "benchmark_summary_all_seeds.csv", index=False)
    _group_summary(metrics, ["split", "model"], metric_values).to_csv(
        output_dir / "benchmark_summary_agg.csv", index=False
    )
    scores.to_csv(output_dir / "ood_score_summary_all_seeds.csv", index=False)
    _group_summary(scores, ["split", "model", "method"], score_values).to_csv(
        output_dir / "ood_score_summary_agg.csv", index=False
    )
    predictions.to_csv(output_dir / "predictions_all_seeds.csv", index=False)

    chemical_panel = aggregate_chemical_predictions(predictions)
    chemical_panel.to_csv(output_dir / "chemical_split_panel.csv", index=False)
    direct_rows, direct_summary = summarize_direct_rule(chemical_panel)
    direct_rows.to_csv(output_dir / "screening_rule_classified.csv", index=False)
    direct_summary.to_csv(output_dir / "screening_rule_summary_all_seeds.csv", index=False)
    direct_values = [
        col
        for col in direct_summary.columns
        if col not in {"seed", "model", "split"} and pd.api.types.is_numeric_dtype(direct_summary[col])
    ]
    _group_summary(direct_summary, ["model", "split"], direct_values).to_csv(
        output_dir / "screening_rule_summary_agg.csv", index=False
    )

    workload_rows, workload_summary = summarize_fixed_workload(chemical_panel)
    workload_rows.to_csv(output_dir / "review_workload_classified.csv", index=False)
    workload_summary.to_csv(output_dir / "review_workload_summary_all_seeds.csv", index=False)
    workload_values = [
        col
        for col in workload_summary.columns
        if col not in {"seed", "model", "method"} and pd.api.types.is_numeric_dtype(workload_summary[col])
    ]
    _group_summary(
        workload_summary,
        ["model", "method", "review_burden"],
        workload_values,
    ).to_csv(output_dir / "review_workload_summary_agg.csv", index=False)

    class_panel, class_counts = build_class_focused_panel(predictions)
    class_panel.to_csv(output_dir / "class_focused_screening_panel.csv", index=False)
    class_counts.to_csv(output_dir / "class_focused_screening_counts.csv", index=False)

    decision_rows, decision_summary = build_decision_map_summary(predictions)
    decision_rows.to_csv(output_dir / "decision_map_rows.csv", index=False)
    decision_summary.to_csv(output_dir / "decision_map_summary.csv", index=False)
    build_threshold_sensitivity(decision_rows).to_csv(
        output_dir / "decision_map_threshold_sensitivity.csv", index=False
    )


def main() -> None:
    args = parse_args()
    metrics, scores, predictions = load_core_outputs(args.input_root, args.seeds, args.models)
    write_outputs(args.output_dir, metrics, scores, predictions)
    print(f"Wrote multi-seed summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
