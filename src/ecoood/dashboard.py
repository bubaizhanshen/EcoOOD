from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import ParserError


OUTPUT_TABLES_DIR = Path("outputs") / "release_tables"
DEMO_DATA = Path("data") / "processed" / "demo_ecoood.csv"

DECISION_COLOR_MAP = {
    "predict": "#009E73",
    "warn": "#E69F00",
    "abstain": "#D55E00",
    "reliable_screening_concern": "#0072B2",
    "false_reassurance_warning": "#CC79A7",
    "testing_required": "#D55E00",
    "lower_priority": "#7F8C8D",
    "screen_now": "#0072B2",
    "prioritize_testing": "#E69F00",
    "withhold_review": "#D55E00",
}


@dataclass(frozen=True)
class DashboardBundle:
    benchmark_metrics: pd.DataFrame
    decision_points: pd.DataFrame
    screening_panel: pd.DataFrame
    screening_examples: pd.DataFrame
    gate_summary: pd.DataFrame
    gate_examples: pd.DataFrame


def _load_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", **kwargs)


def _release_tables_available(root: Path) -> bool:
    figures = root / OUTPUT_TABLES_DIR
    required = [
        "figure3_source_metrics.csv",
        "figure6_source_predictions.csv",
        "policy_relevant_screening_panel.csv",
        "policy_relevant_screening_examples.csv",
        "screening_gate_validation_summary.csv",
        "screening_gate_validation_examples.csv",
    ]
    return all((figures / name).exists() for name in required)


def _demo_decision_points(root: Path) -> pd.DataFrame:
    demo_path = root / DEMO_DATA
    if not demo_path.exists():
        raise FileNotFoundError(
            "Dashboard data were not found. Generate dashboard tables locally "
            f"or keep the demo table at {DEMO_DATA}."
        )
    df = _load_csv(demo_path).copy()
    centered = df["target_log_molar"] - df["target_log_molar"].median()
    spread = float(centered.abs().quantile(0.95)) or 1.0
    df["y_pred"] = df["target_log_molar"] + 0.08 * np.sin(np.arange(len(df)))
    df["interval_width"] = 0.12 + 0.22 * df["known_ood"].astype(float) + 0.06 * np.abs(centered / spread)
    df["d_chem"] = np.clip((df["physchem_logp"].rank(pct=True) - 0.5).abs() * 2.0, 0.0, 1.0)
    df["d_species"] = df["species"].map(df["species"].value_counts(normalize=True)).rsub(1.0).fillna(0.5)
    df["d_context"] = np.clip((df["study_year"].rank(pct=True) - 0.5).abs() * 2.0, 0.0, 1.0)
    df["d_mech"] = np.clip(df["mech_hit_rate"].rank(pct=True), 0.0, 1.0)
    df["ecoood_score"] = np.clip(
        0.28 * df["d_chem"]
        + 0.22 * df["d_species"]
        + 0.18 * df["d_context"]
        + 0.17 * df["d_mech"]
        + 0.15 * df["known_ood"].astype(float),
        0.0,
        1.0,
    )
    warn = float(df["ecoood_score"].quantile(0.55))
    abstain = float(df["ecoood_score"].quantile(0.88))
    df["decision"] = np.where(df["ecoood_score"] >= abstain, "abstain", np.where(df["ecoood_score"] >= warn, "warn", "predict"))
    split_names = np.array(["random", "scaffold", "temporal", "species", "chemical_class", "hard_ood"])
    df["split"] = split_names[np.arange(len(df)) % len(split_names)]
    return df


def _demo_benchmark_metrics(decision_points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, offset in [("lightgbm", 0.0), ("random_forest", 0.025)]:
        for split, group in decision_points.groupby("split", sort=False):
            err = group["target_log_molar"] - group["y_pred"]
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "rmse": float(np.sqrt(np.mean(err**2)) + offset),
                    "coverage": float(np.clip(0.92 - group["known_ood"].mean() * 0.12 - offset, 0.65, 0.98)),
                    "aurc": float(group["ecoood_score"].mean() * 0.12 + offset),
                    "abstain_fraction": float((group["decision"] == "abstain").mean()),
                }
            )
    return pd.DataFrame(rows)


def _demo_screening_panel(decision_points: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        decision_points.groupby(["chemical_id", "chemical_name", "chemical_class"], as_index=False)
        .agg(
            min_pred_tox=("y_pred", "min"),
            max_ecoood=("ecoood_score", "max"),
            n_rows=("chemical_id", "size"),
            endpoint_breadth=("endpoint", "nunique"),
            split_breadth=("split", "nunique"),
            abstain_fraction=("decision", lambda x: float((x == "abstain").mean())),
            warn_fraction=("decision", lambda x: float((x == "warn").mean())),
            median_interval=("interval_width", "median"),
        )
        .rename(columns={"chemical_class": "primary_class"})
    )
    tox_cut = float(grouped["min_pred_tox"].quantile(0.30))
    ood_cut = float(grouped["max_ecoood"].quantile(0.75))
    conditions = [
        (grouped["min_pred_tox"] <= tox_cut) & (grouped["max_ecoood"] >= ood_cut),
        (grouped["min_pred_tox"] <= tox_cut),
        grouped["max_ecoood"] >= ood_cut,
    ]
    labels = ["prioritize testing", "screen now", "withhold/review"]
    codes = ["prioritize_testing", "screen_now", "withhold_review"]
    grouped["screening_action_label"] = np.select(conditions, labels, default="lower priority")
    grouped["screening_action"] = np.select(conditions, codes, default="lower_priority")
    return grouped


def _demo_gate_summary(decision_points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["lightgbm", "random_forest"]:
        for split, group in decision_points.groupby("split", sort=False):
            high_concern = group["target_log_molar"] <= group["target_log_molar"].quantile(0.30)
            baseline_fn = high_concern & (group["y_pred"] > group["y_pred"].quantile(0.30))
            rescued = baseline_fn & (group["ecoood_score"] >= group["ecoood_score"].quantile(0.75))
            base_rate = float(baseline_fn.mean())
            gated_rate = float((baseline_fn & ~rescued).mean())
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "baseline_false_reassurance_rate": base_rate,
                    "gated_false_reassurance_rate": gated_rate,
                    "rescued_false_negative_fraction": float(rescued.sum() / max(int(baseline_fn.sum()), 1)),
                    "baseline_false_negatives": int(baseline_fn.sum()),
                    "rescued_false_negatives": int(rescued.sum()),
                }
            )
    return pd.DataFrame(rows)


def _load_demo_bundle(root: Path) -> DashboardBundle:
    decision_points = _demo_decision_points(root)
    screening_panel = _demo_screening_panel(decision_points)
    gate_summary = _demo_gate_summary(decision_points)
    return DashboardBundle(
        benchmark_metrics=_demo_benchmark_metrics(decision_points),
        decision_points=decision_points,
        screening_panel=screening_panel,
        screening_examples=screening_panel.head(12).copy(),
        gate_summary=gate_summary,
        gate_examples=gate_summary.head(12).copy(),
    )


def load_dashboard_bundle(root: str | Path = ".") -> DashboardBundle:
    root = Path(root)
    if not _release_tables_available(root):
        return _load_demo_bundle(root)
    figures = root / OUTPUT_TABLES_DIR
    return DashboardBundle(
        benchmark_metrics=_load_csv(figures / "figure3_source_metrics.csv"),
        decision_points=_load_csv(figures / "figure6_source_predictions.csv"),
        screening_panel=_load_csv(figures / "policy_relevant_screening_panel.csv"),
        screening_examples=_load_csv(figures / "policy_relevant_screening_examples.csv"),
        gate_summary=_load_csv(figures / "screening_gate_validation_summary.csv"),
        gate_examples=_load_csv(figures / "screening_gate_validation_examples.csv"),
    )


def decision_thresholds(df: pd.DataFrame) -> tuple[float, float]:
    score_threshold = float(df["ecoood_score"].quantile(0.64))
    toxicity_threshold = float(df["y_pred"].quantile(0.30))
    return score_threshold, toxicity_threshold


def screening_panel_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("screening_action_label", as_index=False)
        .agg(
            chemicals=("chemical_id", "nunique"),
            median_predicted_toxicity=("min_pred_tox", "median"),
            median_ecoood=("max_ecoood", "median"),
            median_interval=("median_interval", "median"),
        )
        .sort_values("chemicals", ascending=False)
    )
    return summary


def upload_ready_scores(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    aliases = {
        "prediction": "y_pred",
        "predicted_toxicity": "y_pred",
        "predicted_log_molar": "y_pred",
        "ood_score": "ecoood_score",
        "novelty_score": "ecoood_score",
        "interval": "interval_width",
    }
    renamed = renamed.rename(columns={k: v for k, v in aliases.items() if k in renamed.columns})
    required = {"y_pred", "ecoood_score"}
    missing = required - set(renamed.columns)
    if missing:
        raise ValueError(
            "Uploaded CSV must include at least 'y_pred' and 'ecoood_score'. "
            f"Missing: {', '.join(sorted(missing))}."
        )
    if "chemical_name" not in renamed.columns:
        renamed["chemical_name"] = [f"Chemical {i + 1}" for i in range(len(renamed))]
    if "interval_width" not in renamed.columns:
        renamed["interval_width"] = 0.0
    if "decision" not in renamed.columns:
        score_warn = float(renamed["ecoood_score"].quantile(0.50))
        score_abstain = float(renamed["ecoood_score"].quantile(0.85))
        width_warn = float(renamed["interval_width"].quantile(0.50))
        width_abstain = float(renamed["interval_width"].quantile(0.85))
        decisions: list[str] = []
        for _, row in renamed.iterrows():
            if row["ecoood_score"] >= score_abstain or row["interval_width"] >= width_abstain:
                decisions.append("abstain")
            elif row["ecoood_score"] >= score_warn or row["interval_width"] >= width_warn:
                decisions.append("warn")
            else:
                decisions.append("predict")
        renamed["decision"] = decisions
    return renamed


def top_flagged(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    sort_cols = [c for c in ["ecoood_score", "interval_width"] if c in df.columns]
    ascending = [False] * len(sort_cols)
    return df.sort_values(sort_cols, ascending=ascending).head(n).reset_index(drop=True)


def summary_cards(df: pd.DataFrame) -> dict[str, str]:
    score_threshold, toxicity_threshold = decision_thresholds(df)
    return {
        "rows": f"{len(df):,}",
        "chemicals": f"{df['chemical_id'].nunique():,}" if "chemical_id" in df.columns else "N/A",
        "predict_fraction": f"{(df['decision'] == 'predict').mean():.0%}" if "decision" in df.columns else "N/A",
        "high_ood_fraction": f"{(df['ecoood_score'] >= score_threshold).mean():.0%}",
        "toxicity_cutoff": f"{toxicity_threshold:.2f}",
        "ood_cutoff": f"{score_threshold:.2f}",
    }


def split_metric_table(df: pd.DataFrame, model: str = "lightgbm") -> pd.DataFrame:
    keep = df[df["model"] == model].copy()
    keep["split_label"] = keep["split"].str.replace("_", " ").str.title()
    return keep[
        [
            "split_label",
            "rmse",
            "coverage",
            "aurc",
            "abstain_fraction",
        ]
    ].rename(
        columns={
            "split_label": "Split",
            "rmse": "RMSE",
            "coverage": "Coverage",
            "aurc": "AURC",
            "abstain_fraction": "Diagnostic abstention fraction",
        }
    )


def gate_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = df.copy()
    keep["false_reassurance_delta"] = keep["baseline_false_reassurance_rate"] - keep["gated_false_reassurance_rate"]
    keep["rescued_fraction_pct"] = keep["rescued_false_negative_fraction"] * 100.0
    return keep[
        [
            "model",
            "split",
            "baseline_false_reassurance_rate",
            "gated_false_reassurance_rate",
            "false_reassurance_delta",
            "rescued_false_negative_fraction",
        ]
    ]
