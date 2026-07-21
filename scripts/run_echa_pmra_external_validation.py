from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from ecoood.ad import ApplicabilityDomainScorer
from ecoood.conformal import ScaledConformalRegressor
from ecoood.evaluation import ood_metrics, regression_metrics
from ecoood.features import EcoFeatureBuilder, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.ood import EcoOODScorer
from ecoood.schema import DEFAULT_SCHEMA

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv"
DEFAULT_PANEL_PATH = (
    ROOT
    / "results"
    / "external_regulatory_prep"
    / "echa_pmra_external_clean"
    / "echa_pmra_case_panel_main.csv"
)
DEFAULT_OUT_DIR = ROOT / "results" / "echa_pmra_external_validation_main"
MODEL_NAME = "lightgbm"
DEFAULT_SEEDS = [40, 41, 42, 43, 44]
REVIEW_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run case-level external validation on cleaned ECHA PMRA panels."
    )
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def write_status(out_dir: Path, message: str) -> None:
    text = message.rstrip()
    (out_dir / "run_status.txt").write_text(text + "\n")
    print(text, flush=True)


def calibration_split_by_chemical(
    df: pd.DataFrame,
    seed: int,
    calib_fraction: float = 0.125,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    group_sizes = df.groupby(DEFAULT_SCHEMA.chemical_id, dropna=False).size().sort_values(ascending=False)
    groups = group_sizes.index.to_list()
    rng.shuffle(groups)
    target_rows = max(1, int(round(len(df) * calib_fraction)))
    calib_groups: list[object] = []
    running = 0
    for group in groups:
        calib_groups.append(group)
        running += int(group_sizes.loc[group])
        if running >= target_rows:
            break
    calib_mask = df[DEFAULT_SCHEMA.chemical_id].isin(calib_groups)
    calib_df = df.loc[calib_mask].reset_index(drop=True)
    train_df = df.loc[~calib_mask].reset_index(drop=True)
    return train_df, calib_df


def fixed_burden_metrics(
    frame: pd.DataFrame,
    score_col: str,
    *,
    review_fraction: float = REVIEW_FRACTION,
) -> dict[str, float]:
    scores = frame[score_col].to_numpy(dtype=float)
    order = np.argsort(scores)[::-1]
    review_n = max(1, int(round(len(frame) * review_fraction)))
    review_idx = order[:review_n]
    review_mask = np.zeros(len(frame), dtype=bool)
    review_mask[review_idx] = True

    errors = frame["abs_error"].to_numpy(dtype=float)
    severe_threshold = float(np.quantile(errors, 0.9))
    severe_mask = errors >= severe_threshold
    return {
        "review_fraction": review_fraction,
        "review_n": int(review_mask.sum()),
        "mean_abs_error_review": float(errors[review_mask].mean()) if review_mask.any() else float("nan"),
        "mean_abs_error_propagate": float(errors[~review_mask].mean()) if (~review_mask).any() else float("nan"),
        "severe_error_capture_rate": float(review_mask[severe_mask].mean()) if severe_mask.any() else float("nan"),
        "propagate_severe_error_rate": float(severe_mask[~review_mask].mean()) if (~review_mask).any() else float("nan"),
    }


def summarize_seed(seed: int, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    burden_rows: list[dict[str, object]] = []
    methods = {
        "ecoood": "ecoood_score",
        "distance_to_model": "ad_distance_to_model",
        "similarity_ad_risk": "ad_similarity_risk",
    }
    grouped = list(frame.groupby("endpoint", sort=False)) + [("pooled", frame)]
    for endpoint_name, subset in grouped:
        if subset.empty:
            continue
        y_true = subset["target_log_molar"].to_numpy(dtype=float)
        y_pred = subset["y_pred"].to_numpy(dtype=float)
        base = {
            "seed": seed,
            "endpoint": endpoint_name,
            "n_cases": int(len(subset)),
            "n_chemicals": int(subset["chemical_id"].nunique()),
            **regression_metrics(y_true, y_pred),
        }
        for method_name, score_col in methods.items():
            score = subset[score_col].to_numpy(dtype=float)
            direct = ood_metrics(y_true, y_pred, score, None)
            metric_rows.append(
                {
                    **base,
                    "method": method_name,
                    "score_col": score_col,
                    "aurc": direct["aurc"],
                    "catastrophic_error_capture_rate": direct["catastrophic_error_capture_rate"],
                }
            )
            burden = fixed_burden_metrics(
                subset,
                score_col,
                review_fraction=REVIEW_FRACTION,
            )
            burden_rows.append(
                {
                    "seed": seed,
                    "endpoint": endpoint_name,
                    "method": method_name,
                    "score_col": score_col,
                    **burden,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(burden_rows)


def aggregate_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    value_cols = [col for col in frame.columns if col not in group_cols]
    summary = (
        frame.groupby(group_cols, as_index=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        col if isinstance(col, str) else "_".join([piece for piece in col if piece]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return summary


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_status(out_dir, "Loading train benchmark and cleaned external panel.")
    train_df = attach_rdkit_descriptors(pd.read_csv(DATA_PATH), DEFAULT_SCHEMA)
    panel = attach_rdkit_descriptors(pd.read_csv(args.panel_path), DEFAULT_SCHEMA)

    seed_predictions: list[pd.DataFrame] = []
    metrics_frames: list[pd.DataFrame] = []
    burden_frames: list[pd.DataFrame] = []

    for idx, seed in enumerate(args.seeds, start=1):
        write_status(out_dir, f"Running seed {seed} ({idx}/{len(args.seeds)})")
        train_seed_df, calib_df = calibration_split_by_chemical(train_df, seed=seed)

        feature_builder = EcoFeatureBuilder(schema=DEFAULT_SCHEMA)
        train_bundle = feature_builder.fit_transform(train_seed_df)
        calib_bundle = feature_builder.transform(calib_df)
        external_bundle = feature_builder.transform(panel)

        model = BootstrapEnsembleRegressor(
            model_name=MODEL_NAME,
            n_members=5,
            seed=seed,
        ).fit(train_bundle.full, train_seed_df[DEFAULT_SCHEMA.target].to_numpy())
        calib_pred = model.predict(calib_bundle.full)
        external_pred = model.predict(external_bundle.full)

        conformal = ScaledConformalRegressor(alpha=0.1).fit(
            calib_df[DEFAULT_SCHEMA.target].to_numpy(),
            calib_pred.mean,
            scale=np.maximum(calib_pred.std, 1e-3),
        )
        calib_interval = conformal.predict(calib_pred.mean, scale=np.maximum(calib_pred.std, 1e-3))
        external_interval = conformal.predict(external_pred.mean, scale=np.maximum(external_pred.std, 1e-3))

        scorer = EcoOODScorer(schema=DEFAULT_SCHEMA).fit(train_seed_df, train_bundle)
        calib_components = scorer.component_frame(
            calib_df,
            calib_bundle,
            model_std=calib_pred.std,
            interval_width=calib_interval.width,
        )
        scorer.fit_meta(
            calib_components,
            residuals=np.abs(calib_df[DEFAULT_SCHEMA.target].to_numpy() - calib_pred.mean),
        )
        external_components = scorer.predict(
            panel,
            external_bundle,
            model_std=external_pred.std,
            interval_width=external_interval.width,
        )

        ad_scorer = ApplicabilityDomainScorer().fit(train_bundle)
        external_ad = ad_scorer.predict(
            external_bundle,
            model_std=external_pred.std,
            interval_width=external_interval.width,
        )

        pred = panel[
            [
                "case_id",
                "chemical_id",
                "chemical_name",
                "casrn",
                "endpoint",
                "species",
                "target_log_molar",
                "case_row_count",
                "document_count",
                "case_spread_log_molar",
                "standard_species_flag",
                "freshwater_keyword_flag",
            ]
        ].copy()
        pred["seed"] = seed
        pred["y_pred"] = external_pred.mean
        pred["abs_error"] = np.abs(pred["y_pred"] - pred["target_log_molar"])
        pred["model_std"] = external_pred.std
        pred["interval_width"] = external_interval.width
        pred["ecoood_score"] = external_components.ecoood_score
        pred["ad_similarity"] = external_ad.similarity
        pred["ad_distance_to_model"] = external_ad.distance_to_model
        pred["ad_similarity_risk"] = pred["ad_similarity"]
        seed_predictions.append(pred)
        pred.to_csv(out_dir / f"external_predictions_seed_{seed}.csv", index=False)

        metric_frame, burden_frame = summarize_seed(seed, pred)
        metrics_frames.append(metric_frame)
        burden_frames.append(burden_frame)

    predictions = pd.concat(seed_predictions, ignore_index=True)
    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    burden_all = pd.concat(burden_frames, ignore_index=True)

    predictions.to_csv(out_dir / "external_predictions_all.csv", index=False)
    metrics_all.to_csv(out_dir / "external_metrics_all.csv", index=False)
    burden_all.to_csv(out_dir / "external_burden_all.csv", index=False)

    metrics_summary = aggregate_summary(metrics_all, ["endpoint", "method", "score_col"])
    burden_summary = aggregate_summary(burden_all, ["endpoint", "method", "score_col"])
    metrics_summary.to_csv(out_dir / "external_metrics_summary.csv", index=False)
    burden_summary.to_csv(out_dir / "external_burden_summary.csv", index=False)

    case_summary = (
        predictions.groupby(
            ["case_id", "chemical_id", "chemical_name", "casrn", "endpoint", "species"],
            as_index=False,
        )
        .agg(
            target_log_molar=("target_log_molar", "first"),
            case_row_count=("case_row_count", "first"),
            document_count=("document_count", "first"),
            case_spread_log_molar=("case_spread_log_molar", "first"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_std=("y_pred", "std"),
            abs_error_mean=("abs_error", "mean"),
            abs_error_std=("abs_error", "std"),
            ecoood_score_mean=("ecoood_score", "mean"),
            distance_to_model_mean=("ad_distance_to_model", "mean"),
            similarity_risk_mean=("ad_similarity_risk", "mean"),
        )
    )
    case_summary.to_csv(out_dir / "external_case_summary.csv", index=False)

    lines = [
        f"Panel path: {args.panel_path}",
        f"Cases: {panel.shape[0]}",
        f"Chemicals: {panel['chemical_name'].nunique()}",
        "Endpoint counts:",
        panel["endpoint"].value_counts().to_string(),
    ]
    (out_dir / "external_validation_notes.txt").write_text("\n".join(lines) + "\n")
    write_status(out_dir, "Completed ECHA PMRA external validation.")


if __name__ == "__main__":
    main()
