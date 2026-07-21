from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ad import ApplicabilityDomainScorer
from .conformal import ScaledConformalRegressor, decision_labels
from .evaluation import interval_metrics, ood_metrics, reference_ood_metrics, regression_metrics, save_metrics, save_predictions, score_method_metrics
from .features import EcoFeatureBuilder, attach_rdkit_descriptors
from .models import BootstrapEnsembleRegressor
from .ood import EcoOODScorer
from .schema import DEFAULT_SCHEMA, EcoOODSchema
from .splits import SplitIndices, build_split


@dataclass
class ExperimentConfig:
    split: str
    model_name: str = "lightgbm"
    alpha: float = 0.1
    seed: int = 42
    n_members: int = 5
    output_dir: str | None = None
    estimator_params: dict | None = None
    ensemble_n_jobs: int = -1


def _select(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return df.loc[idx].reset_index(drop=True)


def _known_ood_labels(df_test: pd.DataFrame, split_indices: SplitIndices, schema: EcoOODSchema) -> np.ndarray:
    if split_indices.split_name == "hard_ood":
        if schema.known_ood in df_test.columns:
            return df_test[schema.known_ood].fillna(False).astype(bool).to_numpy()
        if schema.hard_ood in df_test.columns:
            return df_test[schema.hard_ood].fillna(False).astype(bool).to_numpy()
    return split_indices.test_is_ood


def run_single_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
) -> tuple[dict[str, float], pd.DataFrame]:
    working = attach_rdkit_descriptors(df, schema)
    working = working[working[schema.target].notna()].reset_index(drop=True)
    split_indices = build_split(working, split=config.split, schema=schema, seed=config.seed)
    train_df = _select(working, split_indices.train)
    calib_df = _select(working, split_indices.calib)
    test_df = _select(working, split_indices.test)

    feature_builder = EcoFeatureBuilder(schema=schema)
    train_bundle = feature_builder.fit_transform(train_df)
    calib_bundle = feature_builder.transform(calib_df)
    test_bundle = feature_builder.transform(test_df)

    model = BootstrapEnsembleRegressor(
        model_name=config.model_name,
        n_members=config.n_members,
        seed=config.seed,
        n_jobs=config.ensemble_n_jobs,
        estimator_params=config.estimator_params,
    ).fit(train_bundle.full, train_df[schema.target].to_numpy())
    calib_pred = model.predict(calib_bundle.full)
    test_pred = model.predict(test_bundle.full)

    conformal = ScaledConformalRegressor(alpha=config.alpha).fit(
        calib_df[schema.target].to_numpy(),
        calib_pred.mean,
        scale=np.maximum(calib_pred.std, 1e-3),
    )
    calib_interval = conformal.predict(calib_pred.mean, scale=np.maximum(calib_pred.std, 1e-3))
    test_interval = conformal.predict(test_pred.mean, scale=np.maximum(test_pred.std, 1e-3))

    scorer = EcoOODScorer(schema=schema).fit(train_df, train_bundle)
    calib_components = scorer.component_frame(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=calib_interval.width,
    )
    scorer.fit_meta(
        calib_components,
        residuals=np.abs(calib_df[schema.target].to_numpy() - calib_pred.mean),
    )
    test_components = scorer.predict(
        test_df,
        test_bundle,
        model_std=test_pred.std,
        interval_width=test_interval.width,
    )
    calib_components_pred = scorer.predict(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=calib_interval.width,
    )

    ad_scorer = ApplicabilityDomainScorer().fit(train_bundle)
    calib_ad = ad_scorer.predict(
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=calib_interval.width,
    )
    test_ad = ad_scorer.predict(
        test_bundle,
        model_std=test_pred.std,
        interval_width=test_interval.width,
    )

    score_warn = float(np.quantile(scorer.meta_model.predict_proba(scorer.component_scaler.transform(calib_components))[:, 1], 0.5)) if scorer.meta_model is not None else float(np.quantile(scorer.component_scaler.transform(calib_components).mean(axis=1), 0.5))
    score_abstain = float(np.quantile(scorer.meta_model.predict_proba(scorer.component_scaler.transform(calib_components))[:, 1], 0.85)) if scorer.meta_model is not None else float(np.quantile(scorer.component_scaler.transform(calib_components).mean(axis=1), 0.85))
    width_warn = float(np.quantile(calib_interval.width, 0.5))
    width_abstain = float(np.quantile(calib_interval.width, 0.85))
    decisions = decision_labels(
        test_components.ecoood_score,
        test_interval.width,
        score_warn_threshold=score_warn,
        score_abstain_threshold=score_abstain,
        width_warn_threshold=width_warn,
        width_abstain_threshold=width_abstain,
    )

    y_test = test_df[schema.target].to_numpy()
    known_ood = _known_ood_labels(test_df, split_indices, schema)
    metrics = {
        "split": config.split,
        "model": config.model_name,
        **regression_metrics(y_test, test_pred.mean),
        **interval_metrics(
            y_test,
            test_interval.lower,
            test_interval.upper,
            uncertainty=test_pred.std,
            novelty=test_components.ecoood_score,
        ),
        **ood_metrics(
            y_test,
            test_pred.mean,
            test_components.ecoood_score,
            known_ood,
        ),
        **reference_ood_metrics(
            id_scores=calib_components_pred.ecoood_score,
            ood_scores=test_components.ecoood_score if np.any(split_indices.test_is_ood) else np.array([]),
        ),
        "predict_fraction": float(np.mean(decisions == "predict")),
        "warn_fraction": float(np.mean(decisions == "warn")),
        "abstain_fraction": float(np.mean(decisions == "abstain")),
    }

    predictions = test_df.copy()
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred.mean
    predictions["model_std"] = test_pred.std
    predictions["interval_lower"] = test_interval.lower
    predictions["interval_upper"] = test_interval.upper
    predictions["interval_width"] = test_interval.width
    predictions["ecoood_score"] = test_components.ecoood_score
    predictions["d_chem"] = test_components.chemical
    predictions["d_species"] = test_components.species
    predictions["d_context"] = test_components.context
    predictions["d_mech"] = test_components.mechanism
    predictions["decision"] = decisions
    predictions["known_ood"] = known_ood
    predictions["ad_similarity"] = test_ad.similarity
    predictions["ad_leverage"] = test_ad.leverage
    predictions["ad_range"] = test_ad.descriptor_range
    predictions["ad_distance_to_model"] = test_ad.distance_to_model
    predictions["uncertainty_interval_width_score"] = test_ad.interval_width
    predictions["ood_mahalanobis"] = test_ad.mahalanobis
    predictions["ood_isolation_forest"] = test_ad.isolation_forest
    predictions["ood_lof"] = test_ad.lof

    score_rows = []
    score_specs = {
        "ecoood": (calib_components_pred.ecoood_score, test_components.ecoood_score),
        "ad_similarity": (calib_ad.similarity, test_ad.similarity),
        "ad_leverage": (calib_ad.leverage, test_ad.leverage),
        "ad_range": (calib_ad.descriptor_range, test_ad.descriptor_range),
        "ad_distance_to_model": (calib_ad.distance_to_model, test_ad.distance_to_model),
        "uncertainty_interval_width": (calib_ad.interval_width, test_ad.interval_width),
        "ood_mahalanobis": (calib_ad.mahalanobis, test_ad.mahalanobis),
        "ood_isolation_forest": (calib_ad.isolation_forest, test_ad.isolation_forest),
        "ood_lof": (calib_ad.lof, test_ad.lof),
    }
    for method, (calib_score, test_score) in score_specs.items():
        row = {
            "split": config.split,
            "model": config.model_name,
            **score_method_metrics(
                method=method,
                y_true=y_test,
                y_pred=test_pred.mean,
                score=test_score,
                known_ood=known_ood,
                id_scores=calib_score,
                ood_scores=test_score if np.any(split_indices.test_is_ood) else np.array([]),
            ),
        }
        score_rows.append(row)
    score_summary = pd.DataFrame(score_rows)

    if config.output_dir:
        base = Path(config.output_dir) / config.split / config.model_name
        save_metrics(metrics, base / "metrics.json")
        save_predictions(predictions, base / "predictions.csv")
        save_predictions(score_summary, base / "ood_score_summary.csv")
    return metrics, predictions, score_summary


def run_benchmark(
    df: pd.DataFrame,
    splits: list[str],
    models: list[str],
    output_dir: str,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    alpha: float = 0.1,
    seed: int = 42,
    n_members: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    score_rows: list[pd.DataFrame] = []
    for split in splits:
        for model_name in models:
            config = ExperimentConfig(
                split=split,
                model_name=model_name,
                alpha=alpha,
                seed=seed,
                n_members=n_members,
                output_dir=output_dir,
            )
            metrics, _, score_summary = run_single_experiment(df, config=config, schema=schema)
            rows.append(metrics)
            score_rows.append(score_summary)
    summary = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    if score_rows:
        pd.concat(score_rows, ignore_index=True).to_csv(output / "ood_score_summary.csv", index=False)
    return summary
