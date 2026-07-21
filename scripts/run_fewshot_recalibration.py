from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecoood.conformal import ScaledConformalRegressor
from ecoood.evaluation import interval_metrics, ood_metrics
from ecoood.features import EcoFeatureBuilder, FeatureBundle, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.ood import EcoOODScorer
from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, add_panel_label, apply_publication_style, finish_axis, save_figure
from ecoood.schema import DEFAULT_SCHEMA, EcoOODSchema
from ecoood.splits import SplitIndices, build_split


@dataclass
class SplitFit:
    train_df: pd.DataFrame
    calib_df: pd.DataFrame
    test_df: pd.DataFrame
    train_bundle: FeatureBundle
    calib_bundle: FeatureBundle
    test_bundle: FeatureBundle
    split_indices: SplitIndices
    y_test: np.ndarray
    y_pred_test: np.ndarray
    model_std_test: np.ndarray
    base_interval_lower: np.ndarray
    base_interval_upper: np.ndarray
    base_interval_width: np.ndarray
    base_ecoood_score: np.ndarray


def _select(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return df.loc[idx].reset_index(drop=True)


def _slice_bundle(bundle: FeatureBundle, idx: np.ndarray) -> FeatureBundle:
    return FeatureBundle(
        full=bundle.full[idx],
        fingerprint=bundle.fingerprint[idx],
        descriptor=bundle.descriptor[idx],
        species=bundle.species[idx],
        context=bundle.context[idx],
        mechanism=bundle.mechanism[idx],
        feature_names=bundle.feature_names,
    )


def _sample_adaptation_indices(n_total: int, n_adapt: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_adapt <= 0:
        return np.array([], dtype=int), np.arange(n_total, dtype=int)
    if n_adapt >= n_total:
        raise ValueError("n_adapt must be smaller than n_total.")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_total, size=n_adapt, replace=False))
    mask = np.ones(n_total, dtype=bool)
    mask[chosen] = False
    remaining = np.arange(n_total, dtype=int)[mask]
    return chosen, remaining


def _known_ood_labels(df_test: pd.DataFrame, split_indices: SplitIndices, schema: EcoOODSchema) -> np.ndarray:
    if split_indices.split_name == "hard_ood":
        if schema.known_ood in df_test.columns:
            return df_test[schema.known_ood].fillna(False).astype(bool).to_numpy()
        if schema.hard_ood in df_test.columns:
            return df_test[schema.hard_ood].fillna(False).astype(bool).to_numpy()
    return split_indices.test_is_ood


def _fit_split(
    df: pd.DataFrame,
    split: str,
    model_name: str,
    seed: int,
    alpha: float,
    members: int,
    schema: EcoOODSchema,
) -> SplitFit:
    working = attach_rdkit_descriptors(df, schema)
    working = working[working[schema.target].notna()].reset_index(drop=True)
    split_indices = build_split(working, split=split, schema=schema, seed=seed)
    train_df = _select(working, split_indices.train)
    calib_df = _select(working, split_indices.calib)
    test_df = _select(working, split_indices.test)

    feature_builder = EcoFeatureBuilder(schema=schema)
    train_bundle = feature_builder.fit_transform(train_df)
    calib_bundle = feature_builder.transform(calib_df)
    test_bundle = feature_builder.transform(test_df)

    model = BootstrapEnsembleRegressor(
        model_name=model_name,
        n_members=members,
        seed=seed,
    ).fit(train_bundle.full, train_df[schema.target].to_numpy())
    calib_pred = model.predict(calib_bundle.full)
    test_pred = model.predict(test_bundle.full)

    conformal = ScaledConformalRegressor(alpha=alpha).fit(
        calib_df[schema.target].to_numpy(),
        calib_pred.mean,
        scale=np.maximum(calib_pred.std, 1e-3),
    )
    base_interval = conformal.predict(test_pred.mean, scale=np.maximum(test_pred.std, 1e-3))

    scorer = EcoOODScorer(schema=schema).fit(train_df, train_bundle)
    calib_components = scorer.component_frame(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=conformal.predict(calib_pred.mean, scale=np.maximum(calib_pred.std, 1e-3)).width,
    )
    scorer.fit_meta(
        calib_components,
        residuals=np.abs(calib_df[schema.target].to_numpy() - calib_pred.mean),
    )
    test_components = scorer.predict(
        test_df,
        test_bundle,
        model_std=test_pred.std,
        interval_width=base_interval.width,
    )

    return SplitFit(
        train_df=train_df,
        calib_df=calib_df,
        test_df=test_df,
        train_bundle=train_bundle,
        calib_bundle=calib_bundle,
        test_bundle=test_bundle,
        split_indices=split_indices,
        y_test=test_df[schema.target].to_numpy(),
        y_pred_test=test_pred.mean,
        model_std_test=test_pred.std,
        base_interval_lower=base_interval.lower,
        base_interval_upper=base_interval.upper,
        base_interval_width=base_interval.width,
        base_ecoood_score=test_components.ecoood_score,
    )


def _evaluate_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    score: np.ndarray,
    known_ood: np.ndarray,
) -> dict[str, float]:
    return {
        **interval_metrics(
            y_true,
            lower,
            upper,
            uncertainty=model_std,
            novelty=score,
        ),
        **ood_metrics(
            y_true,
            y_pred,
            score,
            known_ood,
        ),
    }


def _recalibrated_score(
    fit: SplitFit,
    adapt_idx: np.ndarray,
    eval_idx: np.ndarray,
    adapt_interval_width: np.ndarray,
    eval_interval_width: np.ndarray,
    schema: EcoOODSchema,
    catastrophic_quantile: float,
) -> np.ndarray:
    if len(adapt_idx) == 0:
        return fit.base_ecoood_score[eval_idx]
    local_scorer = EcoOODScorer(schema=schema).fit(fit.train_df, fit.train_bundle)
    adapt_df = fit.test_df.iloc[adapt_idx].reset_index(drop=True)
    adapt_bundle = _slice_bundle(fit.test_bundle, adapt_idx)
    adapt_components = local_scorer.component_frame(
        adapt_df,
        adapt_bundle,
        model_std=fit.model_std_test[adapt_idx],
        interval_width=adapt_interval_width,
    )
    local_scorer.fit_meta(
        adapt_components,
        residuals=np.abs(fit.y_test[adapt_idx] - fit.y_pred_test[adapt_idx]),
        catastrophic_quantile=catastrophic_quantile,
    )
    eval_df = fit.test_df.iloc[eval_idx].reset_index(drop=True)
    eval_bundle = _slice_bundle(fit.test_bundle, eval_idx)
    eval_components = local_scorer.predict(
        eval_df,
        eval_bundle,
        model_std=fit.model_std_test[eval_idx],
        interval_width=eval_interval_width,
    )
    return eval_components.ecoood_score


def _aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in frame.columns
        if col not in {"split", "model", "seed", "adapt_seed", "shots", "n_eval"}
    ]
    aggregated = (
        frame.groupby(["split", "model", "shots"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregated.columns.to_flat_index()
    ]
    return aggregated


def _plot_curves(
    summary: pd.DataFrame,
    output_dir: Path,
    target_coverage: float,
    *,
    stem: str = "Figure_6",
) -> None:
    palette = {
        "temporal": PALETTE["blue"],
        "species": PALETTE["red"],
        "chemical_class": PALETTE["green"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(ACS_DOUBLE_WIDTH, 2.75), constrained_layout=True)
    for split, frame in summary.groupby("split"):
        color = palette.get(split, None)
        x = frame["shots"]
        axes[0].plot(x, frame["coverage_after_mean"], marker="o", color=color, label=split)
        axes[0].fill_between(
            x,
            frame["coverage_after_mean"] - frame["coverage_after_std"].fillna(0.0),
            frame["coverage_after_mean"] + frame["coverage_after_std"].fillna(0.0),
            alpha=0.18,
            color=color,
        )
        axes[1].plot(x, frame["aurc_after_mean"], marker="o", color=color, label=split)
        axes[1].fill_between(
            x,
            frame["aurc_after_mean"] - frame["aurc_after_std"].fillna(0.0),
            frame["aurc_after_mean"] + frame["aurc_after_std"].fillna(0.0),
            alpha=0.18,
            color=color,
        )
    axes[0].axhline(target_coverage, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("New-domain labeled samples")
    axes[0].set_ylabel("90% interval coverage")
    axes[1].set_xlabel("New-domain labeled samples")
    axes[1].set_ylabel("AURC (lower is better)")
    axes[0].set_title("Coverage recovery", pad=6)
    axes[1].set_title("Post-recalibration risk", pad=6)
    add_panel_label(axes[0], "A", x=-0.18, y=1.08)
    add_panel_label(axes[1], "B", x=-0.18, y=1.08)
    finish_axis(axes[0])
    finish_axis(axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [label.replace("_", " ").title() for label in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        title="",
        frameon=False,
    )
    save_figure(fig, output_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run few-shot OOD recalibration experiments.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["temporal", "species", "chemical_class"])
    parser.add_argument("--models", nargs="+", default=["lightgbm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--adapt-seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 20, 50, 100, 200])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--catastrophic-quantile", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fewshot_recalibration"))
    parser.add_argument("--stem", default="Figure_6")
    args = parser.parse_args()
    apply_publication_style()

    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    rows: list[dict[str, float | int | str]] = []
    schema = DEFAULT_SCHEMA
    for split in args.splits:
        for model_name in args.models:
            for seed in args.seeds:
                fit = _fit_split(
                    df=df,
                    split=split,
                    model_name=model_name,
                    seed=seed,
                    alpha=args.alpha,
                    members=args.members,
                    schema=schema,
                )
                known_ood = _known_ood_labels(fit.test_df, fit.split_indices, schema)
                n_test = len(fit.test_df)
                valid_shots = [shot for shot in args.shots if shot < n_test]
                for shot in valid_shots:
                    active_adapt_seeds = args.adapt_seeds if shot > 0 else [0]
                    for adapt_seed in active_adapt_seeds:
                        adapt_idx, eval_idx = _sample_adaptation_indices(n_test, shot, seed=adapt_seed) if shot > 0 else (np.array([], dtype=int), np.arange(n_test, dtype=int))
                        base_metrics = _evaluate_subset(
                            y_true=fit.y_test[eval_idx],
                            y_pred=fit.y_pred_test[eval_idx],
                            model_std=fit.model_std_test[eval_idx],
                            lower=fit.base_interval_lower[eval_idx],
                            upper=fit.base_interval_upper[eval_idx],
                            score=fit.base_ecoood_score[eval_idx],
                            known_ood=known_ood[eval_idx],
                        )

                        if shot == 0:
                            recal_lower = fit.base_interval_lower[eval_idx]
                            recal_upper = fit.base_interval_upper[eval_idx]
                            recal_width = fit.base_interval_width[eval_idx]
                            recal_score = fit.base_ecoood_score[eval_idx]
                        else:
                            conformal = ScaledConformalRegressor(alpha=args.alpha).fit(
                                fit.y_test[adapt_idx],
                                fit.y_pred_test[adapt_idx],
                                scale=np.maximum(fit.model_std_test[adapt_idx], 1e-3),
                            )
                            adapt_interval = conformal.predict(
                                fit.y_pred_test[adapt_idx],
                                scale=np.maximum(fit.model_std_test[adapt_idx], 1e-3),
                            )
                            eval_interval = conformal.predict(
                                fit.y_pred_test[eval_idx],
                                scale=np.maximum(fit.model_std_test[eval_idx], 1e-3),
                            )
                            recal_lower = eval_interval.lower
                            recal_upper = eval_interval.upper
                            recal_width = eval_interval.width
                            recal_score = _recalibrated_score(
                                fit=fit,
                                adapt_idx=adapt_idx,
                                eval_idx=eval_idx,
                                adapt_interval_width=adapt_interval.width,
                                eval_interval_width=recal_width,
                                schema=schema,
                                catastrophic_quantile=args.catastrophic_quantile,
                            )

                        recal_metrics = _evaluate_subset(
                            y_true=fit.y_test[eval_idx],
                            y_pred=fit.y_pred_test[eval_idx],
                            model_std=fit.model_std_test[eval_idx],
                            lower=recal_lower,
                            upper=recal_upper,
                            score=recal_score,
                            known_ood=known_ood[eval_idx],
                        )

                        rows.append(
                            {
                                "split": split,
                                "model": model_name,
                                "seed": seed,
                                "adapt_seed": adapt_seed,
                                "shots": shot,
                                "n_eval": int(len(eval_idx)),
                                "coverage_before": base_metrics["coverage"],
                                "coverage_after": recal_metrics["coverage"],
                                "coverage_recovery": recal_metrics["coverage"] - base_metrics["coverage"],
                                "mean_interval_width_before": base_metrics["mean_interval_width"],
                                "mean_interval_width_after": recal_metrics["mean_interval_width"],
                                "aurc_before": base_metrics["aurc"],
                                "aurc_after": recal_metrics["aurc"],
                                "aurc_improvement": base_metrics["aurc"] - recal_metrics["aurc"],
                                "catastrophic_capture_before": base_metrics["catastrophic_error_capture_rate"],
                                "catastrophic_capture_after": recal_metrics["catastrophic_error_capture_rate"],
                                "target_coverage_gap_before": (1 - args.alpha) - base_metrics["coverage"],
                                "target_coverage_gap_after": (1 - args.alpha) - recal_metrics["coverage"],
                            }
                        )

    all_results = pd.DataFrame(rows)
    summary = _aggregate_results(all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.output_dir / "fewshot_recalibration_all.csv", index=False)
    summary.to_csv(args.output_dir / "fewshot_recalibration_summary.csv", index=False)
    _plot_curves(summary, args.output_dir, target_coverage=1 - args.alpha, stem=args.stem)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
