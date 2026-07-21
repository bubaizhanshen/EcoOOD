from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ecoood.plotting import ACS_DOUBLE_WIDTH, PALETTE, add_panel_label, apply_publication_style, finish_axis, save_figure
from ecoood.conformal import ScaledConformalRegressor
from ecoood.evaluation import interval_metrics, ood_metrics, regression_metrics
from ecoood.features import EcoFeatureBuilder, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.ood import EcoOODScorer
from ecoood.pipeline import _known_ood_labels
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


SPLIT_ORDER = ["random", "scaffold", "chemical_class"]
SPLIT_LABELS = {
    "random": "Random",
    "scaffold": "Scaffold",
    "chemical_class": "Class Holdout",
}
MODEL_ORDER = ["lightgbm", "xgboost", "mlp", "random_forest"]
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "mlp": "Fingerprint MLP",
    "random_forest": "Random Forest",
}


def _select(df: pd.DataFrame, idx) -> pd.DataFrame:
    return df.loc[idx].reset_index(drop=True)


def _aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in frame.columns if col not in {"profile", "split", "model", "seed"}]
    aggregated = (
        frame.groupby(["profile", "split", "model"], dropna=False)[metric_cols]
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


def _profile_columns(df: pd.DataFrame, schema) -> list[str]:
    always_keep = {
        schema.target,
        schema.smiles,
        schema.endpoint,
        schema.chemical_id,
        schema.chemical_name,
        schema.casrn,
        schema.hard_ood,
        schema.known_ood,
    }
    chemical_cols = {col for col in df.columns if col.startswith("physchem_")}
    keep = set(always_keep) | chemical_cols
    return [col for col in df.columns if col in keep]


def _prepare_profile_frame(df: pd.DataFrame, schema) -> pd.DataFrame:
    return df[_profile_columns(df, schema=schema)].copy()


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    frame = summary.copy()
    frame = frame[frame["split"].isin(SPLIT_ORDER) & frame["model"].isin(MODEL_ORDER)].copy()
    frame["split_label"] = frame["split"].map(SPLIT_LABELS)
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    palette = {
        "LightGBM": PALETTE["blue"],
        "XGBoost": PALETTE["green"],
        "Fingerprint MLP": PALETTE["orange"],
        "Random Forest": PALETTE["red"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(ACS_DOUBLE_WIDTH, 2.8), constrained_layout=True)
    present_model_set = set(frame["model"])
    present_models = [model for model in MODEL_ORDER if model in present_model_set]
    present_labels = [MODEL_LABELS[m] for m in present_models]
    panel_specs = [
        ("rmse_mean", "RMSE on log toxicity", "Chemical-only deployment error"),
        ("coverage_mean", "90% interval coverage", "Chemical-only coverage"),
        ("aurc_mean", "AURC", "Chemical-only risk ordering"),
    ]
    legend_handles = None
    legend_labels = None
    for idx, (ax, (metric, ylabel, title)) in enumerate(zip(axes, panel_specs)):
        sns.barplot(
            data=frame,
            x="split_label",
            y=metric,
            hue="model_label",
            order=[SPLIT_LABELS[s] for s in SPLIT_ORDER],
            hue_order=present_labels,
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        ax.set_title(title, pad=6)
        if metric == "coverage_mean":
            ax.axhline(0.9, color=PALETTE["ink"], linestyle="--", linewidth=0.9)
        if ax.legend_ is not None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
        add_panel_label(ax, chr(ord("A") + idx), x=-0.18, y=1.1)
        finish_axis(ax)
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=max(1, len(legend_labels)),
            bbox_to_anchor=(0.5, 1.1),
            frameon=False,
        )
    save_figure(fig, output_dir, "Figure_S2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chemical-only structure baseline comparisons.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--splits", nargs="+", default=SPLIT_ORDER)
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/structure_baselines"))
    args = parser.parse_args()

    apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_csv is not None:
        summary = pd.read_csv(args.summary_csv)
        summary.to_csv(args.output_dir / "structure_baselines_summary.csv", index=False)
        _plot(summary, args.output_dir)
        print(summary.to_string(index=False))
        return

    if args.data is None:
        raise SystemExit("Either --data or --summary-csv must be provided.")

    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    schema = DEFAULT_SCHEMA
    working = attach_rdkit_descriptors(df, schema)
    working = working[working[schema.target].notna()].reset_index(drop=True)

    rows: list[dict[str, float | int | str]] = []
    for split in args.splits:
        for seed in args.seeds:
            split_indices = build_split(working, split=split, schema=schema, seed=seed)
            train_full = _select(working, split_indices.train)
            calib_full = _select(working, split_indices.calib)
            test_full = _select(working, split_indices.test)
            known_ood = _known_ood_labels(test_full, split_indices, schema)

            train_df = _prepare_profile_frame(train_full, schema=schema)
            calib_df = _prepare_profile_frame(calib_full, schema=schema)
            test_df = _prepare_profile_frame(test_full, schema=schema)

            feature_builder = EcoFeatureBuilder(schema=schema)
            train_bundle = feature_builder.fit_transform(train_df)
            calib_bundle = feature_builder.transform(calib_df)
            test_bundle = feature_builder.transform(test_df)

            for model_name in args.models:
                model = BootstrapEnsembleRegressor(
                    model_name=model_name,
                    n_members=args.members,
                    seed=seed,
                ).fit(train_bundle.full, train_df[schema.target].to_numpy())
                calib_pred = model.predict(calib_bundle.full)
                test_pred = model.predict(test_bundle.full)

                conformal = ScaledConformalRegressor(alpha=args.alpha).fit(
                    calib_df[schema.target].to_numpy(),
                    calib_pred.mean,
                    scale=(calib_pred.std.clip(min=1e-3)),
                )
                calib_interval = conformal.predict(calib_pred.mean, scale=calib_pred.std.clip(min=1e-3))
                test_interval = conformal.predict(test_pred.mean, scale=test_pred.std.clip(min=1e-3))

                scorer = EcoOODScorer(schema=schema).fit(train_df, train_bundle)
                calib_components = scorer.component_frame(
                    calib_df,
                    calib_bundle,
                    model_std=calib_pred.std,
                    interval_width=calib_interval.width,
                )
                scorer.fit_meta(
                    calib_components,
                    residuals=abs(calib_df[schema.target].to_numpy() - calib_pred.mean),
                )
                test_components = scorer.predict(
                    test_df,
                    test_bundle,
                    model_std=test_pred.std,
                    interval_width=test_interval.width,
                )

                y_test = test_df[schema.target].to_numpy()
                rows.append(
                    {
                        "profile": "chemical_only",
                        "split": split,
                        "model": model_name,
                        "seed": seed,
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
                    }
                )

    all_results = pd.DataFrame(rows)
    summary = _aggregate_results(all_results)
    all_results.to_csv(args.output_dir / "structure_baselines_all.csv", index=False)
    summary.to_csv(args.output_dir / "structure_baselines_summary.csv", index=False)
    _plot(summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
