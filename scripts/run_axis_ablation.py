from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecoood.conformal import ScaledConformalRegressor
from ecoood.evaluation import interval_metrics, ood_metrics, regression_metrics
from ecoood.features import EcoFeatureBuilder, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.ood import EcoOODScorer
from ecoood.pipeline import _known_ood_labels
from ecoood.plotting import ACS_DOUBLE_WIDTH, add_panel_label, apply_publication_style, save_figure
from ecoood.schema import DEFAULT_SCHEMA, EcoOODSchema
from ecoood.splits import build_split


PROFILE_ORDER = [
    "chemical_only",
    "chemical_species",
    "chemical_species_context",
    "chemical_species_context_mechanism",
]
PROFILE_LABELS = {
    "chemical_only": "Chemical",
    "chemical_species": "Chemical +\nSpecies",
    "chemical_species_context": "Chemical + Species\n+ Context",
    "chemical_species_context_mechanism": "Chemical + Species\n+ Context + Bioactivity proxy",
}
SPLIT_ORDER = ["temporal", "species", "chemical_class"]
SPLIT_LABELS = {
    "temporal": "Temporal",
    "species": "Species",
    "chemical_class": "Class Holdout",
}


def _select(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return df.loc[idx].reset_index(drop=True)


def _profile_columns(df: pd.DataFrame, profile: str, schema: EcoOODSchema) -> list[str]:
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
    chemical_cols = {
        col for col in df.columns if col.startswith("physchem_")
    }
    species_cols = {
        schema.species,
        schema.genus,
        schema.family,
        schema.order,
        schema.clazz,
        schema.phylum,
        schema.trophic_group,
    }
    context_cols = {
        schema.duration_h,
        schema.medium,
        schema.temperature_c,
        schema.ph,
        schema.study_year,
        schema.source,
        *(col for col in df.columns if col.startswith("ctx_")),
    }
    mech_cols = {
        col for col in df.columns if col.startswith("mech_")
    }

    keep = set(always_keep) | chemical_cols
    if profile in {"chemical_species", "chemical_species_context", "chemical_species_context_mechanism"}:
        keep |= {col for col in species_cols if col in df.columns}
    if profile in {"chemical_species_context", "chemical_species_context_mechanism"}:
        keep |= {col for col in context_cols if col in df.columns}
    if profile == "chemical_species_context_mechanism":
        keep |= mech_cols
    return [col for col in df.columns if col in keep]


def _prepare_profile_frame(df: pd.DataFrame, profile: str, schema: EcoOODSchema) -> pd.DataFrame:
    cols = _profile_columns(df, profile=profile, schema=schema)
    return df[cols].copy()


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


def _plot_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(ACS_DOUBLE_WIDTH, 5.7), constrained_layout=True)
    specs = [
        ("rmse_mean", "RMSE", "YlOrRd"),
        ("coverage_mean", "Coverage", "YlGn"),
        ("aurc_mean", "AURC", "YlOrRd_r"),
    ]
    for idx, (ax, (metric, title, cmap)) in enumerate(zip(axes, specs)):
        pivot = (
            summary.pivot(index="profile", columns="split", values=metric)
            .reindex(index=PROFILE_ORDER, columns=SPLIT_ORDER)
            .rename(index=PROFILE_LABELS, columns=SPLIT_LABELS)
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            linewidths=0.8,
            linecolor="white",
            cbar=True,
            ax=ax,
            annot_kws={"fontsize": 7},
            cbar_kws={"shrink": 0.82, "pad": 0.02},
        )
        ax.set_title(title, pad=6)
        ax.set_xlabel("" if idx < len(specs) - 1 else "Benchmark split")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        add_panel_label(ax, chr(ord("A") + idx), x=-0.08, y=1.03)
    save_figure(fig, output_dir, "figureS_axis_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature-axis ablation for EcoOOD benchmark splits.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=SPLIT_ORDER)
    parser.add_argument("--profiles", nargs="+", default=PROFILE_ORDER)
    parser.add_argument("--models", nargs="+", default=["lightgbm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/axis_ablation"))
    args = parser.parse_args()
    apply_publication_style()

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

            for profile in args.profiles:
                train_df = _prepare_profile_frame(train_full, profile=profile, schema=schema)
                calib_df = _prepare_profile_frame(calib_full, profile=profile, schema=schema)
                test_df = _prepare_profile_frame(test_full, profile=profile, schema=schema)

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
                        scale=np.maximum(calib_pred.std, 1e-3),
                    )
                    test_interval = conformal.predict(
                        test_pred.mean,
                        scale=np.maximum(test_pred.std, 1e-3),
                    )

                    scorer = EcoOODScorer(schema=schema).fit(train_df, train_bundle)
                    calib_components = scorer.component_frame(
                        calib_df,
                        calib_bundle,
                        model_std=calib_pred.std,
                        interval_width=conformal.predict(
                            calib_pred.mean,
                            scale=np.maximum(calib_pred.std, 1e-3),
                        ).width,
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

                    y_test = test_df[schema.target].to_numpy()
                    rows.append(
                        {
                            "profile": profile,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.output_dir / "axis_ablation_all.csv", index=False)
    summary.to_csv(args.output_dir / "axis_ablation_summary.csv", index=False)
    _plot_heatmaps(summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
