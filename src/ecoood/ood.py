from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from .features import FeatureBundle
from .schema import DEFAULT_SCHEMA, EcoOODSchema


AXIS_COMPONENTS = {
    "chemical": ("d_chem_knn", "d_chem_mahal"),
    "biological": ("d_species_knn", "d_species_tax"),
    "contextual": ("d_context",),
    "bioactivity": ("d_mech",),
    "uncertainty": ("u_model",),
}


def _mean_knn_distance(
    train: np.ndarray,
    query: np.ndarray,
    metric: str = "euclidean",
    n_neighbors: int = 5,
) -> np.ndarray:
    if train.shape[1] == 0:
        return np.zeros(query.shape[0], dtype=float)
    k = max(1, min(n_neighbors, train.shape[0]))
    nn = NearestNeighbors(metric=metric, n_neighbors=k)
    nn.fit(train)
    distances, _ = nn.kneighbors(query)
    return distances.mean(axis=1)


def _mahalanobis_scores(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    if train.shape[1] == 0:
        return np.zeros(len(query), dtype=float)
    centered = train - train.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov += np.eye(cov.shape[0]) * 1e-6
    inv = np.linalg.pinv(cov)
    delta = query - train.mean(axis=0, keepdims=True)
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, inv, delta))


def _taxon_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().casefold()
    return text or None


def _taxonomy_novelty(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    schema: EcoOODSchema,
) -> np.ndarray:
    """Return normalized lineage distance to the deepest training-supported rank.

    A query receives zero when its full species lineage is represented in the
    training archive. An unseen species in a seen genus receives 1/6, whereas a
    lineage with no represented phylum receives 1. This preserves information
    about higher-rank support instead of assigning every unseen species the same
    maximum novelty.
    """
    levels = [
        schema.phylum,
        schema.clazz,
        schema.order,
        schema.family,
        schema.genus,
        schema.species,
    ]
    available_levels = [level for level in levels if level in train_df and level in query_df]
    if not available_levels:
        return np.zeros(len(query_df), dtype=float)

    seen_prefixes: dict[int, set[tuple[str, ...]]] = {
        depth: set() for depth in range(1, len(available_levels) + 1)
    }
    for _, row in train_df.iterrows():
        lineage: list[str] = []
        for depth, level in enumerate(available_levels, start=1):
            value = _taxon_value(row.get(level))
            if value is None:
                break
            lineage.append(value)
            seen_prefixes[depth].add(tuple(lineage))

    scores = np.zeros(len(query_df), dtype=float)
    for i, (_, row) in enumerate(query_df.iterrows()):
        lineage: list[str] = []
        deepest_supported = 0
        for depth, level in enumerate(available_levels, start=1):
            value = _taxon_value(row.get(level))
            if value is None:
                break
            lineage.append(value)
            if tuple(lineage) in seen_prefixes[depth]:
                deepest_supported = depth
            else:
                break
        scores[i] = 1.0 - deepest_supported / len(available_levels)
    return scores


def _high_error_labels(
    residuals: np.ndarray,
    high_error_quantile: float,
    *,
    groups: pd.Series | np.ndarray | None = None,
    groupwise: bool = False,
) -> tuple[np.ndarray, float, dict[str, float]]:
    residuals = np.asarray(residuals, dtype=float)
    if not 0 < high_error_quantile < 1:
        raise ValueError("high_error_quantile must be between 0 and 1.")
    pooled_threshold = float(np.quantile(residuals, high_error_quantile))
    if groups is None or not groupwise:
        return residuals >= pooled_threshold, pooled_threshold, {}

    group_values = pd.Series(groups, dtype="object").fillna("missing").astype(str).to_numpy()
    labels = np.zeros(len(residuals), dtype=bool)
    thresholds: dict[str, float] = {}
    for group in sorted(set(group_values)):
        mask = group_values == group
        threshold = float(np.quantile(residuals[mask], high_error_quantile))
        thresholds[group] = threshold
        labels[mask] = residuals[mask] >= threshold
    return labels, pooled_threshold, thresholds


def _group_balance_weights(groups: pd.Series | np.ndarray | None) -> np.ndarray | None:
    if groups is None:
        return None
    values = pd.Series(groups, dtype="object").fillna("missing").astype(str)
    counts = values.value_counts()
    n_groups = len(counts)
    if n_groups <= 1:
        return None
    return values.map(lambda value: len(values) / (n_groups * counts[value])).to_numpy(dtype=float)


def _make_logistic_model() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        class_weight=None,
        fit_intercept=True,
        max_iter=1000,
        random_state=0,
    )


@dataclass
class OODComponents:
    chemical: np.ndarray
    species: np.ndarray
    context: np.ndarray
    mechanism: np.ndarray
    model_uncertainty: np.ndarray
    ecoood_score: np.ndarray


class CalibrationRiskScorer:
    """Calibration-trained high-error risk scorer for matched supervision checks.

    The scorer uses only calibration-fold features and residual labels. It is
    intentionally generic so EcoOOD can be compared with simpler risk models
    trained with the same residual-supervision budget.
    """

    def __init__(self) -> None:
        self.scaler = MinMaxScaler(clip=False)
        self.model: LogisticRegression | None = None
        self.high_error_quantile_: float | None = None
        self.high_error_threshold_: float | None = None
        self.positive_count_: int = 0
        self.converged_: bool = True

    def fit(
        self,
        features: pd.DataFrame,
        residuals: np.ndarray,
        high_error_quantile: float = 0.9,
    ) -> "CalibrationRiskScorer":
        if features.empty:
            raise ValueError("Calibration risk features must contain at least one column.")
        residuals = np.asarray(residuals, dtype=float)
        labels, threshold, _ = _high_error_labels(residuals, high_error_quantile)
        self.high_error_quantile_ = high_error_quantile
        self.high_error_threshold_ = threshold
        self.positive_count_ = int(labels.sum())
        self.scaler.fit(features)
        scaled = self.scaler.transform(features)
        if len(np.unique(labels)) > 1:
            self.model = _make_logistic_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                self.model.fit(scaled, labels.astype(int))
            self.converged_ = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.transform(features)
        if self.model is not None:
            return self.model.predict_proba(scaled)[:, 1]
        return np.asarray(scaled.mean(axis=1), dtype=float)


class EcoOODScorer:
    def __init__(self, schema: EcoOODSchema = DEFAULT_SCHEMA) -> None:
        self.schema = schema
        self.train_df: pd.DataFrame | None = None
        self.train_bundle: FeatureBundle | None = None
        self.component_scaler = MinMaxScaler(clip=False)
        self.meta_model: LogisticRegression | None = None
        self.high_error_quantile_: float | None = None
        self.high_error_threshold_: float | None = None
        self.group_thresholds_: dict[str, float] = {}
        self.positive_count_: int = 0
        self.calibration_count_: int = 0
        self.converged_: bool = True

    def fit(self, train_df: pd.DataFrame, train_bundle: FeatureBundle) -> "EcoOODScorer":
        self.train_df = train_df.copy()
        self.train_bundle = train_bundle
        return self

    def component_frame(
        self,
        df: pd.DataFrame,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if self.train_df is None or self.train_bundle is None:
            raise RuntimeError("EcoOODScorer must be fit before use.")
        train_bundle = self.train_bundle
        chem_knn = _mean_knn_distance(train_bundle.fingerprint, bundle.fingerprint, metric="cosine")
        chem_mahal = _mahalanobis_scores(train_bundle.descriptor, bundle.descriptor)
        species_knn = _mean_knn_distance(train_bundle.species, bundle.species)
        species_tax = _taxonomy_novelty(self.train_df, df, self.schema)
        context = _mean_knn_distance(train_bundle.context, bundle.context)
        mechanism = _mean_knn_distance(train_bundle.mechanism, bundle.mechanism)
        # With scaled conformal prediction, interval width is a fold-specific
        # constant multiple of the ensemble standard deviation. Retaining both
        # would duplicate the same uncertainty signal in the meta-model.
        model_uncertainty = np.asarray(model_std, dtype=float)
        return pd.DataFrame(
            {
                "d_chem_knn": chem_knn,
                "d_chem_mahal": chem_mahal,
                "d_species_knn": species_knn,
                "d_species_tax": species_tax,
                "d_context": context,
                "d_mech": mechanism,
                "u_model": model_uncertainty,
            },
            index=df.index,
        )

    def fit_meta(
        self,
        components: pd.DataFrame,
        residuals: np.ndarray,
        high_error_quantile: float = 0.9,
        *,
        groups: pd.Series | np.ndarray | None = None,
        groupwise_labels: bool = False,
        balance_groups: bool = False,
    ) -> "EcoOODScorer":
        labels, threshold, group_thresholds = _high_error_labels(
            residuals,
            high_error_quantile,
            groups=groups,
            groupwise=groupwise_labels,
        )
        self.high_error_quantile_ = high_error_quantile
        self.high_error_threshold_ = threshold
        self.group_thresholds_ = group_thresholds
        self.positive_count_ = int(labels.sum())
        self.calibration_count_ = int(len(labels))
        self.component_scaler.fit(components)
        scaled = self.component_scaler.transform(components)
        if len(np.unique(labels)) > 1:
            sample_weight = _group_balance_weights(groups) if balance_groups else None
            self.meta_model = _make_logistic_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                self.meta_model.fit(
                    scaled,
                    labels.astype(int),
                    sample_weight=sample_weight,
                )
            self.converged_ = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
        return self

    def scaled_axis_frame(self, components: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.component_scaler, "n_features_in_"):
            raise RuntimeError("EcoOODScorer meta-model must be fit before axis scaling.")
        scaled = pd.DataFrame(
            self.component_scaler.transform(components),
            columns=components.columns,
            index=components.index,
        )
        return pd.DataFrame(
            {
                axis: scaled.loc[:, list(columns)].mean(axis=1)
                for axis, columns in AXIS_COMPONENTS.items()
            },
            index=components.index,
        )

    def diagnostics(self) -> dict[str, float | int | bool]:
        result: dict[str, float | int | bool] = {
            "calibration_n": self.calibration_count_,
            "high_error_n": self.positive_count_,
            "high_error_quantile": (
                float(self.high_error_quantile_)
                if self.high_error_quantile_ is not None
                else float("nan")
            ),
            "high_error_threshold": (
                float(self.high_error_threshold_)
                if self.high_error_threshold_ is not None
                else float("nan")
            ),
            "meta_converged": self.converged_,
            "meta_minmax_clip": False,
        }
        if self.meta_model is not None:
            for column, coefficient in zip(
                self.component_scaler.feature_names_in_,
                self.meta_model.coef_[0],
                strict=True,
            ):
                result[f"coef_{column}"] = float(coefficient)
            result["intercept"] = float(self.meta_model.intercept_[0])
        return result

    def score_components(self, components: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.component_scaler, "n_features_in_"):
            raise RuntimeError("EcoOODScorer meta-model must be fit before scoring.")
        scaled = self.component_scaler.transform(components)
        if self.meta_model is not None:
            return self.meta_model.predict_proba(scaled)[:, 1]
        return np.asarray(scaled.mean(axis=1), dtype=float)

    def predict(
        self,
        df: pd.DataFrame,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray | None = None,
    ) -> OODComponents:
        components = self.component_frame(df, bundle, model_std=model_std, interval_width=interval_width)
        ecoood_score = self.score_components(components)
        axes = self.scaled_axis_frame(components)
        return OODComponents(
            chemical=axes["chemical"].to_numpy(),
            species=axes["biological"].to_numpy(),
            context=axes["contextual"].to_numpy(),
            mechanism=axes["bioactivity"].to_numpy(),
            model_uncertainty=axes["uncertainty"].to_numpy(),
            ecoood_score=np.asarray(ecoood_score, dtype=float),
        )
