from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from .features import FeatureBundle
from .schema import DEFAULT_SCHEMA, EcoOODSchema


def _mean_knn_distance(train: np.ndarray, query: np.ndarray, metric: str = "euclidean", n_neighbors: int = 5) -> np.ndarray:
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


def _taxonomy_novelty(train_df: pd.DataFrame, query_df: pd.DataFrame, schema: EcoOODSchema) -> np.ndarray:
    levels = [
        schema.phylum,
        schema.clazz,
        schema.order,
        schema.family,
        schema.genus,
        schema.species,
    ]
    seen = {level: set(train_df[level].dropna().astype(str)) for level in levels if level in train_df}
    weights = {
        schema.phylum: 0.1,
        schema.clazz: 0.2,
        schema.order: 0.35,
        schema.family: 0.5,
        schema.genus: 0.75,
        schema.species: 1.0,
    }
    scores = np.zeros(len(query_df), dtype=float)
    for i, (_, row) in enumerate(query_df.iterrows()):
        novelty = 0.0
        for level in levels:
            if level not in row or level not in seen:
                continue
            value = str(row[level])
            if value and value not in seen[level]:
                novelty = max(novelty, weights[level])
        scores[i] = novelty
    return scores


@dataclass
class OODComponents:
    chemical: np.ndarray
    species: np.ndarray
    context: np.ndarray
    mechanism: np.ndarray
    model_uncertainty: np.ndarray
    ecoood_score: np.ndarray


class EcoOODScorer:
    def __init__(self, schema: EcoOODSchema = DEFAULT_SCHEMA) -> None:
        self.schema = schema
        self.train_df: pd.DataFrame | None = None
        self.train_bundle: FeatureBundle | None = None
        self.component_scaler = MinMaxScaler()
        self.meta_model: LogisticRegression | None = None

    def fit(self, train_df: pd.DataFrame, train_bundle: FeatureBundle) -> "EcoOODScorer":
        self.train_df = train_df.copy()
        self.train_bundle = train_bundle
        return self

    def component_frame(
        self,
        df: pd.DataFrame,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray,
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
        model_uncertainty = np.asarray(model_std, dtype=float) + 0.5 * np.asarray(interval_width, dtype=float)
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

    def fit_meta(self, components: pd.DataFrame, residuals: np.ndarray, catastrophic_quantile: float = 0.9) -> "EcoOODScorer":
        labels = np.asarray(residuals, dtype=float) >= np.quantile(residuals, catastrophic_quantile)
        self.component_scaler.fit(components)
        scaled = self.component_scaler.transform(components)
        if len(np.unique(labels)) > 1:
            self.meta_model = LogisticRegression(max_iter=1000)
            self.meta_model.fit(scaled, labels.astype(int))
        return self

    def predict(
        self,
        df: pd.DataFrame,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray,
    ) -> OODComponents:
        components = self.component_frame(df, bundle, model_std=model_std, interval_width=interval_width)
        scaled = self.component_scaler.transform(components) if hasattr(self.component_scaler, "n_features_in_") else components.to_numpy()
        if self.meta_model is not None:
            ecoood_score = self.meta_model.predict_proba(scaled)[:, 1]
        else:
            ecoood_score = scaled.mean(axis=1)
        return OODComponents(
            chemical=components[["d_chem_knn", "d_chem_mahal"]].mean(axis=1).to_numpy(),
            species=components[["d_species_knn", "d_species_tax"]].mean(axis=1).to_numpy(),
            context=components["d_context"].to_numpy(),
            mechanism=components["d_mech"].to_numpy(),
            model_uncertainty=components["u_model"].to_numpy(),
            ecoood_score=np.asarray(ecoood_score, dtype=float),
        )
