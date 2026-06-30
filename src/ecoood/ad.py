from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .features import FeatureBundle
from .ood import _mahalanobis_scores, _mean_knn_distance


@dataclass
class ADBaselineScores:
    similarity: np.ndarray
    leverage: np.ndarray
    descriptor_range: np.ndarray
    distance_to_model: np.ndarray
    interval_width: np.ndarray
    mahalanobis: np.ndarray
    isolation_forest: np.ndarray
    lof: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "ad_similarity": self.similarity,
            "ad_leverage": self.leverage,
            "ad_range": self.descriptor_range,
            "ad_distance_to_model": self.distance_to_model,
            "uncertainty_interval_width": self.interval_width,
            "ood_mahalanobis": self.mahalanobis,
            "ood_isolation_forest": self.isolation_forest,
            "ood_lof": self.lof,
        }


class ApplicabilityDomainScorer:
    def __init__(self) -> None:
        self.train_bundle: FeatureBundle | None = None
        self.descriptor_min: np.ndarray | None = None
        self.descriptor_max: np.ndarray | None = None
        self.xtx_inv: np.ndarray | None = None
        self.dense_scaler: StandardScaler | None = None
        self.dense_train: np.ndarray | None = None
        self.isolation_forest: IsolationForest | None = None
        self.lof: LocalOutlierFactor | None = None

    @staticmethod
    def _stack_dense(bundle: FeatureBundle) -> np.ndarray:
        blocks = [
            np.asarray(bundle.descriptor, dtype=float),
            np.asarray(bundle.species, dtype=float),
            np.asarray(bundle.context, dtype=float),
            np.asarray(bundle.mechanism, dtype=float),
        ]
        dense = np.hstack(blocks)
        if dense.ndim == 1:
            dense = dense.reshape(-1, 1)
        return np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, train_bundle: FeatureBundle) -> "ApplicabilityDomainScorer":
        self.train_bundle = train_bundle
        descriptor = np.asarray(train_bundle.descriptor, dtype=float)
        if descriptor.ndim == 1:
            descriptor = descriptor.reshape(-1, 1)
        if descriptor.shape[1] == 0:
            self.descriptor_min = np.zeros(1, dtype=float)
            self.descriptor_max = np.zeros(1, dtype=float)
            self.xtx_inv = np.zeros((1, 1), dtype=float)
            return self

        self.descriptor_min = descriptor.min(axis=0)
        self.descriptor_max = descriptor.max(axis=0)
        design = np.hstack([np.ones((descriptor.shape[0], 1), dtype=float), descriptor])
        xtx = design.T @ design
        xtx += np.eye(xtx.shape[0]) * 1e-8
        self.xtx_inv = np.linalg.pinv(xtx)

        dense = self._stack_dense(train_bundle)
        self.dense_train = dense
        self.dense_scaler = StandardScaler()
        dense_scaled = self.dense_scaler.fit_transform(dense)
        self.isolation_forest = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
        ).fit(dense_scaled)
        n_neighbors = max(2, min(35, len(dense_scaled) - 1))
        if len(dense_scaled) > 2:
            self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
            self.lof.fit(dense_scaled)
        return self

    def _descriptor_range_score(self, descriptor: np.ndarray) -> np.ndarray:
        if self.descriptor_min is None or self.descriptor_max is None:
            return np.zeros(len(descriptor), dtype=float)
        if descriptor.shape[1] == 0:
            return np.zeros(len(descriptor), dtype=float)
        out_of_range = (descriptor < self.descriptor_min) | (descriptor > self.descriptor_max)
        return out_of_range.mean(axis=1).astype(float)

    def _leverage_score(self, descriptor: np.ndarray) -> np.ndarray:
        if self.xtx_inv is None or descriptor.shape[1] == 0:
            return np.zeros(len(descriptor), dtype=float)
        design = np.hstack([np.ones((descriptor.shape[0], 1), dtype=float), descriptor])
        return np.einsum("ij,jk,ik->i", design, self.xtx_inv, design)

    def predict(
        self,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray | None = None,
    ) -> ADBaselineScores:
        if self.train_bundle is None:
            raise RuntimeError("ApplicabilityDomainScorer must be fit before use.")
        descriptor = np.asarray(bundle.descriptor, dtype=float)
        if descriptor.ndim == 1:
            descriptor = descriptor.reshape(-1, 1)
        dense = self._stack_dense(bundle)
        similarity = _mean_knn_distance(self.train_bundle.fingerprint, bundle.fingerprint, metric="cosine")
        leverage = self._leverage_score(descriptor)
        descriptor_range = self._descriptor_range_score(descriptor)
        distance_to_model = np.asarray(model_std, dtype=float)
        interval_width_arr = (
            np.asarray(interval_width, dtype=float)
            if interval_width is not None
            else np.zeros(len(distance_to_model), dtype=float)
        )
        mahalanobis = (
            _mahalanobis_scores(self.dense_train, dense)
            if self.dense_train is not None and dense.shape[1] > 0
            else np.zeros(len(distance_to_model), dtype=float)
        )
        if self.dense_scaler is not None:
            dense_scaled = self.dense_scaler.transform(dense)
        else:
            dense_scaled = dense
        isolation_forest = (
            -self.isolation_forest.score_samples(dense_scaled)
            if self.isolation_forest is not None
            else np.zeros(len(distance_to_model), dtype=float)
        )
        lof = (
            -self.lof.score_samples(dense_scaled)
            if self.lof is not None
            else np.zeros(len(distance_to_model), dtype=float)
        )
        return ADBaselineScores(
            similarity=np.asarray(similarity, dtype=float),
            leverage=np.asarray(leverage, dtype=float),
            descriptor_range=np.asarray(descriptor_range, dtype=float),
            distance_to_model=distance_to_model,
            interval_width=interval_width_arr,
            mahalanobis=np.asarray(mahalanobis, dtype=float),
            isolation_forest=np.asarray(isolation_forest, dtype=float),
            lof=np.asarray(lof, dtype=float),
        )
