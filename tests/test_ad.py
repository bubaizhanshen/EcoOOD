from __future__ import annotations

import numpy as np

from ecoood.ad import ApplicabilityDomainScorer
from ecoood.features import FeatureBundle


def _bundle(descriptor: np.ndarray, fingerprint: np.ndarray) -> FeatureBundle:
    arr = np.asarray(descriptor, dtype=np.float32)
    fp = np.asarray(fingerprint, dtype=np.float32)
    return FeatureBundle(
        full=fp,
        fingerprint=fp,
        descriptor=arr,
        species=np.zeros((len(arr), 1), dtype=np.float32),
        context=np.zeros((len(arr), 1), dtype=np.float32),
        mechanism=np.zeros((len(arr), 1), dtype=np.float32),
        feature_names=[],
    )


def test_ad_scorer_increases_for_outside_descriptor_range() -> None:
    train = _bundle(
        descriptor=np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        fingerprint=np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
    )
    query = _bundle(
        descriptor=np.array([[0.25, 0.25], [2.0, 2.0]]),
        fingerprint=np.array([[1, 0, 0], [0, 0, 1]]),
    )
    scorer = ApplicabilityDomainScorer().fit(train)
    scores = scorer.predict(query, model_std=np.array([0.1, 0.5]), interval_width=np.array([0.2, 0.8]))
    assert scores.descriptor_range[0] == 0.0
    assert scores.descriptor_range[1] == 1.0
    assert scores.leverage[1] > scores.leverage[0]
    assert scores.distance_to_model[1] > scores.distance_to_model[0]
    assert scores.interval_width[1] > scores.interval_width[0]
    assert scores.mahalanobis[1] > scores.mahalanobis[0]
    assert scores.isolation_forest.shape == (2,)
    assert scores.lof.shape == (2,)
