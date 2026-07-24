from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ecoood.ad import ApplicabilityDomainScorer
from ecoood.features import FeatureBundle
from ecoood.conformal import decision_labels
from ecoood.ood import CalibrationRiskScorer, EcoOODScorer, _taxonomy_novelty
from ecoood.schema import DEFAULT_SCHEMA


def _bundle(n_rows: int) -> FeatureBundle:
    fingerprint = np.zeros((n_rows, 4), dtype=np.float32)
    fingerprint[np.arange(n_rows), np.arange(n_rows) % 4] = 1.0
    return FeatureBundle(
        full=sparse.csr_matrix(fingerprint),
        fingerprint=sparse.csr_matrix(fingerprint),
        descriptor=np.arange(n_rows, dtype=np.float32).reshape(-1, 1),
        species=np.zeros((n_rows, 1), dtype=np.float32),
        context=np.zeros((n_rows, 1), dtype=np.float32),
        mechanism=np.zeros((n_rows, 1), dtype=np.float32),
        feature_names=[],
    )


def _frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phylum": ["Chordata"] * n_rows,
            "class_name": ["Actinopterygii"] * n_rows,
            "order": ["Cypriniformes"] * n_rows,
            "family": ["Cyprinidae"] * n_rows,
            "genus": ["Danio"] * n_rows,
            "species": ["Danio rerio"] * n_rows,
        }
    )


def test_uncertainty_component_uses_ensemble_standard_deviation_only() -> None:
    train = _frame(3)
    query = _frame(2)
    scorer = EcoOODScorer(schema=DEFAULT_SCHEMA).fit(train, _bundle(3))
    components = scorer.component_frame(
        query,
        _bundle(2),
        model_std=np.array([0.12, 0.34]),
        interval_width=np.array([1.2, 9.9]),
    )

    assert np.allclose(components["u_model"], [0.12, 0.34])


def test_calibration_risk_scorer_uses_calibration_residual_labels() -> None:
    features = pd.DataFrame({"score": [0.0, 0.1, 0.2, 0.9, 1.0]})
    residuals = np.array([0.01, 0.02, 0.03, 0.8, 1.0])
    scorer = CalibrationRiskScorer().fit(features, residuals, high_error_quantile=0.8)
    scores = scorer.predict(features)

    assert scores.shape == (5,)
    assert scores[-1] > scores[0]


def test_score_only_routing_does_not_repeat_interval_width() -> None:
    labels = decision_labels(
        np.array([0.1, 0.6, 0.9]),
        None,
        score_warn_threshold=0.5,
        score_abstain_threshold=0.85,
    )

    assert labels.tolist() == ["predict", "warn", "abstain"]


def test_taxonomy_novelty_preserves_higher_rank_support() -> None:
    train = _frame(2)
    query = _frame(3)
    query.loc[0, ["species"]] = ["Danio aesculapii"]
    query.loc[1, ["genus", "species"]] = ["Pimephales", "Pimephales promelas"]
    query.loc[
        2,
        ["phylum", "class_name", "order", "family", "genus", "species"],
    ] = [
        "Arthropoda",
        "Branchiopoda",
        "Diplostraca",
        "Daphniidae",
        "Daphnia",
        "Daphnia magna",
    ]

    novelty = _taxonomy_novelty(train, query, DEFAULT_SCHEMA)

    assert np.allclose(novelty, [1 / 6, 2 / 6, 1.0])


def test_displayed_axes_use_calibration_scaled_subcomponents() -> None:
    scorer = EcoOODScorer(schema=DEFAULT_SCHEMA)
    components = pd.DataFrame(
        {
            "d_chem_knn": [0.0, 2.0],
            "d_chem_mahal": [0.0, 4.0],
            "d_species_knn": [0.0, 6.0],
            "d_species_tax": [0.0, 1.0],
            "d_context": [0.0, 8.0],
            "d_mech": [0.0, 10.0],
            "u_model": [0.0, 12.0],
        }
    )
    scorer.fit_meta(components, residuals=np.array([0.0, 1.0]), high_error_quantile=0.5)

    axes = scorer.scaled_axis_frame(components)

    assert np.allclose(axes.iloc[0], 0.0)
    assert np.allclose(axes.iloc[1], 1.0)


def test_equal_block_distance_is_available() -> None:
    train = _bundle(4)
    query = _bundle(2)
    scorer = ApplicabilityDomainScorer().fit(train)

    scores = scorer.predict(query, model_std=np.zeros(2))

    assert scores.equal_block_distance.shape == (2,)
    assert np.isfinite(scores.equal_block_distance).all()
