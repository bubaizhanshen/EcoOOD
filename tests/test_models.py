from __future__ import annotations

import numpy as np
from scipy import sparse

from ecoood.models import BootstrapEnsembleRegressor


def test_mlp_ensemble_handles_sparse_inputs() -> None:
    rng = np.random.default_rng(123)
    X_dense = rng.normal(size=(40, 16)).astype(float)
    X = sparse.csr_matrix(X_dense)
    y = X_dense[:, 0] * 0.5 - X_dense[:, 1] * 0.25 + rng.normal(scale=0.05, size=40)

    model = BootstrapEnsembleRegressor(model_name="mlp", n_members=2, seed=123, n_jobs=1).fit(X, y)
    pred = model.predict(X)

    assert pred.mean.shape == (40,)
    assert pred.std.shape == (40,)
    assert np.isfinite(pred.mean).all()
