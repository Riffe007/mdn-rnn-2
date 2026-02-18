from __future__ import annotations

import numpy as np

from pre.infer.predict import PredictiveDistribution


def test_predictive_distribution_from_samples_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    samples = rng.normal(loc=0.0, scale=1.0, size=(500, 12))

    dist = PredictiveDistribution.from_samples(horizon=np.arange(12), samples=samples)

    assert dist.mean is not None
    assert dist.std is not None
    assert dist.quantile(0.5).shape == (12,)
    low, high = dist.interval(0.1, 0.9)
    assert np.all(low <= high)
