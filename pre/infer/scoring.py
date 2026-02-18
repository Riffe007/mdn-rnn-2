from __future__ import annotations

import numpy as np


def tail_exceedance_probability(samples: np.ndarray, threshold: float) -> np.ndarray:
    """Estimate per-step exceedance risk from Monte Carlo samples."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape [num_samples, horizon]")
    return (samples > threshold).mean(axis=0)
