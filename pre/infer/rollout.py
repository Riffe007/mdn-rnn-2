from __future__ import annotations

from collections.abc import Callable

import numpy as np


def monte_carlo_rollout(
    initial_state: np.ndarray,
    transition_fn: Callable[[np.ndarray], np.ndarray],
    steps: int,
    num_samples: int,
) -> np.ndarray:
    """Generate Monte Carlo trajectories with a caller-provided transition function."""
    trajectories = np.zeros((num_samples, steps), dtype=float)
    for i in range(num_samples):
        state = initial_state.copy()
        for t in range(steps):
            state = transition_fn(state)
            trajectories[i, t] = state.squeeze()
    return trajectories
