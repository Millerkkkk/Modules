from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.stats import truncnorm


def generate_normal_scenarios(
    mu: np.ndarray,
    beta: float,
    n_scenarios: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate stochastic demand scenarios from a left-truncated normal distribution.

    xi_i^s ~ TN(mu_i, (beta * mu_i)^2; lower=0)
    """
    mu = np.asarray(mu, dtype=float)

    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    sigma = beta * mu
    scenarios = np.empty((n_scenarios, len(mu)), dtype=float)

    rng = np.random.default_rng(random_state)

    for i, (m, s) in enumerate(zip(mu, sigma)):
        if m < 0:
            raise ValueError(f"mu[{i}] must be nonnegative, got {m}.")

        if s == 0:
            scenarios[:, i] = m
        else:
            a = (0.0 - m) / s
            b = np.inf
            scenarios[:, i] = truncnorm.rvs(
                a=a,
                b=b,
                loc=m,
                scale=s,
                size=n_scenarios,
                random_state=rng,
            )

    return scenarios

