from __future__ import annotations
from typing import Optional
import numpy as np


def generate_normal_scenarios(
    mu: np.ndarray,
    beta: float,
    n_scenarios: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate stochastic demand scenarios without truncation.

    xi_i^s ~ Normal(mu_i, (beta * mu_i)^2)

    Parameters
    ----------
    mu : array-like, shape (n_customers,)
        Mean demand for each customer (deterministic demand).
    beta : float
        Standard deviation ratio, sigma_i = beta * mu_i.
    n_scenarios : int
        Number of scenarios to generate.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    scenarios : ndarray, shape (n_scenarios, n_customers)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = beta * mu

    rng = np.random.default_rng(random_state)

    # 若 beta=0 或 mu_i=0，对应 sigma_i=0，会自然退化为常数 mu_i
    scenarios = rng.normal(
        loc=mu,
        scale=sigma,
        size=(n_scenarios, len(mu))
    )

    # ✅ 消除负值（关键修复）
    scenarios = np.maximum(scenarios, 0.0)

    return scenarios

