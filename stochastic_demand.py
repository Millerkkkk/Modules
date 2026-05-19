from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
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





def pairwise_euclidean_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix for scenarios.
    X: shape (n_scenarios, n_features)
    Returns: shape (n_scenarios, n_scenarios)
    """
    sq_norms = np.sum(X**2, axis=1, keepdims=True)
    dist2 = sq_norms + sq_norms.T - 2 * X @ X.T
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2)


def fast_forward_selection(
    scenarios: np.ndarray,
    n_select: int,
    probs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast Forward Selection (FFS) for scenario reduction.

    Parameters
    ----------
    scenarios : np.ndarray
        Shape (S, d), where S is number of scenarios.
    n_select : int
        Number of representative scenarios to keep.
    probs : Optional[np.ndarray]
        Shape (S,), scenario probabilities. If None, uniform probabilities are used.

    Returns
    -------
    selected_idx : np.ndarray
        Indices of selected scenarios in the original scenario set.
    reduced_scenarios : np.ndarray
        Shape (n_select, d), selected representative scenarios.
    reduced_probs : np.ndarray
        Shape (n_select,), redistributed probabilities.
    """
    scenarios = np.asarray(scenarios, dtype=float)

    if scenarios.ndim != 2:
        raise ValueError("scenarios must be a 2D array of shape (n_scenarios, n_features).")

    S = scenarios.shape[0]

    if not (1 <= n_select <= S):
        raise ValueError("n_select must satisfy 1 <= n_select <= n_scenarios.")

    if probs is None:
        probs = np.full(S, 1.0 / S)
    else:
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (S,):
            raise ValueError("probs must have shape (n_scenarios,).")
        if np.any(probs < 0):
            raise ValueError("probs must be nonnegative.")
        total = probs.sum()
        if total <= 0:
            raise ValueError("probs must sum to a positive number.")
        probs = probs / total

    # Pairwise distances c(i, j)
    D = pairwise_euclidean_distance(scenarios)

    selected = []
    remaining = list(range(S))

    # min_dist[j] = distance from scenario j to the selected set
    min_dist = np.full(S, np.inf)

    for _ in range(n_select):
        best_idx = None
        best_obj = np.inf

        for cand in remaining:
            # If candidate is added, new distance to selected set for each scenario
            new_min_dist = np.minimum(min_dist, D[:, cand])
            obj = np.sum(probs * new_min_dist)

            if obj < best_obj:
                best_obj = obj
                best_idx = cand

        selected.append(best_idx)
        remaining.remove(best_idx)
        min_dist = np.minimum(min_dist, D[:, best_idx])

    selected_idx = np.array(selected, dtype=int)
    reduced_scenarios = scenarios[selected_idx]

    # Probability redistribution:
    # each original scenario is assigned to the nearest selected scenario
    dist_to_selected = D[:, selected_idx]  # shape (S, n_select)
    nearest_selected_pos = np.argmin(dist_to_selected, axis=1)

    reduced_probs = np.zeros(n_select, dtype=float)
    for original_idx, sel_pos in enumerate(nearest_selected_pos):
        reduced_probs[sel_pos] += probs[original_idx]

    return selected_idx, reduced_scenarios, reduced_probs


