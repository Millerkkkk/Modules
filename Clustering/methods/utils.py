# ---------------- feature functions ----------------

from typing import Callable

import numpy as np


FeatureFn = Callable[[int], np.ndarray]


def feat_location(problem) -> FeatureFn:
    """(x, y)"""
    x = np.asarray(problem.x, dtype=float)
    y = np.asarray(problem.y, dtype=float)

    def f(i: int) -> np.ndarray:
        return np.array([x[i], y[i]], dtype=float)

    return f

def feat_time_window(problem) -> FeatureFn:
    """(ready, due)"""
    rt = np.asarray(problem.ready_time, dtype=float)
    dd = np.asarray(problem.due_time, dtype=float)

    def f(i: int) -> np.ndarray:
        return np.array([rt[i], dd[i]], dtype=float)

    return f

def feat_time_window_mid_width(problem, *, scale_mid: float = 1.0, scale_width: float = 1.0) -> FeatureFn:
    """
    (mid, width) where:
      mid   = (ready + due) / 2
      width = due - ready
    Optional scaling can balance magnitudes.
    """
    rt = np.asarray(problem.ready_time, dtype=float)
    dd = np.asarray(problem.due_time, dtype=float)

    sm = float(scale_mid)
    sw = float(scale_width)

    def f(i: int) -> np.ndarray:
        e = rt[i]
        l = dd[i]
        mid = 0.5 * (e + l)
        width = l - e
        return np.array([sm * mid, sw * width], dtype=float)

    return f


