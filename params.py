# params.py
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ACOParams:
    num_ants: int = 30
    num_iter: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    epsilon: float = 0.1
    greedy_prob: float = 0.3
    seed: Optional[int] = None
