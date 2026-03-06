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



@dataclass
class GAParams:
    seed: int = 42
    pop_size: int = 80
    num_gen: int = 300

    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2

    # mutation details
    inversion_rate: float = 0.7   # 变异时用 inversion 的概率，否则用 swap

    # elitism
    elite_size: int = 2
