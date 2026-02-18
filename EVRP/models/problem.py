from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import math
import numpy as np

from .node import Node, NodeType


@dataclass(slots=True)
class EVRPProblem:
    nodes: List[Node]

    depots: List[int]     # global idx list
    customers: List[int]  # global idx list
    css: List[int]        # global idx list

    # arrays for speed (global idx -> value)
    node_type: List[NodeType]
    x: List[float]
    y: List[float]

    demand: List[float]
    service_time: List[float]

    ready_time: List[float]
    due_time: List[float]

    # EV params
    soc_max: float
    vehicle_capacity: float

    # distance matrix computed once at init if not provided
    distance_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        dist = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi, yi = self.x[i], self.y[i]
            for j in range(n):
                dist[i, j] = math.hypot(xi - self.x[j], yi - self.y[j])
        return dist

    def is_customer(self, i: int) -> bool:
        return self.node_type[i] == NodeType.CUSTOMER

    def is_depot(self, i: int) -> bool:
        return self.node_type[i] == NodeType.DEPOT

    def is_cs(self, i: int) -> bool:
        return self.node_type[i] == NodeType.CS
    
    def location(self, i):
        return (self.x[i], self.y[i])
    
    def time_window(self, i):
        return (self.ready_time[i], self.due_time[i])
    
    def nearest_depot(self, i: int) -> tuple[int, float]:
        depots = self.depots
        if not depots:
            raise ValueError("No depots in problem")

        dists = self.distance_matrix[i, depots]
        k = int(np.argmin(dists))
        return depots[k], float(dists[k])


    def nearest_cs(self, i: int) -> tuple[int, float]:
        css = self.css
        if not css:
            raise ValueError("No charging stations in problem")

        dists = self.distance_matrix[i, css]
        k = int(np.argmin(dists))
        return css[k], float(dists[k])


    # def route_initial_load(self, route_customers: Sequence[int]) -> float:
    #     """默认：路线初始载重 = 客户需求之和（可按你的建模替换）"""
    #     return sum(self.demand[c] for c in route_customers)


def build_evrp_problem(nodes: List[Node], ev_params: Dict) -> EVRPProblem:
    node_type = [n.type for n in nodes]
    x = [float(getattr(n, "x", 0.0)) for n in nodes]
    y = [float(getattr(n, "y", 0.0)) for n in nodes]

    demand = [float(getattr(n, "demand", 0.0)) for n in nodes]
    service_time = [float(getattr(n, "service_time", 0.0)) for n in nodes]

    ready_time = [float(getattr(n, "ready_time", 0.0)) for n in nodes]
    due_time   = [float(getattr(n, "due_time", float("inf"))) for n in nodes]

    depots = [n.idx for n in nodes if n.type == NodeType.DEPOT]
    customers = [n.idx for n in nodes if n.type == NodeType.CUSTOMER]
    css = [n.idx for n in nodes if n.type == NodeType.CS]

    if "soc_max" not in ev_params or "vehicle_capacity" not in ev_params:
        raise KeyError("ev_params must include: soc_max, vehicle_capacity")

    soc_max = float(ev_params["soc_max"])
    vehicle_capacity = float(ev_params["vehicle_capacity"])

    return EVRPProblem(
        nodes=nodes,
        depots=depots,
        customers=customers,
        css=css,
        node_type=node_type,
        x = x,
        y = y,
        demand=demand,
        service_time=service_time,
        ready_time=ready_time,
        due_time=due_time,
        soc_max=soc_max,
        vehicle_capacity=vehicle_capacity,
    )
