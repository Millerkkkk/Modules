from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Iterable
import copy


@dataclass
class Route:
    customers: List[int] = field(default_factory=list)

    def copy(self) -> "Route":
        return Route(customers=self.customers.copy())

    def __len__(self) -> int:
        return len(self.customers)

    def is_empty(self) -> bool:
        return len(self.customers) == 0


@dataclass
class RouteSolution:
    routes: List[Route] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def copy(self) -> "RouteSolution":
        return RouteSolution(
            routes=[r.copy() for r in self.routes],
            metadata=copy.deepcopy(self.metadata),
        )

    def all_customers(self) -> List[int]:
        out = []
        for r in self.routes:
            out.extend(r.customers)
        return out

    def remove_empty_routes(self) -> None:
        self.routes = [r for r in self.routes if not r.is_empty()]

    def num_routes(self) -> int:
        return len(self.routes)

    def contains_customer(self, customer_id: int) -> bool:
        return customer_id in self.all_customers()

    def missing_customers(self, all_customer_ids: Iterable[int]) -> List[int]:
        present = set(self.all_customers())
        return [c for c in all_customer_ids if c not in present]

    def duplicated_customers(self) -> List[int]:
        seen = set()
        dup = []
        for c in self.all_customers():
            if c in seen:
                dup.append(c)
            seen.add(c)
        return dup

    def is_complete(self, all_customer_ids: Iterable[int]) -> bool:
        all_ids = list(all_customer_ids)
        flat = self.all_customers()
        return sorted(flat) == sorted(all_ids)

    def route_load(self, route, instance) -> float:
        raise NotImplementedError

    def is_capacity_feasible(self, instance) -> bool:
        raise NotImplementedError

    def is_feasible(self, instance) -> bool:
        raise NotImplementedError