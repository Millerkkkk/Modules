from __future__ import annotations

from dataclasses import dataclass
from enum import Enum



class NodeType(Enum):
    DEPOT = "d"
    CUSTOMER = "c"
    CS = "f"  # charging station

    @classmethod
    def from_str(cls, s: str) -> "NodeType":
        s2 = s.strip().lower()
        for t in cls:
            if t.value == s2:
                return t
        raise ValueError(f"Unknown node type: {s!r} (expect one of: d/c/f)")
    

@dataclass(slots=True)
class Node:
    idx: int
    name: str
    type: NodeType
    x: float
    y: float
    demand: float = 0.0
    service_time: float = 0.0
    ready_time: float = 0.0
    due_time: float = float("inf")

    def __post_init__(self) -> None:
        if self.idx < 0:
            raise ValueError(f"Node.idx must be >= 0, got {self.idx}")
        if not self.name:
            raise ValueError("Node.name must be non-empty")
        


