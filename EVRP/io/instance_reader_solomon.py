from __future__ import annotations

from typing import List

from ..models.node import Node, NodeType


def read_instance(path: str) -> List[Node]:
    """
    读取实例文件，返回 Node 列表（idx 为 0..n-1）。
    期望节点行至少 8 列：
      StringID  Type(d/c/f)  x  y  demand  ready  due  service
    会跳过空行、表头（StringID 开头）以及以 # 开头的注释行。
    """
    nodes: List[Node] = []

    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("StringID"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            node_type_str = parts[1].lower()
            if node_type_str not in ("d", "c", "f"):
                continue

            if len(parts) < 8:
                raise ValueError(
                    f"Bad node line at {path}:{ln} (expect >= 8 cols): {raw.rstrip()}"
                )

            try:
                node = Node(
                    idx=len(nodes),
                    name=parts[0],
                    type=NodeType.from_str(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    demand=float(parts[4]),
                    ready_time=float(parts[5]),
                    due_time=float(parts[6]),
                    service_time=float(parts[7]),
                )
            except Exception as e:
                raise ValueError(f"Parse error at {path}:{ln}: {raw.rstrip()}") from e

            nodes.append(node)

    if not nodes:
        raise ValueError(f"No nodes parsed from {path!r}")

    return nodes
