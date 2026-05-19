from __future__ import annotations

from typing import Dict, Tuple

from EVRP.models.problem import EVRPProblem, build_evrp_problem
from EVRP.io.instance_reader_solomon import read_instance




def build_customer_mappings(customers_global_indices: list[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    customer_id: 1..n_customers
    cust_id_to_global: customer_id -> global_idx
    global_to_cust_id: global_idx -> customer_id
    """
    cust_id_to_global = {i + 1: customers_global_indices[i] for i in range(len(customers_global_indices))}
    global_to_cust_id = {customers_global_indices[i]: i + 1 for i in range(len(customers_global_indices))}
    return cust_id_to_global, global_to_cust_id


def load_problem(
    path: str,
    ev_params: Dict,
    *,
    return_mappings: bool = False,
) -> EVRPProblem | Tuple[EVRPProblem, Dict]:
    """
    对外入口：
      输入：实例文件路径 + ev_params
      输出：EVRPProblem
      可选：附带 customer 映射（1..n <-> global idx）
    """
    nodes = read_instance(path)
    problem = build_evrp_problem(nodes, ev_params)

    if not return_mappings:
        return problem

    cust_id_to_global, global_to_cust_id = build_customer_mappings(problem.customers)
    mappings = {
        "cust_id_to_global": cust_id_to_global,
        "global_to_cust_id": global_to_cust_id,
    }
    return problem, mappings
