from collections import namedtuple
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from Decode.models.energy import EnergyModel
from Decode.models.RA import RangeAnxietyModel
from Decode.decoder import Decoder




@dataclass
class CostParams:
    c1: float = 120.0    # fixed dispatch cost
    c2: float = 0.5    # travel cost per minute
    c3: float = 0.3    # service cost per minute
    c4: float = 0.6    # charging cost per minute



def calculate_cost(ev_number, travel_time, charging_time, service_time, params):

    CostDetail = namedtuple("CostDetail", [
        "dispatch_cost",
        "travel_cost",
        "service_cost",
        "charging_cost",
        # "wait_cost",
        # "delay_cost"
    ])

    dispatch_cost = params.c1 * ev_number
    travel_cost = params.c2 * travel_time
    service_cost = params.c3 * service_time
    charging_cost = params.c4 * charging_time

    cost_detail = CostDetail(dispatch_cost, travel_cost, service_cost, charging_cost)
    total_cost = sum(cost_detail)
    return total_cost, cost_detail


def calculate_cost_spr(plan_ev_number, plan_travel_time, plan_charging_time, plan_service_time, 
                       recourse_travel_time, recourse_charging_time, params):

    CostDetail = namedtuple("CostDetail", [
        "dispatch_cost",
        "plan_travel_cost",
        "plan_service_cost",
        "plan_charging_cost",
        "recourse_travel_cost",
        "recourse_charging_cost"
    ])

    dispatch_cost = params.c1 * plan_ev_number
    plan_travel_cost = params.c2 * plan_travel_time
    plan_service_cost = params.c3 * plan_service_time
    plan_charging_cost = params.c4 * plan_charging_time


    recourse_travel_cost = params.c2 * recourse_travel_time
    recourse_charging_cost = params.c4 * recourse_charging_time


    cost_detail = CostDetail(dispatch_cost, plan_travel_cost, plan_service_cost, plan_charging_cost, 
                             recourse_travel_cost, recourse_charging_cost)
    total_cost = sum(cost_detail)
    return total_cost, cost_detail



def _sum_cost_list(d: dict) -> float:
    return float(sum(float(v) for v in d.values()))

def _stage_costs_from_cend(cend: dict) -> tuple[float, float]:
    # 第一阶段：dispatch + plan travel + plan service + plan charging
    stage1 = (
        _sum_cost_list(cend.get("dispatch_cost_list", {})) +
        _sum_cost_list(cend.get("plan_travel_cost_list", {})) +
        _sum_cost_list(cend.get("plan_service_cost_list", {})) +
        _sum_cost_list(cend.get("plan_charging_cost_list", {}))
    )
    # 第二阶段：recourse travel + recourse charging
    stage2 = (
        _sum_cost_list(cend.get("recourse_travel_cost_list", {})) +
        _sum_cost_list(cend.get("recourse_charging_cost_list", {}))
    )
    return stage1, stage2


class Evaluator:
    """
    Class version of build_evaluator(...).

    Usage:
        evaluator = Evaluator(problem, ra_safe, ra_risk, margin_energy=0.5, radius_km=3.0)
        eval0, routes_fixed = evaluator.evaluate_init(routes_by_clusters)   # only once at init
        eval_now = evaluator.evaluate_clusters(routes_fixed)                # thereafter
    """

    def __init__(self, problem, ra_safe, ra_risk, *, margin_ratio: float = 0.5, radius_km: float = 3.0):
        self.problem = problem

        self.energy_model = EnergyModel()
        self.ra_model = RangeAnxietyModel(problem.soc_max)
        self.decoder = Decoder(
            problem, self.energy_model, self.ra_model,
            ra_safe, ra_risk,
            margin_ratio=margin_ratio,
            radius_km=radius_km,
        )

        self.cost_params = CostParams()

    def evaluate_route(self, flattened_route_global: List[int]) -> Tuple[float, float, float, Any, int, float, float, float, float]:
        route = flattened_route_global.copy()
        if not route:
            return (0.0, 0.0, 0.0, [], 0, 0.0, 0.0, 0.0, 0.0)

        route_demand = sum(self.problem.demand[i] for i in route)

        start_depot, _ = self.problem.nearest_depot(route[0])
        end_depot, _ = self.problem.nearest_depot(route[-1])
        full_route = [start_depot] + route + [end_depot]

        decode_route, travel_dist, travel_time, charging_time, service_time, num_charge, \
        updated_Q, updated_departure_time, updated_load, total_ra = \
            self.decoder.decode_backtrack_charging_final(full_route, route_demand)

        dispatch_ev = 1
        route_cost, cost_detail = calculate_cost(
            dispatch_ev, travel_time, charging_time, service_time, self.cost_params
        )

        return (
            float(total_ra), float(route_cost), float(travel_dist),
            decode_route, int(num_charge),
            float(cost_detail.travel_cost), float(cost_detail.dispatch_cost),
            float(cost_detail.service_cost), float(cost_detail.charging_cost),
        )

    @staticmethod
    def _pack(
        per_cluster_results: List[
            Tuple[int, Tuple[float, float, float, Any, int, float, float, float, float]]
        ]
    ) -> Dict[str, Any]:
        total_cost = 0.0
        total_num_charge = 0

        decoded_routes = {}
        routes_ra = {}
        routes_cost = {}
        routes_dist = {}

        travel_cost_list = {}
        dispatch_cost_list = {}
        service_cost_list = {}
        charging_cost_list = {}

        for cid, (
            ra, route_cost, travel_dist, decode_route, num_charge,
            travel_cost, dispatch_cost, service_cost, charging_cost
        ) in per_cluster_results:
            total_cost += route_cost
            total_num_charge += num_charge

            decoded_routes[cid] = decode_route
            routes_ra[cid] = ra
            routes_cost[cid] = route_cost
            routes_dist[cid] = travel_dist

            travel_cost_list[cid] = travel_cost
            dispatch_cost_list[cid] = dispatch_cost
            service_cost_list[cid] = service_cost
            charging_cost_list[cid] = charging_cost

        return {
            "total_cost": float(total_cost),
            "total_ra": float(sum(routes_ra.values())),
            "decoded_routes": decoded_routes,
            "travel_cost_list": travel_cost_list,
            "dispatch_cost_list": dispatch_cost_list,
            "service_cost_list": service_cost_list,
            "charging_cost_list": charging_cost_list,
            "routes_ra": routes_ra,
            "routes_cost": routes_cost,
            "routes_dist": routes_dist,
            "total_num_charge": int(total_num_charge),
        }

    def evaluate_clusters(self, routes_by_clusters: Dict[int, List[List[int]]]) -> Dict[str, Any]:
        per_cluster_results = []
        for cid, routes in routes_by_clusters.items():
            flat_route = [n for r in routes for n in r]
            per_cluster_results.append((cid, self.evaluate_route(flat_route)))
        return self._pack(per_cluster_results)

    def evaluate_init(self, routes_by_clusters: Dict[int, List[List[int]]]):
        """
        Only for initialization once:
        - For each cluster: evaluate forward/reverse, pick lower route_cost.
        - If reverse is better: reverse the flattened route and slice back by original segment lengths.
        Returns:
            eval_dict, routes_by_clusters_fixed
        """
        routes_fixed = copy.deepcopy(routes_by_clusters)
        per_cluster_results = []

        for cid, gb_segments in routes_fixed.items():
            lens = [len(seg) for seg in gb_segments]
            flat_route = [n for seg in gb_segments for n in seg]

            res_fwd = self.evaluate_route(flat_route)
            res_bwd = self.evaluate_route(list(reversed(flat_route)))

            if res_bwd[1] < res_fwd[1]:  # compare route_cost
                flat_rev = list(reversed(flat_route))
                new_gbs = []
                idx = 0
                for L in lens:
                    new_gbs.append(flat_rev[idx: idx + L])
                    idx += L
                routes_fixed[cid] = new_gbs
                per_cluster_results.append((cid, res_bwd))
            else:
                per_cluster_results.append((cid, res_fwd))

        return self._pack(per_cluster_results), routes_fixed






    def evaluate_route_spr(self, flattened_route_global, real_demand) -> Tuple[float, float, float, Any, int, float, float, float, float, float, float]:
        route = flattened_route_global.copy()
        if not route:
            return (0.0, 0.0, 0.0, [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    
        route_demand = sum(self.problem.demand[i] for i in route)
        EV_load = min(route_demand, self.problem.vehicle_capacity)

        start_depot, _ = self.problem.nearest_depot(route[0])
        end_depot, _ = self.problem.nearest_depot(route[-1])
        full_route = [start_depot] + route + [end_depot]

        decode_route, plan_travel_dist, plan_travel_time, \
        plan_charging_time, plan_service_time, \
        plan_num_charge, Q, departure_time, load, total_ra, \
        recourse_dist, recourse_travel_time, recourse_charging_time, \
        recourse_num_restock, recourse_num_charge = \
            self.decoder.decode_spr(full_route, EV_load, real_demand)
            
        dispatch_ev = 1
        route_cost, cost_detail = calculate_cost_spr(
            dispatch_ev, plan_travel_time, plan_charging_time, plan_service_time, 
            recourse_travel_time, recourse_charging_time, self.cost_params
        )

        total_num_charge = plan_num_charge + recourse_num_charge

        return (
            float(total_ra),
            float(route_cost),
            float(plan_travel_dist),
            decode_route,
            int(total_num_charge),
            float(cost_detail.plan_travel_cost),
            float(cost_detail.dispatch_cost),
            float(cost_detail.plan_service_cost),
            float(cost_detail.plan_charging_cost),
            float(cost_detail.recourse_travel_cost),
            float(cost_detail.recourse_charging_cost),
        )
    
    
    @staticmethod
    def _pack_spr(
        per_cluster_results: List[
            Tuple[int, Tuple[float, float, float, Any, int, float, float, float, float, float, float]]
        ]
    ) -> Dict[str, Any]:

        total_cost = 0.0
        total_num_charge = 0

        decoded_routes = {}
        routes_ra = {}
        routes_cost = {}
        routes_dist = {}

        plan_travel_cost_list = {}
        dispatch_cost_list = {}
        plan_service_cost_list = {}
        plan_charging_cost_list = {}

        recourse_travel_cost_list = {}
        recourse_charging_cost_list = {}

        for cid, (
            ra, route_cost, travel_dist, decode_route, num_charge,
            plan_travel_cost, dispatch_cost, plan_service_cost,
            plan_charging_cost, recourse_travel_cost,
            recourse_charging_cost
        ) in per_cluster_results:

            total_cost += route_cost
            total_num_charge += num_charge

            decoded_routes[cid] = decode_route
            routes_ra[cid] = ra
            routes_cost[cid] = route_cost
            routes_dist[cid] = travel_dist

            plan_travel_cost_list[cid] = plan_travel_cost
            dispatch_cost_list[cid] = dispatch_cost
            plan_service_cost_list[cid] = plan_service_cost
            plan_charging_cost_list[cid] = plan_charging_cost

            recourse_travel_cost_list[cid] = recourse_travel_cost
            recourse_charging_cost_list[cid] = recourse_charging_cost

        return {
            "total_cost": float(total_cost),
            "total_ra": float(sum(routes_ra.values())),
            "decoded_routes": decoded_routes,

            "plan_travel_cost_list": plan_travel_cost_list,
            "dispatch_cost_list": dispatch_cost_list,
            "plan_service_cost_list": plan_service_cost_list,
            "plan_charging_cost_list": plan_charging_cost_list,

            "recourse_travel_cost_list": recourse_travel_cost_list,
            "recourse_charging_cost_list": recourse_charging_cost_list,

            "routes_ra": routes_ra,
            "routes_cost": routes_cost,
            "routes_dist": routes_dist,
            "total_num_charge": int(total_num_charge),
        }

    def evaluate_clusters_spr(self, routes_by_clusters, real_demand):
        per_cluster_results = []
        for cid, routes in routes_by_clusters.items():
            flat_route = [n for r in routes for n in r]
            per_cluster_results.append((cid, self.evaluate_route_spr(flat_route, real_demand)))
        return self._pack_spr(per_cluster_results)

    def evaluate_init_spr(self, routes_by_clusters, real_demand):
        """
        Only for initialization once:
        - For each cluster: evaluate forward/reverse, pick lower route_cost.
        - If reverse is better: reverse the flattened route and slice back by original segment lengths.
        Returns:
            eval_dict, routes_by_clusters_fixed
        """
        routes_fixed = copy.deepcopy(routes_by_clusters)
        per_cluster_results = []

        for cid, gb_segments in routes_fixed.items():
            lens = [len(seg) for seg in gb_segments]
            flat_route = [n for seg in gb_segments for n in seg]

            res_fwd = self.evaluate_route_spr(flat_route, real_demand)
            res_bwd = self.evaluate_route_spr(list(reversed(flat_route)), real_demand)

            if res_bwd[1] < res_fwd[1]:  # compare route_cost
                flat_rev = list(reversed(flat_route))
                new_gbs = []
                idx = 0
                for L in lens:
                    new_gbs.append(flat_rev[idx: idx + L])
                    idx += L
                routes_fixed[cid] = new_gbs
                per_cluster_results.append((cid, res_bwd))
            else:
                per_cluster_results.append((cid, res_fwd))

        return self._pack_spr(per_cluster_results), routes_fixed




    def evaluate_saa(self, routes_by_clusters, scenarios):
        S = int(scenarios.shape[0])

        cends = []
        total_costs = []
        total_ras = []
        stage1_samples = []
        stage2_samples = []

        for s in range(S):
            real_demand = scenarios[s]
            cend = self.evaluate_clusters_spr(routes_by_clusters, real_demand)  # full dict

            cends.append(cend)
            total_costs.append(float(cend["total_cost"]))
            total_ras.append(float(cend.get("total_ra", 0.0)))

            st1, st2 = _stage_costs_from_cend(cend)
            stage1_samples.append(st1)
            stage2_samples.append(st2)


        # 代表场景：总成本最接近平均成本
        avg_total_cost = sum(total_costs) / S
        rep_idx = min(range(S), key=lambda i: abs(total_costs[i] - avg_total_cost))
        avg_cend = copy.deepcopy(cends[rep_idx])

        # 覆盖成平均值（数值字段）
        avg_cend["total_cost"] = float(avg_total_cost)
        avg_cend["total_ra"] = float(sum(total_ras) / S)

        # ✅新增：输出第一/二阶段平均值
        avg_cend["stage1_cost"] = float(sum(stage1_samples) / S)
        avg_cend["stage2_cost"] = float(sum(stage2_samples) / S)

        # 可选：保留样本，方便你检查方差/分布
        avg_cend["stage1_cost_samples"] = stage1_samples
        avg_cend["stage2_cost_samples"] = stage2_samples
        avg_cend["saa_rep_scenario_index"] = int(rep_idx)

        return avg_cend

    def evaluate_init_saa(self, routes_by_clusters, scenarios, eps: float = 1e-9):
        """
        初始化阶段：对每个 cluster 做 forward / reverse 比较（比较的是该 cluster 的 SAA 平均 total_cost），
        选更优方向后更新 routes_fixed；最后返回全体 cluster 的 SAA 评估与 routes_fixed。

        Returns:
            eval_dict (dict), routes_by_clusters_fixed (dict)
        """
        routes_fixed = copy.deepcopy(routes_by_clusters)

        for cid, gb_segments in routes_fixed.items():
            lens = [len(seg) for seg in gb_segments]
            flat_route = [n for seg in gb_segments for n in seg]

            # forward：直接用原 segments
            cend_fwd = self.evaluate_saa({cid: gb_segments}, scenarios)
            cost_fwd = float(cend_fwd["total_cost"])

            # backward：反转扁平序列后按原段长度切回 segments
            flat_rev = list(reversed(flat_route))
            new_gbs = []
            idx = 0
            for L in lens:
                new_gbs.append(flat_rev[idx: idx + L])
                idx += L

            cend_bwd = self.evaluate_saa({cid: new_gbs}, scenarios)
            cost_bwd = float(cend_bwd["total_cost"])

            if cost_bwd < cost_fwd - eps:
                routes_fixed[cid] = new_gbs

        # 对全体 routes 做一次 SAA 汇总评估
        eval_dict = self.evaluate_saa(routes_fixed, scenarios)
        return eval_dict, routes_fixed
    

