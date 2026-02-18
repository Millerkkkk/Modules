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




class Evaluator:
    """
    Class version of build_evaluator(...).

    Usage:
        evaluator = Evaluator(problem, ra_safe, ra_risk, margin_energy=0.5, radius_km=3.0)
        eval0, routes_fixed = evaluator.evaluate_init(routes_by_clusters)   # only once at init
        eval_now = evaluator.evaluate_clusters(routes_fixed)                # thereafter
    """

    def __init__(self, problem, ra_safe, ra_risk, *, margin_energy: float = 0.5, radius_km: float = 3.0):
        self.problem = problem

        self.energy_model = EnergyModel()
        self.ra_model = RangeAnxietyModel(problem.soc_max)
        self.decoder = Decoder(
            problem, self.energy_model, self.ra_model,
            ra_safe, ra_risk,
            margin_energy=margin_energy,
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


