import copy
import numpy as np
from typing import Dict, List, Tuple, Optional

RoutesByClusters = Dict[int, List[List[int]]]  # cid -> list of GB paths




class Shaker:
    def __init__(self, neighborhood, NS):
        self.N = neighborhood
        self.NS = list(NS)

    def shake(self, routes_by_clusters, k, cid=None):
        sol = copy.deepcopy(routes_by_clusters)
        if not self.NS:
            return sol

        # clamp k into [1, len(NS)]
        kk = max(1, min(int(k), len(self.NS)))
        op = self.NS[kk - 1]

        # pass cid only when operator accepts it (Neighborhood.apply 会自动过滤不需要的参数)
        sol = self.N.apply(op, sol, cid=cid, allow_cross_cluster=True)
        return sol