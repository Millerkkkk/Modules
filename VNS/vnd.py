import copy
import numpy as np


class VNDRefiner:
    def __init__(self, neighborhood, evaluator, rng=None, eps=1e-9):
        self.nb = neighborhood
        self.eval = evaluator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.eps = float(eps)

    @staticmethod
    def _pack(routes, cend):
        return {
            "routes_by_clusters": routes,
            "cost": float(cend["total_cost"]),
            "ra": float(cend["total_ra"]),
            "cend": cend
        }
    

    def refine(self, routes_by_clusters, NL=None, cid=None, 
               tries_per_op=3, max_steps=50):
        """
        顺序 VND：
        - 按 NL 顺序依次尝试
        - 某邻域找到改进：接受并 idx=0 重来
        - 某邻域无改进：idx += 1
        """
        def improves_cost(cand, cur, eps):
            return cand["cost"] < cur["cost"] - eps

        cur_routes = copy.deepcopy(routes_by_clusters)
        cur_cend = self.eval(cur_routes)
        cur = self._pack(cur_routes, cur_cend)

        NL_list = list(NL) if NL else []
        steps = 0
        idx = 0

        while NL_list and steps < max_steps and idx < len(NL_list):
            steps += 1
            op = NL_list[idx]

            improved = False
            best_cand = None

            # 由于 op 是“一次随机 move”，给几次机会
            for _ in range(int(tries_per_op)):
                cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)
                cand_cend = self.eval(cand_routes)
                cand = self._pack(cand_routes, cand_cend)

                if improves_cost(cand, cur, self.eps):
                    improved = True
                    best_cand = cand
                    break  # first-improvement

            if improved:
                cur = best_cand
                idx = 0  # 改进后从第一个邻域重新开始（VND标准做法）
            else:
                idx += 1  # 该邻域无改进，换下一个

        return cur["routes_by_clusters"], cur["cend"], cur






class RVNDRefiner:
    def __init__(self, neighborhood, evaluator, dominates_fn, rng=None, eps=1e-9):
        """
        evaluator(routes)->cend, 其中 cend['total_cost'], cend['total_ra']
        dominates_fn(a,b,eps)->bool，a/b 是 pack 后 dict（含 cost/ra）
        """
        self.nb = neighborhood
        self.eval = evaluator
        self.dominates = dominates_fn
        self.rng = rng if rng is not None else np.random.default_rng()
        self.eps = float(eps)

    @staticmethod
    def _pack(routes, cend):
        return {
            "routes_by_clusters": routes,
            "cost": float(cend["total_cost"]),
            "ra": float(cend["total_ra"]),
            "cend": cend
        }

    def refine(self, routes_by_clusters, *, cid=None, NL=None, NI=None,
               tries_per_op=3, intra_tries_per_op=3,
               max_main_steps=50, max_intra_steps=30):
        """
        返回: (best_routes, best_cend, best_sol_dict)
        """
        cur_routes = copy.deepcopy(routes_by_clusters)
        cur_cend = self.eval(cur_routes)
        cur = self._pack(cur_routes, cur_cend)

        # ---------- 主 RVND ----------
        NL_pool = list(NL) if NL else []
        steps = 0
        while NL_pool and steps < max_main_steps:
            steps += 1
            op = str(self.rng.choice(NL_pool))

            improved = False
            best_cand = None

            # 由于 op 是“一次随机 move”，我们给它几次机会
            for _ in range(int(tries_per_op)):
                cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)
                cand_cend = self.eval(cand_routes)
                cand = self._pack(cand_routes, cand_cend)

                if self.dominates(cand, cur, self.eps):
                    improved = True
                    best_cand = cand
                    break

            if improved:
                cur = best_cand
                # ---------- intra-RVND 精修 ----------
                cur = self._intra_refine(cur, cid=cid, NI=NI,
                                         tries_per_op=intra_tries_per_op,
                                         max_steps=max_intra_steps)
                # 改进后重置 NL
                NL_pool = list(NL)
            else:
                # 该邻域在当前解上没用：移除
                NL_pool = [x for x in NL_pool if x != op]

        return cur["routes_by_clusters"], cur["cend"], cur

    def _intra_refine(self, cur, *, cid=None, NI=None, tries_per_op=3, max_steps=30):
        NI_pool = list(NI) if NI else []
        steps = 0
        while NI_pool and steps < max_steps:
            steps += 1
            op = str(self.rng.choice(NI_pool))

            improved = False
            best_cand = None

            for _ in range(int(tries_per_op)):
                cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)
                cand_cend = self.eval(cand_routes)
                cand = self._pack(cand_routes, cand_cend)

                if self.dominates(cand, cur, self.eps):
                    improved = True
                    best_cand = cand
                    break

            if improved:
                cur = best_cand
                # 精修成功：重置 NI，继续榨干
                NI_pool = list(NI)
            else:
                NI_pool = [x for x in NI_pool if x != op]

        return cur







