import copy
import time
import numpy as np


def improves_cost(cand, cur, eps):
    return cand["cost"] < cur["cost"] - eps

class VNDRefiner:
    def __init__(self, neighborhood, evaluator, dominates_fn, eval_proxy=None, rng=None, eps=1e-9, same_eval=None,):
        """
        neighborhood: 邻域算子容器，需支持 nb.apply(op, routes_by_clusters, cid=cid)
        evaluator:    full evaluator, evaluator(routes)->cend
        dominates_fn: dominates(a, b, eps)->bool
                      其中 a/b 至少含 keys: "cost", "ra"
        eval_proxy:   便宜评估器；若为 None，则退化为 evaluator
        """
        self.nb = neighborhood
        self.eval = evaluator
        self.dominates = dominates_fn
        self.eval_proxy = eval_proxy if eval_proxy is not None else evaluator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.eps = float(eps)

        if same_eval is None:
            self.same_eval = (self.eval_proxy is self.eval)
        else:
            self.same_eval = bool(same_eval)

    @staticmethod
    def _pack(routes, cend):
        return {
            "routes_by_clusters": routes,
            "cost": float(cend["total_cost"]),
            "ra": float(cend["total_ra"]),
            "cend": cend
        }
    
    @staticmethod
    def _signature(routes_by_clusters):
        """
        给 routes_by_clusters 做一个可哈希签名，用于 cache。
        假设格式:
            {
              cid1: [[...], [...]],
              cid2: [[...], ...]
            }
        """
        items = []
        for c in sorted(routes_by_clusters.keys()):
            routes_t = tuple(tuple(seg) for seg in routes_by_clusters[c])
            items.append((c, routes_t))
        return tuple(items)
    
    def refine1(self, routes_by_clusters, NL=None, cid=None, tries_per_op=3, max_steps=50):
        """
        顺序 VND（full eval 版本）：
        - 按 NL 顺序依次尝试
        - 某邻域找到支配改进：接受并 idx=0 重来
        - 某邻域无改进：idx += 1

        返回:
            best_routes, best_cend, best_sol_dict
        """
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

            # 一个 op 本身可能是随机 move，所以给多次尝试机会
            for _ in range(int(tries_per_op)):
                cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)
                cand_cend = self.eval(cand_routes)
                cand = self._pack(cand_routes, cand_cend)

                if self.dominates(cand, cur, self.eps):
                    improved = True
                    best_cand = cand
                    break  # first-improvement

            if improved:
                cur = best_cand
                idx = 0
            else:
                idx += 1

        return cur["routes_by_clusters"], cur["cend"], cur
    
    def refine(self, routes_by_clusters, NL=None, cid=None, tries_per_op=3, max_steps=50):
        """
        顺序 VND（proxy + cache + full eval 版本）：
        - 内部用 eval_proxy 判断 move 是否支配当前解（带缓存）
        - refine 结束后用 eval_full 只评一次最终解用于返回
        - 统计 second-stage time（proxy miss + final full eval）

        返回:
            final_routes, final_cend_full, info
        其中 info 保持和你现有外层用法兼容，并额外包含：
            - cost
            - ra
            - t_stage2
            - n_cache
        """

        eval_cache = {}
        t_stage2 = 0.0

        def _eval_proxy_cached(rbc):
            nonlocal t_stage2
            sig = self._signature(rbc)
            hit = eval_cache.get(sig, None)
            if hit is not None:
                return hit

            t0 = time.time()
            cend = self.eval_proxy(rbc)
            t_stage2 += time.time() - t0

            eval_cache[sig] = cend
            return cend

        # 初始化当前解（proxy）
        cur_routes = copy.deepcopy(routes_by_clusters)
        cur_cend_proxy = _eval_proxy_cached(cur_routes)
        cur = self._pack(cur_routes, cur_cend_proxy)

        NL_list = list(NL) if NL else []
        steps = 0
        idx = 0

        while NL_list and steps < max_steps and idx < len(NL_list):
            steps += 1
            op = NL_list[idx]

            improved = False
            best_cand = None

            for _ in range(int(tries_per_op)):
                cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)
                cand_cend_proxy = _eval_proxy_cached(cand_routes)
                cand = self._pack(cand_routes, cand_cend_proxy)

                if self.dominates(cand, cur, self.eps):
                    improved = True
                    best_cand = cand
                    break  # first-improvement

            if improved:
                cur = best_cand
                idx = 0
            else:
                idx += 1

        # ✅ full eval 也属于第二阶段：要计时
        final_routes = copy.deepcopy(cur["routes_by_clusters"])

        if self.same_eval:
            final_cend_full = cur["cend"]
        else:
            t0 = time.time()
            final_cend_full = self.eval(final_routes)
            t_stage2 += time.time() - t0

        # ✅ 不改变你外层解包习惯：把 t_stage2 放在第三返回值里
        info = {
            "routes_by_clusters": final_routes,
            "cost": float(final_cend_full["total_cost"]),
            "ra": float(final_cend_full.get("total_ra", 0.0)),
            "cend": final_cend_full,
            "t_stage2": float(t_stage2),
            "n_cache": int(len(eval_cache)),
        }
        return final_routes, final_cend_full, info

    # def refine(self, routes_by_clusters, NL=None, cid=None, tries_per_op=3, max_steps=50):
    #     """
    #     顺序 VND（full eval + cache 版本）
    #     """

    #     eval_cache = {}

    #     def _eval_cached(rbc):
    #         sig = self._signature(rbc)
    #         if sig in eval_cache:
    #             return eval_cache[sig]

    #         cend = self.eval(rbc)
    #         eval_cache[sig] = cend
    #         return cend

    #     cur_routes = copy.deepcopy(routes_by_clusters)
    #     cur_cend = _eval_cached(cur_routes)
    #     cur = self._pack(cur_routes, cur_cend)

    #     NL_list = list(NL) if NL else []
    #     steps = 0
    #     idx = 0

    #     while NL_list and steps < max_steps and idx < len(NL_list):
    #         steps += 1
    #         op = NL_list[idx]

    #         improved = False
    #         best_cand = None

    #         for _ in range(int(tries_per_op)):
    #             cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)

    #             cand_cend = _eval_cached(cand_routes)
    #             cand = self._pack(cand_routes, cand_cend)

    #             if self.dominates(cand, cur, self.eps):
    #                 improved = True
    #                 best_cand = cand
    #                 break

    #         if improved:
    #             cur = best_cand
    #             idx = 0
    #         else:
    #             idx += 1

    #     return (
    #         copy.deepcopy(cur["routes_by_clusters"]),
    #         copy.deepcopy(cur["cend"]),
    #         copy.deepcopy(cur),
    #     )


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







