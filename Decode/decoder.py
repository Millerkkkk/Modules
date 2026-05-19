#  coding: UTF-8  #
'''
@Project     : decoder
@File        : evaluate.py
@IDE         : VSCode
@Author      : Yingkai
@Date        : 2025/08/08 15:29
'''



import copy
import numpy as np




# =========================
# Decoder 解码器 (把ant_route 解码为插入cs后的完整route)
# =========================
class Decoder:
    def __init__(self, problem, energy_model, ra_model, 
                 ra_safe, ra_risk, 
                 margin_ratio=0.1, radius_km=3.0):
        
        self.problem = problem
        # self.charging_nodes = list(self.problem.css) + list(self.problem.depots)
        self.charging_nodes = list(self.problem.css)

        self.energy_model = energy_model
        self.margin_ratio = margin_ratio
        self.radius_km = radius_km
        
        self.ra_model = ra_model        # 续航焦虑模型（RangeAnxietyModel）
        self.ra_safe = ra_safe          # R < ra_safe：安全区
        self.ra_risk = ra_risk          # R >= ra_risk：风险区

        self.EPS = 1e-9  # 浮点容差，避免边界抖动

    def _build_anchor_cs_map(self, route):
        """
        为每个节点找一个 3km 内的 CS（取最近/能量最小的一个），形成 {node: cs} 映射。
        """
        anchor_cs = {}
        for n in route:
            best = None
            best_d = float('inf')
            for cs in self.problem.css:
                d = self.problem.distance_matrix[n, cs]
                if d <= self.radius_km and d < best_d:
                    best = cs
                    best_d = d
            if best is not None:
                anchor_cs[n] = best
        return anchor_cs

    def _estimate_partial_charge(self, cs_index_in_route, route, Q_remain, departure_time, load):
        """
        预测从当前充电站 curr_cs 出发，到达最终 depot 的整段路径能耗。
        如果预计能耗 > Q_max，则充满；
        否则按需充电，返回 required_charge 和 charge_time。
        """
        Q_max = self.problem.soc_max
        margin = self.margin_ratio * Q_max      # 安全余量（若你的业务允许“刚好到达”即可，可将 margin 设为 0）

        total_energy_needed = 0.0
        temp_departure = departure_time
        temp_load = load

        # 从 curr_cs 出发，到达 route[i+1:] 全部节点，包括终点
        for j in range(cs_index_in_route, len(route)-1):
            prev_node = route[j]
            curr_node = route[j+1]

            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy, travel_time, _ = self.energy_model.calculate_one_node_energy_time(
                dist, temp_departure, temp_load
            )

            total_energy_needed += energy
            temp_departure += travel_time

            # 到客户点服务 + 卸货
            service_time = self.problem.service_time[curr_node]
            demand = self.problem.demand[curr_node]

            temp_departure += service_time
            temp_load -= demand

        total_energy_needed += margin

        # 计算所需充电量
        theoretical_required = max(0.0, total_energy_needed - Q_remain)
        required_charge = min(Q_max - Q_remain, theoretical_required)

        charge_time = self.energy_model.calculate_charging_time(required_charge)

        return required_charge, charge_time

    def _can_reach_final_depot(self, cs_index_in_route, route, Q_after_charge, departure_time, load):
        "判断从某个充电站出发，在当前路径结构下，是否可以不再进桩直接完成剩余路径直到终点 depot。"
        "用于判断是否可以安全启用 “锁定直奔终点模式”，从而跳过之后 curr → CS 的路径检查"
        Q = Q_after_charge
        t = departure_time
        l = load
        for i in range(cs_index_in_route, len(route) - 1):
            u, v = route[i], route[i + 1]
            dist = self.problem.distance_matrix[u, v]
            e, dt, _ = self.energy_model.calculate_one_node_energy_time(dist, t, l)
            if Q + self.EPS < e:
                return False
            Q -= e
            t += dt
            t += self.problem.service_time[v]
            l -= self.problem.demand[v]
        return True

    def _to_nearest_cs(self, prev_node, Q_at_prev, t_depart_prev, load_prev):
            # 找 charging_nodes 中距离最近的节点
            dm = self.problem.distance_matrix
            nearest_node = min(self.charging_nodes, key=lambda n: dm[prev_node, n])
            dist = dm[prev_node, nearest_node]

            e, dt, _ = self.energy_model.calculate_one_node_energy_time(dist, t_depart_prev, load_prev)
            if Q_at_prev + self.EPS >= e:
                return (nearest_node, dist, e, dt)
            return None
    
    def _ra_mode(self, ra):
        """根据当前 RA 值把状态分成三类：safe / opportunity / risk。"""
        if self.ra_model is None:
            return "safe_mode"  # 没有模型就当作不焦虑

        if ra >= self.ra_risk:
            return "risk_mode"
        elif ra < self.ra_safe:
            return "safe_mode"
        else:
            return "opportunity_mode"


    def decode_backtrack_charging_final(self, route, load, departure_time=0): 
        """
        与 decode_backtrack_charging 类似，（前瞻 + 回退插桩策略）+ low_soc_mode 模式调试输出
        区别是：存在两个节点之间距离过于长从而不可达，在检测到不可达并回退插桩后，若依然无法到达下一个目标，
        此时直接返回一个“大惩罚解”，而不是无限尝试或报错。
        """
        # print(route)
        Q = self.problem.soc_max
        EPS = self.EPS
        lock_to_depot = False   # 是否锁定“直奔终点模式”

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])  # {node: nearest_cs}， route[1:-1]：去掉首尾depot
        # print('self.cs_in_3km_node', self.cs_in_3km_node)

        # ---------- RA 状态 ----------
        t_since_charge = 0.0   # 从上次充电起的累计行驶时间（分钟）
        Q_ref = Q              # 上次充电后的电量基线（用于相对消耗Qtr）
        total_ra = 0.0         # 累计焦虑（SUMR 增量累加）

        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0

        state_log = []
        i = 1

        BIG_PENALTY = 1e6           # 无解惩罚
        WATCHDOG_LIMIT = 200        # 看门狗阈值，防死循环
        watchdog = 0
        tried_pairs = set()  # 记录 (prev_i, cs_id) 已尝试过的回退插桩，防止反复

        # 防止“回退插桩分支已充电”后，到达同一CS又重复充电/计数
        skip_charge_once = False
        skip_charge_cs_id = None

        # 统一的惩罚返回
        def penalty_exit():
            return (
                route,
                BIG_PENALTY,    # total_dist
                BIG_PENALTY,    # total_travel_time
                BIG_PENALTY,    # total_charging_time
                BIG_PENALTY,    # total_service_time
                999,            # num_charge
                0.0,            # Q
                departure_time, # departure_time
                load,           # load
                BIG_PENALTY     # total_ra
            )

        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                # print("⚠️ 看门狗触发 → 直接惩罚退出")
                return penalty_exit()

            prev_node = route[i - 1]
            curr_node = route[i]
            # print(f"\n[{i}] 当前处理: {prev_node} → {curr_node}, 当前电量: {Q:.2f}, 负载: {load}, 出发时间: {departure_time:.2f}")
            # print(route)
            # 记录快照（用于回退）
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "Q": Q,
                "Q_ref": Q_ref,
                "t_since_charge": t_since_charge,
                "departure_time": departure_time,
                "load": load,
                "total_travel_time": total_travel_time,
                "total_service_time": total_service_time,
                "total_charging_time": total_charging_time,
                "total_dist": total_dist,
                "total_ra": total_ra,
                "lock_to_depot": lock_to_depot,
            })

            # === 计算 RA & 模式（分钟制）===
            if getattr(self, "ra_model", None) is not None:
                # 用“相对上次充电后的消耗”实现：充电后瞬时 RA≈0
                Qtr = max(0.0, Q_ref - Q)  # 已消耗（相对上次充电）
                Ttr = t_since_charge       # 分钟
                hour_of_day = (departure_time / 60.0) % 24.0  # departure_time也是分钟的话
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                ra = 0.0
                mode = "safe_mode"


            # ---------- risk_mode：主动强制去最近可达 CS ----------
            if (not lock_to_depot) and (mode == "risk_mode"):
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    # 若当前目标已经是这个CS就不重复插
                    if curr_node != cs_id:
                        route.insert(i, cs_id)
                        continue
                    # 如果从 prev_node 到不了任何CS，则后面“不可达”时会触发回溯；回溯仍失败则惩罚退出

            # ---------------- opportunity_mode：只做“位置选择 + 插入”，不做任何状态更新 ----------------
            if (not lock_to_depot) and (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                # print(f"💡 当前节点 {curr_node} 附近有焦虑插桩可用 CS: {cs_id}")

                # 邻接去重：若已经是 [prev, cs, curr] 或 [prev, curr, cs]，则不重复插
                already_before = (route[i-1] == cs_id) if i-1 >= 0 else False
                already_after  = (i+1 < len(route) and route[i+1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i+1] if i+1 < len(route) else None
                    # 以距离增量选择前插/后插
                    d_prev_cs   = self.problem.distance_matrix[prev_node, cs_id]
                    d_cs_curr   = self.problem.distance_matrix[cs_id,   curr_node]

                    d_prev_curr = self.problem.distance_matrix[prev_node, curr_node]
                    d_curr_cs   = self.problem.distance_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.problem.distance_matrix[curr_node, next_node]
                        d_cs_next   = self.problem.distance_matrix[cs_id,    next_node]
                        # 比较四段：prev-cs-curr-next vs prev-curr-cs-next
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost  = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        # 没有 next：退化成三段比较
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost  = d_prev_curr + d_curr_cs

                    if insert_before_cost  + self.EPS < insert_after_cost:
                        route.insert(i, cs_id)
                        inserted_now = True
                        # print("前插 CS", cs_id)
                    else:
                        route.insert(i+1, cs_id)
                        inserted_now = True
                        # print("后插 CS", cs_id)

                # 只插不算：回到同一 prev（i 不变），下一轮再统一推进
                if inserted_now:
                    continue

            # ---------- 常规前进 prev → curr ----------
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(dist, departure_time, load)
            # print(f"➡️ 前往 {curr_node}，需要能量: {energy_needed:.2f}, 当前电量: {Q:.2f}")
            # print('   需要时间：', travel_time, "服务时间", self.all_nodes[curr_node].service_time)

            # 不可达则回退
            if Q + EPS < energy_needed:
                # print(f"⛔ 无法前往 {curr_node}，开始回退寻找可达 CS 插桩点")
                inserted = False

                while state_log:
                    snap = state_log.pop()
                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]

                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]
                    snap_ra_total = snap["total_ra"]
                    snap_lock = snap["lock_to_depot"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        # print(f"🔄 回退点 {prev_node2} 无法到达任何 CS，继续回退...")
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result
                    # print(f"🔁 在回退点 {prev_node2} 后插入 CS[{cs_id}]")

                    # 保险：避免将客户节点当成 CS；避免重复相邻插桩
                    if cs_id == prev_node2:
                        # print(f"⚠️ 最近CS {cs_id} 与回退点相同，跳过该回退点")
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        # print(f"⚠️ CS[{cs_id}] 已经在 {prev_node2} 后面，跳过插入")
                        continue
                    # 防回绕：同一 (prev_i, cs_id) 不重复尝试
                    if (prev_i, cs_id) in tried_pairs:
                        # print(f"⚠️ (prev_i={prev_i}, cs={cs_id}) 已尝试过，仍走回到此状态 → 直接惩罚退出")
                        return penalty_exit()
                    tried_pairs.add((prev_i, cs_id))

                    # 执行插入
                    snap_route.insert(prev_i + 1, cs_id)
                    # 先到CS
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy
                    # 估算部分充电
                    required_charge, charge_time = self._estimate_partial_charge(prev_i + 1, snap_route, Q_arrive_cs, t_arrive_cs, snap_load)

                    # 更新主状态（回退点之后继续）
                    Q = min(self.problem.soc_max, Q_arrive_cs + required_charge)
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load
                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_charging_time = snap_ct + charge_time
                    total_dist = snap_dist + to_cs_dist
                    num_charge += 1

                    # ---- 回退插桩这次充电后：重置“从上次充电起行驶时间”和Q_ref ----
                    t_since_charge = 0.0
                    Q_ref = Q

                    # ---- 累计焦虑：把 prev_node2 -> cs_id 这一段的 SUMR 增量加上 ----
                    if getattr(self, "ra_model", None) is not None:
                        # 段开始权重取回退快照的状态（段起点=prev_node2）
                        Ttr0 = snap_t_since
                        hour0 = (snap_t / 60.0) % 24.0

                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy

                        seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        total_ra = snap_ra_total + seg_ra
                    else:
                        total_ra = snap_ra_total

                    # 锁定直奔终点
                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True

                    # print(f"🔁 插入回退充电桩 CS[{cs_id}] 于 {prev_node2} 之后，恢复 Q = {Q:.2f}，继续主路径")

                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    # 防二次充电：下一次真正“到达这个cs节点”时跳过充电块一次
                    skip_charge_once = True
                    skip_charge_cs_id = cs_id

                    # ⭐ 插完就立刻检查“下一跳”是否可达；不可达则直接惩罚退出
                    if i < len(route):
                        next_node = route[i]
                        dist2 = self.problem.distance_matrix[route[i - 1], next_node]
                        energy2, _, _ = self.energy_model.calculate_one_node_energy_time(
                            dist2, departure_time, load
                        )
                        if Q + EPS < energy2:
                            # print(f"⚠️ 插了 CS[{cs_id}] 后依然到不了 {next_node}，直接返回惩罚")
                            return penalty_exit()
                    
                    break

                # 可能存在 两点之间距离太远，即使充电也不能到达的情况
                if not inserted:
                    # print("⚠️ 回退点用尽仍不可达 → 直接惩罚退出")
                    return penalty_exit()
                
                # 成功插入后，继续主 while
                continue
            
            # ---------- 可达：推进 prev→curr ----------
            if getattr(self, "ra_model", None) is not None:
                # 段起点权重（出发前）
                Ttr0 = t_since_charge
                hour0 = (departure_time / 60.0) % 24.0

                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed

                seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                total_ra += seg_ra

            # 2) 走这段路
            arrival_time = departure_time + travel_time
            service_time = self.problem.service_time[curr_node]
            departure_time = arrival_time + service_time
            Q -= energy_needed
            load -= self.problem.demand[curr_node]

            total_travel_time += travel_time
            total_service_time += service_time
            total_dist += dist
            # print(f"✅ 到达 {curr_node}：Q={Q:.2f}, t={departure_time:.2f}")

            # “从上次充电起的行驶时间”只加行驶，不加服务
            t_since_charge += travel_time

            # ---------- 到达CS：执行充电 ----------
            if curr_node in self.problem.css:
                # 若这是“回退插桩分支已算过充电”的那一次，到达时跳过一次
                if skip_charge_once and (curr_node == skip_charge_cs_id):
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    required_charge, charge_time = self._estimate_partial_charge(
                        i, route, Q, departure_time, load
                    )

                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.problem.soc_max, Q + required_charge)
                        departure_time += charge_time
                        total_charging_time += charge_time
                        num_charge += 1

                    # 充电后：焦虑参考点重置（实现“瞬时RA≈0”）
                    t_since_charge = 0.0
                    Q_ref = Q

                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True
            # print('---------------------------------', ra, total_ra)
            i += 1

        # print(f"\n🔚 解码完成，总充电次数: {num_charge}, 总距离: {total_dist:.2f}, 总行驶时间: {total_travel_time:.2f}, 剩余电量: {Q:.2f}")
        return (
            route, total_dist, total_travel_time,
            total_charging_time, total_service_time,
            num_charge, Q, departure_time, load, total_ra
        )






    
    def decode_ccp(self, route, load, departure_time=0): 
        """
        CCP decoder:
        - 路径的需求可行性已在上游通过 CCP 分簇/分配保证
        - 本函数不再检查容量机会约束
        - 仅负责在给定路径下进行 EV 执行解码：
            1) 电量推进
            2) 焦虑驱动插桩
            3) 不可达时回退插入可达充电站
            4) 若修复后仍不可达，则返回惩罚解
        """
        # print(route)
        Q = self.problem.soc_max
        EPS = self.EPS
        lock_to_depot = False   # 是否锁定“直奔终点模式”

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])  # {node: nearest_cs}， route[1:-1]：去掉首尾depot
        # print('self.cs_in_3km_node', self.cs_in_3km_node)

        # ---------- RA 状态 ----------
        t_since_charge = 0.0   # 从上次充电起的累计行驶时间（分钟）
        Q_ref = Q              # 上次充电后的电量基线（用于相对消耗Qtr）
        total_ra = 0.0         # 累计焦虑（SUMR 增量累加）

        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0

        state_log = []
        i = 1

        BIG_PENALTY = 1e6           # 无解惩罚
        WATCHDOG_LIMIT = 200        # 看门狗阈值，防死循环
        watchdog = 0
        tried_pairs = set()  # 记录 (prev_i, cs_id) 已尝试过的回退插桩，防止反复

        # 防止“回退插桩分支已充电”后，到达同一CS又重复充电/计数
        skip_charge_once = False
        skip_charge_cs_id = None

        # 统一的惩罚返回
        def penalty_exit():
            return (
                route,
                BIG_PENALTY,    # total_dist
                BIG_PENALTY,    # total_travel_time
                BIG_PENALTY,    # total_charging_time
                BIG_PENALTY,    # total_service_time
                999,            # num_charge
                0.0,            # Q
                departure_time, # departure_time
                load,           # load
                BIG_PENALTY     # total_ra
            )

        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                # print("⚠️ 看门狗触发 → 直接惩罚退出")
                return penalty_exit()

            prev_node = route[i - 1]
            curr_node = route[i]
            # print(f"\n[{i}] 当前处理: {prev_node} → {curr_node}, 当前电量: {Q:.2f}, 负载: {load}, 出发时间: {departure_time:.2f}")
            # print(route)
            # 记录快照（用于回退）
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "Q": Q,
                "Q_ref": Q_ref,
                "t_since_charge": t_since_charge,
                "departure_time": departure_time,
                "load": load,
                "total_travel_time": total_travel_time,
                "total_service_time": total_service_time,
                "total_charging_time": total_charging_time,
                "total_dist": total_dist,
                "total_ra": total_ra,
                "lock_to_depot": lock_to_depot,
            })

            # === 计算 RA & 模式（分钟制）===
            if getattr(self, "ra_model", None) is not None:
                # 用“相对上次充电后的消耗”实现：充电后瞬时 RA≈0
                Qtr = max(0.0, Q_ref - Q)  # 已消耗（相对上次充电）
                Ttr = t_since_charge       # 分钟
                hour_of_day = (departure_time / 60.0) % 24.0  # departure_time也是分钟的话
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                ra = 0.0
                mode = "safe_mode"


            # ---------- risk_mode：主动强制去最近可达 CS ----------
            if (not lock_to_depot) and (mode == "risk_mode"):
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    # 若当前目标已经是这个CS就不重复插
                    if curr_node != cs_id:
                        route.insert(i, cs_id)
                        continue
                    # 如果从 prev_node 到不了任何CS，则后面“不可达”时会触发回溯；回溯仍失败则惩罚退出

            # ---------------- opportunity_mode：只做“位置选择 + 插入”，不做任何状态更新 ----------------
            if (not lock_to_depot) and (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                # print(f"💡 当前节点 {curr_node} 附近有焦虑插桩可用 CS: {cs_id}")

                # 邻接去重：若已经是 [prev, cs, curr] 或 [prev, curr, cs]，则不重复插
                already_before = (route[i-1] == cs_id) if i-1 >= 0 else False
                already_after  = (i+1 < len(route) and route[i+1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i+1] if i+1 < len(route) else None
                    # 以距离增量选择前插/后插
                    d_prev_cs   = self.problem.distance_matrix[prev_node, cs_id]
                    d_cs_curr   = self.problem.distance_matrix[cs_id,   curr_node]

                    d_prev_curr = self.problem.distance_matrix[prev_node, curr_node]
                    d_curr_cs   = self.problem.distance_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.problem.distance_matrix[curr_node, next_node]
                        d_cs_next   = self.problem.distance_matrix[cs_id,    next_node]
                        # 比较四段：prev-cs-curr-next vs prev-curr-cs-next
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost  = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        # 没有 next：退化成三段比较
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost  = d_prev_curr + d_curr_cs

                    if insert_before_cost  + self.EPS < insert_after_cost:
                        route.insert(i, cs_id)
                        inserted_now = True
                        # print("前插 CS", cs_id)
                    else:
                        route.insert(i+1, cs_id)
                        inserted_now = True
                        # print("后插 CS", cs_id)

                # 只插不算：回到同一 prev（i 不变），下一轮再统一推进
                if inserted_now:
                    continue

            # ---------- 常规前进 prev → curr ----------
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(dist, departure_time, load)
            # print(f"➡️ 前往 {curr_node}，需要能量: {energy_needed:.2f}, 当前电量: {Q:.2f}")
            # print('   需要时间：', travel_time, "服务时间", self.all_nodes[curr_node].service_time)

            # 不可达则回退
            if Q + EPS < energy_needed:
                # print(f"⛔ 无法前往 {curr_node}，开始回退寻找可达 CS 插桩点")
                inserted = False

                while state_log:
                    snap = state_log.pop()
                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]

                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]
                    snap_ra_total = snap["total_ra"]
                    snap_lock = snap["lock_to_depot"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        # print(f"🔄 回退点 {prev_node2} 无法到达任何 CS，继续回退...")
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result
                    # print(f"🔁 在回退点 {prev_node2} 后插入 CS[{cs_id}]")

                    # 保险：避免将客户节点当成 CS；避免重复相邻插桩
                    if cs_id == prev_node2:
                        # print(f"⚠️ 最近CS {cs_id} 与回退点相同，跳过该回退点")
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        # print(f"⚠️ CS[{cs_id}] 已经在 {prev_node2} 后面，跳过插入")
                        continue
                    # 防回绕：同一 (prev_i, cs_id) 不重复尝试
                    if (prev_i, cs_id) in tried_pairs:
                        # print(f"⚠️ (prev_i={prev_i}, cs={cs_id}) 已尝试过，仍走回到此状态 → 直接惩罚退出")
                        return penalty_exit()
                    tried_pairs.add((prev_i, cs_id))

                    # 执行插入
                    snap_route.insert(prev_i + 1, cs_id)
                    # 先到CS
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy
                    # 估算部分充电
                    required_charge, charge_time = self._estimate_partial_charge(prev_i + 1, snap_route, Q_arrive_cs, t_arrive_cs, snap_load)

                    # 更新主状态（回退点之后继续）
                    Q = min(self.problem.soc_max, Q_arrive_cs + required_charge)
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load
                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_charging_time = snap_ct + charge_time
                    total_dist = snap_dist + to_cs_dist
                    num_charge += 1

                    # ---- 回退插桩这次充电后：重置“从上次充电起行驶时间”和Q_ref ----
                    t_since_charge = 0.0
                    Q_ref = Q

                    # ---- 累计焦虑：把 prev_node2 -> cs_id 这一段的 SUMR 增量加上 ----
                    if getattr(self, "ra_model", None) is not None:
                        # 段开始权重取回退快照的状态（段起点=prev_node2）
                        Ttr0 = snap_t_since
                        hour0 = (snap_t / 60.0) % 24.0

                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy

                        seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        total_ra = snap_ra_total + seg_ra
                    else:
                        total_ra = snap_ra_total

                    # 锁定直奔终点
                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True

                    # print(f"🔁 插入回退充电桩 CS[{cs_id}] 于 {prev_node2} 之后，恢复 Q = {Q:.2f}，继续主路径")

                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    # 防二次充电：下一次真正“到达这个cs节点”时跳过充电块一次
                    skip_charge_once = True
                    skip_charge_cs_id = cs_id

                    # ⭐ 插完就立刻检查“下一跳”是否可达；不可达则直接惩罚退出
                    if i < len(route):
                        next_node = route[i]
                        dist2 = self.problem.distance_matrix[route[i - 1], next_node]
                        energy2, _, _ = self.energy_model.calculate_one_node_energy_time(
                            dist2, departure_time, load
                        )
                        if Q + EPS < energy2:
                            # print(f"⚠️ 插了 CS[{cs_id}] 后依然到不了 {next_node}，直接返回惩罚")
                            return penalty_exit()
                    
                    break

                # 可能存在 两点之间距离太远，即使充电也不能到达的情况
                if not inserted:
                    # print("⚠️ 回退点用尽仍不可达 → 直接惩罚退出")
                    return penalty_exit()
                
                # 成功插入后，继续主 while
                continue
            
            # ---------- 可达：推进 prev→curr ----------
            if getattr(self, "ra_model", None) is not None:
                # 段起点权重（出发前）
                Ttr0 = t_since_charge
                hour0 = (departure_time / 60.0) % 24.0

                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed

                seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                total_ra += seg_ra

            # 2) 走这段路
            arrival_time = departure_time + travel_time
            service_time = self.problem.service_time[curr_node]
            departure_time = arrival_time + service_time
            Q -= energy_needed
            load -= self.problem.demand[curr_node]

            total_travel_time += travel_time
            total_service_time += service_time
            total_dist += dist
            # print(f"✅ 到达 {curr_node}：Q={Q:.2f}, t={departure_time:.2f}")

            # “从上次充电起的行驶时间”只加行驶，不加服务
            t_since_charge += travel_time

            # ---------- 到达CS：执行充电 ----------
            if curr_node in self.problem.css:
                # 若这是“回退插桩分支已算过充电”的那一次，到达时跳过一次
                if skip_charge_once and (curr_node == skip_charge_cs_id):
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    required_charge, charge_time = self._estimate_partial_charge(
                        i, route, Q, departure_time, load
                    )

                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.problem.soc_max, Q + required_charge)
                        departure_time += charge_time
                        total_charging_time += charge_time
                        num_charge += 1

                    # 充电后：焦虑参考点重置（实现“瞬时RA≈0”）
                    t_since_charge = 0.0
                    Q_ref = Q

                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True
            # print('---------------------------------', ra, total_ra)
            i += 1

        # print(f"\n🔚 解码完成，总充电次数: {num_charge}, 总距离: {total_dist:.2f}, 总行驶时间: {total_travel_time:.2f}, 剩余电量: {Q:.2f}")
        return (
            route, total_dist, total_travel_time,
            total_charging_time, total_service_time,
            num_charge, Q, departure_time, load, total_ra
        )


    def decode_spr_origin(self, route, load, demand=None, departure_time=0):
        """
        SPR decode (完整版本，按你当前代码风格最小侵入式修正)：

        - demand: 传入情景需求（不传则用 self.problem.demand）
        - plan_*：主路径的 travel/dist/service 与“正常到达CS后的充电”
        - recourse_*：
            1) 回退插桩产生的 repair 充电时间/次数
            2) 补货 detour-to-depot 的 travel/time/dist
            3) 补货 detour 中因电量不足产生的“去最近可达CS+充电” -> 记入 recourse
        - Case2: 到达客户发现 load < demand[curr] -> curr->depot->curr（detour）
        - Case3: 服务后 load==0 且后面还有客户 -> curr->depot->next（detour）
                **完成 detour 后会跳过下一轮的 curr->next 计划行驶计算（避免重复计费/状态错位）**

        注意：
        - detour 中的 RA 状态更新：这里选择“只累加行驶时间到 t_since_charge；若发生充电则重置 t_since_charge 与 Q_ref”
        这样 total_ra 不会明显失真。
        """
        if demand is None:
            demand = self.problem.demand

        Q = self.problem.soc_max
        EPS = self.EPS
        lock_to_depot = False

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])

        # ---------- RA 状态 ----------
        t_since_charge = 0.0
        Q_ref = Q
        plan_ra = 0.0
        recourse_ra = 0.0

        # 计划成本（第一阶段）
        plan_charging_time = 0.0
        plan_travel_time = 0.0
        plan_service_time = 0.0
        plan_dist = 0.0
        plan_num_charge = 0

        # 补救成本（第二阶段）
        recourse_charging_time = 0.0
        recourse_travel_time = 0.0
        recourse_dist = 0.0
        recourse_num_restock = 0
        recourse_num_charge = 0

        state_log = []
        i = 1

        BIG_PENALTY = 1e6
        WATCHDOG_LIMIT = 200
        watchdog = 0
        tried_pairs = set()

        skip_charge_once = False
        skip_charge_cs_id = None
        # ---------- detour（补货绕行）标记 ----------
        detour_active = False       # True 表示当前正在走补货绕行段（其 travel/charge 计入 recourse）
        detour_end_node = None      # detour 结束目标节点：Case2=回到同一客户；Case3=到达 next 客户
        detour_depot_id = None      # 当前 detour 使用的 depot（用于判断 detour 结束）
        detour_case = None          # "case2" 或 "case3"
        detour_curr_customer = None # Case2 时记录触发的 curr 客户 id

        # vehicle capacity
        CAP = getattr(self.problem, "vehicle_capacity", None)
        if CAP is None:
            CAP = getattr(self.problem, "cap", None)
        if CAP is None:
            raise AttributeError("problem 中找不到 vehicle_capacity/cap")

        def _nearest_depot(node_id: int) -> int:
            depots = list(self.problem.depots)
            if not depots:
                raise AttributeError("problem.depots 为空")
            dm = self.problem.distance_matrix
            best = depots[0]
            best_d = dm[node_id, best]
            for d in depots[1:]:
                dd = dm[node_id, d]
                if dd < best_d:
                    best_d = dd
                    best = d
            return best

        def _remaining_expected_demand(start_index: int) -> float:
            """从 route[start_index:] 中取所有客户节点的期望需求（用 self.problem.demand 当 μ）。"""
            mu = self.problem.demand
            total = 0.0
            for n in route[start_index:]:
                if n in self.problem.customers:
                    total += float(mu[n])
            return total

        def penalty_exit(reason: str):
            # print("=== PENALTY ===", reason)
            # print("i =", i)
            # print("prev_node =", prev_node)
            # print("curr_node =", curr_node)
            # print("Q =", Q)
            # print("energy_needed =", energy_needed if "energy_needed" in locals() else None)
            # print("departure_time =", departure_time)
            # print("load =", load)
            # print("route[i-2:i+3] =", route[max(0, i-2): min(len(route), i+3)])
            # print("state_log_len =", len(state_log))
            # print("tried_pairs_len =", len(tried_pairs))

            return (
                route,
                BIG_PENALTY, BIG_PENALTY,
                BIG_PENALTY, BIG_PENALTY,
                999,
                0.0,
                departure_time,
                load,
                BIG_PENALTY, BIG_PENALTY, BIG_PENALTY,
                BIG_PENALTY, BIG_PENALTY, BIG_PENALTY,
                999, 999
            )


        # print('start_route', route)
        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                return penalty_exit("WATCHDOG")

            prev_node = route[i - 1]
            curr_node = route[i]
            # print(prev_node, curr_node, 'float(demand[curr_node])', float(demand[curr_node]), load)

            # ---------- detour 结束判定 ----------
            # Case2：curr->depot->curr，回到 curr 且上一节点是 detour_depot_id 时结束
            # Case3：curr->depot->next，抵达 next 且上一节点是 detour_depot_id 时结束
            if detour_active and (detour_end_node is not None) and (curr_node == detour_end_node) and (prev_node == detour_depot_id):
                detour_active = False
                detour_end_node = None
                detour_depot_id = None
                detour_case = None
                detour_curr_customer = None


            # === RA & mode ===
            if getattr(self, "ra_model", None) is not None:
                Qtr = max(0.0, Q_ref - Q)
                Ttr = t_since_charge
                hour_of_day = (departure_time / 60.0) % 24.0
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                ra = 0.0
                mode = "safe_mode"

            # ---------- risk_mode：强制去最近可达 CS ----------
            if (not lock_to_depot) and (mode == "risk_mode"):
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    if curr_node != cs_id:
                        route.insert(i, cs_id)
                        continue

            # ---------- opportunity_mode：只插不算 ----------
            if (not lock_to_depot) and (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                already_before = (route[i - 1] == cs_id) if i - 1 >= 0 else False
                already_after = (i + 1 < len(route) and route[i + 1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i + 1] if i + 1 < len(route) else None
                    d_prev_cs = self.problem.distance_matrix[prev_node, cs_id]
                    d_cs_curr = self.problem.distance_matrix[cs_id, curr_node]
                    d_prev_curr = self.problem.distance_matrix[prev_node, curr_node]
                    d_curr_cs = self.problem.distance_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.problem.distance_matrix[curr_node, next_node]
                        d_cs_next = self.problem.distance_matrix[cs_id, next_node]
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost = d_prev_curr + d_curr_cs

                    if insert_before_cost + EPS < insert_after_cost:
                        route.insert(i, cs_id)
                        inserted_now = True
                    else:
                        route.insert(i + 1, cs_id)
                        inserted_now = True

                if inserted_now:
                    continue
            
            # 记录快照（用于回退）
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "Q": Q,
                "Q_ref": Q_ref,
                "t_since_charge": t_since_charge,
                "departure_time": departure_time,
                "load": load,

                "plan_travel_time": plan_travel_time,
                "plan_service_time": plan_service_time,
                "plan_charging_time": plan_charging_time,
                "plan_dist": plan_dist,
                "plan_num_charge": plan_num_charge,
                "plan_ra": plan_ra,
                "recourse_ra": recourse_ra,
                "lock_to_depot": lock_to_depot,

                "recourse_charging_time": recourse_charging_time,
                "recourse_travel_time": recourse_travel_time,
                "recourse_dist": recourse_dist,
                "recourse_num_restock": recourse_num_restock,
                "recourse_num_charge": recourse_num_charge,

                "detour_active": detour_active,
            })


            # ---------- 常规前进 prev → curr ----------
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(dist, departure_time, load)

            # 不可达：回退插桩（repair）
            if Q + EPS < energy_needed:
                inserted = False

                while state_log:
                    snap = state_log.pop()
                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]

                    snap_tt = snap["plan_travel_time"]
                    snap_st = snap["plan_service_time"]
                    snap_ct = snap["plan_charging_time"]
                    snap_dist = snap["plan_dist"]
                    snap_nchg = snap["plan_num_charge"]
                    snap_plan_ra = snap["plan_ra"]
                    snap_recourse_ra = snap["recourse_ra"]
                    snap_lock = snap["lock_to_depot"]

                    snap_rct = snap["recourse_charging_time"]
                    snap_rtt = snap["recourse_travel_time"]
                    snap_rdist = snap["recourse_dist"]
                    snap_rrest = snap["recourse_num_restock"]
                    snap_rnchg = snap["recourse_num_charge"]

                    snap_detour_active = snap["detour_active"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result

                    if cs_id == prev_node2:
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        continue
                    if (prev_i, cs_id) in tried_pairs:
                        return penalty_exit("TRIED_PAIRS")
                    tried_pairs.add((prev_i, cs_id))

                    # 在快照路由中插入 CS
                    snap_route.insert(prev_i + 1, cs_id)

                    # 到达 CS
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy

                    # fully充电
                    required_charge = max(0.0, self.problem.soc_max - Q_arrive_cs)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)
                    Q = self.problem.soc_max

                    # # 估算部分充电
                    # required_charge, charge_time = self._estimate_partial_charge(prev_i + 1, snap_route, Q_arrive_cs, t_arrive_cs, snap_load)

                    departure_time = t_arrive_cs + charge_time
                    load = snap_load

                    # -------- repair 成本归属：由快照时是否在 detour 中决定 --------
                    if not snap_detour_active:
                        # 情况 1：主路径本来就需要 repair -> 记到第一阶段
                        plan_travel_time = snap_tt + to_cs_time
                        plan_service_time = snap_st
                        plan_dist = snap_dist + to_cs_dist
                        plan_charging_time = snap_ct + charge_time
                        plan_num_charge = snap_nchg + (1 if (charge_time > 0 or required_charge > 0) else 0)

                        recourse_travel_time = snap_rtt
                        recourse_dist = snap_rdist
                        recourse_charging_time = snap_rct
                        recourse_num_restock = snap_rrest
                        recourse_num_charge = snap_rnchg
                    else:
                        # 情况 2：因为补货 detour 才产生 repair -> 记到第二阶段
                        plan_travel_time = snap_tt
                        plan_service_time = snap_st
                        plan_dist = snap_dist
                        plan_charging_time = snap_ct
                        plan_num_charge = snap_nchg

                        recourse_travel_time = snap_rtt + to_cs_time
                        recourse_dist = snap_rdist + to_cs_dist
                        recourse_charging_time = snap_rct + charge_time
                        recourse_num_restock = snap_rrest
                        recourse_num_charge = snap_rnchg + (1 if (charge_time > 0 or required_charge > 0) else 0)

                    # -------- RA 重置 --------
                    t_since_charge = 0.0
                    Q_ref = Q

                    if getattr(self, "ra_model", None) is not None:
                        Ttr0 = snap_t_since
                        hour0 = (snap_t / 60.0) % 24.0
                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy
                        seg_ra = (
                            self.ra_model.SUMR(Qtr_after, Ttr0, hour0)
                            - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        )

                        if not snap_detour_active:
                            plan_ra = snap_plan_ra + seg_ra
                            recourse_ra = snap_recourse_ra
                        else:
                            plan_ra = snap_plan_ra
                            recourse_ra = snap_recourse_ra + seg_ra
                    else:
                        plan_ra = snap_plan_ra
                        recourse_ra = snap_recourse_ra

                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True
                    else:
                        lock_to_depot = snap_lock

                    # 用修复后的状态覆盖当前状态
                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    skip_charge_once = True
                    skip_charge_cs_id = cs_id

                    break

                if not inserted:
                    return penalty_exit("REPAIR_FAILED")

                continue

            # ---------- 可达：推进 prev→curr（计划行驶） ----------
            if getattr(self, "ra_model", None) is not None:
                Ttr0 = t_since_charge
                hour0 = (departure_time / 60.0) % 24.0
                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed
                seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                
                if detour_active:
                    recourse_ra += seg_ra
                else:
                    plan_ra += seg_ra

            # 先走 prev->curr
            arrival_time = departure_time + travel_time
            Q -= energy_needed
            if Q + EPS < 0:
                return penalty_exit("NEGATIVE_Q")

            # detour 段行驶计入 recourse，否则计入 plan
            if detour_active:
                recourse_travel_time += travel_time
                recourse_dist += dist
            else:
                plan_travel_time += travel_time
                plan_dist += dist

            t_since_charge += travel_time

            # ---------- 到达 depot：补货（只在 detour_active 时执行） ----------
            if detour_active and (curr_node in self.problem.depots):

                if detour_case == "case2":
                    # Case2：已知触发客户 curr 的真实需求（场景 demand）
                    # 补货量 = curr真实需求 + 后续客户期望需求
                    curr_real = float(demand[detour_curr_customer]) if detour_curr_customer is not None else 0.0
                    tail_exp = _remaining_expected_demand(i + 1)   # 注意：这里用期望 demand（problem.demand）
                    target_load = curr_real + tail_exp

                else:
                    # Case3（或兜底）：未知后续真实需求，只用期望需求
                    target_load = _remaining_expected_demand(i + 1)

                load = min(CAP, max(0.0, target_load))
                recourse_num_restock += 1

                # depot service_time（如有）
                service_time_depot = self.problem.service_time[curr_node]
                departure_time = arrival_time + service_time_depot
                plan_service_time += service_time_depot  # depot=0则无影响

                i += 1
                continue

            # ----------------------------
            # Case 2: 到达客户发现 load 不足 -> curr->depot->curr（插入 route）
            # ----------------------------
            if (curr_node in self.problem.customers) and (load + EPS < float(demand[curr_node])):

                depot_id = _nearest_depot(curr_node)

                # 在 route 中插入：curr -> depot -> curr
                # 注意：我们当前已经到达 curr_node，但第一次到达不服务，先去补货再回来服务
                route.insert(i + 1, depot_id)
                route.insert(i + 2, curr_node)

                # 打开 detour 标记：直到 “从 depot 回到 curr” 的那次到达才结束
                detour_active = True
                detour_end_node = curr_node
                detour_depot_id = depot_id

                detour_case = "case2"
                detour_curr_customer = curr_node

                # 第一次到达 curr_node 不服务，直接从 curr->depot（所以把 departure_time 更新为 arrival_time）
                departure_time = arrival_time

                i += 1
                continue


            # 服务（一次）
            service_time = self.problem.service_time[curr_node]
            departure_time = arrival_time + service_time
            plan_service_time += service_time

            # 扣减载重（按情景需求）
            load -= float(demand[curr_node])
            # ✅ 浮点/需求误差防护：避免 load 变成很小负数影响 Case3/后续判断
            if load < 0:
                load = 0.0

            # ----------------------------
            # Case 3: 服务后 load==0 且下一个是客户 -> curr->depot->next（插入 route）
            # ----------------------------
            if (curr_node in self.problem.customers) and (abs(load) <= EPS) and (i + 1 < len(route)):
                next_node = route[i + 1]
                if next_node in self.problem.customers:
                    depot_id = _nearest_depot(curr_node)

                    # 插入：curr -> depot -> next
                    route.insert(i + 1, depot_id)

                    detour_active = True
                    detour_end_node = next_node
                    detour_depot_id = depot_id

                    detour_case = "case3"
                    detour_curr_customer = None

                    # 下一轮将从 curr->depot 开始走（计入 recourse），然后 depot 处补货，再去 next
                    i += 1
                    continue

            # ---------- 到达CS：计划充电 ----------
            if curr_node in self.charging_nodes:
                if skip_charge_once and (curr_node == skip_charge_cs_id):
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    # # 部分充电
                    # required_charge, charge_time = self._estimate_partial_charge(i, route, Q, departure_time, load)
                    
                    # fully充电
                    # 用当前电量 Q（不是 Q_arrive_cs）
                    required_charge = max(0.0, self.problem.soc_max - Q)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)

                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.problem.soc_max, Q + required_charge)
                        # Q = self.problem.soc_max
                        departure_time += charge_time

                        # detour 段充电计入 recourse，否则计入 plan
                        if detour_active:
                            recourse_charging_time += charge_time
                            recourse_num_charge += 1
                        else:
                            plan_charging_time += charge_time
                            plan_num_charge += 1

                    t_since_charge = 0.0
                    Q_ref = Q

                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True

            i += 1
        # print('final_route', route)
        total_ra = plan_ra + recourse_ra
        return (
            route,
            plan_dist, plan_travel_time,
            plan_charging_time, plan_service_time,
            plan_num_charge,
            Q, departure_time, load, 
            total_ra, plan_ra, recourse_ra,
            recourse_dist, recourse_travel_time, recourse_charging_time,
            recourse_num_restock, recourse_num_charge
        )


    def decode_spr(self, route, load, demand=None, departure_time=0):
        """
        SPR decode（精简合并版）

        功能：
        - demand: 传入情景需求（不传则用 self.problem.demand）
        - 保留 EV 执行解码：
            1) 电量推进
            2) 焦虑驱动插桩
            3) 不可达时回退插入可达充电站
            4) 若修复后仍不可达，则返回惩罚解
        - 保留随机需求补货 detour：
            Case2: 到达客户发现 load < demand[curr] -> curr->depot->curr
            Case3: 服务后 load==0 且后面还有客户 -> curr->depot->next
        - 不再区分 plan / recourse，统一累计总量
        """

        if demand is None:
            demand = self.problem.demand

        route = route[:]  # 避免直接改外部传入对象

        Q = self.problem.soc_max
        EPS = self.EPS
        lock_to_depot = False

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])

        # ---------- RA 状态 ----------
        t_since_charge = 0.0
        Q_ref = Q
        total_ra = 0.0

        # ---------- 总成本 ----------
        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0
        num_restock = 0

        state_log = []
        i = 1

        BIG_PENALTY = 1e6
        WATCHDOG_LIMIT = 200
        watchdog = 0
        tried_pairs = set()

        skip_charge_once = False
        skip_charge_cs_id = None

        # ---------- detour（补货绕行）标记 ----------
        detour_active = False       # True 表示当前正在走补货绕行段（其 travel/charge 计入 recourse）
        detour_end_node = None      # detour 结束目标节点：Case2=回到同一客户；Case3=到达 next 客户
        detour_depot_id = None      # 当前 detour 使用的 depot（用于判断 detour 结束）
        detour_case = None          # "case2" 或 "case3"
        detour_curr_customer = None # Case2 时记录触发的 curr 客户 id

        CAP = getattr(self.problem, "vehicle_capacity", None)
        if CAP is None:
            CAP = getattr(self.problem, "cap", None)
        if CAP is None:
            raise AttributeError("problem 中找不到 vehicle_capacity/cap")

        def _nearest_depot(node_id: int) -> int:
            depots = list(self.problem.depots)
            if not depots:
                raise AttributeError("problem.depots 为空")
            dm = self.problem.distance_matrix
            best = depots[0]
            best_d = dm[node_id, best]
            for d in depots[1:]:
                dd = dm[node_id, d]
                if dd < best_d:
                    best_d = dd
                    best = d
            return best
    
        def _remaining_expected_demand(start_index: int) -> float:
            """从 route[start_index:] 中取后续客户的期望需求（用 self.problem.demand 当 μ）"""
            mu = self.problem.demand
            total = 0.0
            for n in route[start_index:]:
                if n in self.problem.customers:
                    total += float(mu[n])
            return total

        def penalty_exit(reason: str):
            return (
                route,
                BIG_PENALTY,  # total_dist
                BIG_PENALTY,  # total_travel_time
                BIG_PENALTY,  # total_charging_time
                BIG_PENALTY,  # total_service_time
                999,          # num_charge
                0.0,          # Q
                departure_time,
                load,
                BIG_PENALTY,  # total_ra
                999,          # num_restock
            )


        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                return penalty_exit("WATCHDOG")

            prev_node = route[i - 1]
            curr_node = route[i]

            # ---------- detour 结束判定 ----------
            if (
                detour_active
                and (detour_end_node is not None)
                and (curr_node == detour_end_node)
                and (prev_node == detour_depot_id)
            ):
                detour_active = False
                detour_end_node = None
                detour_depot_id = None
                detour_case = None
                detour_curr_customer = None

            # ---------- RA & mode ----------
            if getattr(self, "ra_model", None) is not None:
                Qtr = max(0.0, Q_ref - Q)
                Ttr = t_since_charge
                hour_of_day = (departure_time / 60.0) % 24.0
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                mode = "safe_mode"

            # ---------- risk_mode：强制去最近可达 CS ----------
            if (not lock_to_depot) and (mode == "risk_mode"):
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    if curr_node != cs_id:
                        route.insert(i, cs_id)
                        continue

            # ---------- opportunity_mode：只插不算 ----------
            if (not lock_to_depot) and (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                already_before = (route[i - 1] == cs_id) if i - 1 >= 0 else False
                already_after = (i + 1 < len(route) and route[i + 1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i + 1] if i + 1 < len(route) else None

                    d_prev_cs = self.problem.distance_matrix[prev_node, cs_id]
                    d_cs_curr = self.problem.distance_matrix[cs_id, curr_node]
                    d_prev_curr = self.problem.distance_matrix[prev_node, curr_node]
                    d_curr_cs = self.problem.distance_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.problem.distance_matrix[curr_node, next_node]
                        d_cs_next = self.problem.distance_matrix[cs_id, next_node]
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost = d_prev_curr + d_curr_cs

                    if insert_before_cost + EPS < insert_after_cost:
                        route.insert(i, cs_id)
                    else:
                        route.insert(i + 1, cs_id)
                    inserted_now = True

                if inserted_now:
                    continue

            # ---------- 记录快照（用于回退修复） ----------
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "Q": Q,
                "Q_ref": Q_ref,
                "t_since_charge": t_since_charge,
                "departure_time": departure_time,
                "load": load,
                "total_travel_time": total_travel_time,
                "total_service_time": total_service_time,
                "total_charging_time": total_charging_time,
                "total_dist": total_dist,
                "total_ra": total_ra,
                "num_charge": num_charge,
                "num_restock": num_restock,
                "lock_to_depot": lock_to_depot,
            })

            # ---------- 常规前进 prev -> curr ----------
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(
                dist, departure_time, load
            )

            # ---------- 不可达：回退插桩修复 ----------
            if Q + EPS < energy_needed:
                inserted = False

                while state_log:
                    snap = state_log.pop()
                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]

                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]
                    snap_ra = snap["total_ra"]
                    snap_nchg = snap["num_charge"]
                    snap_nrest = snap["num_restock"]
                    snap_lock = snap["lock_to_depot"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result

                    if cs_id == prev_node2:
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        continue
                    if (prev_i, cs_id) in tried_pairs:
                        return penalty_exit("TRIED_PAIRS")
                    tried_pairs.add((prev_i, cs_id))

                    # 插入 CS
                    snap_route.insert(prev_i + 1, cs_id)

                    # 到达 CS
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy

                    # full charging
                    required_charge = max(0.0, self.problem.soc_max - Q_arrive_cs)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)

                    Q = self.problem.soc_max
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load

                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_dist = snap_dist + to_cs_dist
                    total_charging_time = snap_ct + charge_time
                    num_charge = snap_nchg + (1 if (required_charge > 0 or charge_time > 0) else 0)
                    num_restock = snap_nrest

                    # RA：这段 prev_node2 -> cs 的增量
                    t_since_charge = 0.0
                    Q_ref = Q
                    if getattr(self, "ra_model", None) is not None:
                        Ttr0 = snap_t_since
                        hour0 = (snap_t / 60.0) % 24.0
                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy
                        seg_ra = (
                            self.ra_model.SUMR(Qtr_after, Ttr0, hour0)
                            - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        )
                        total_ra = snap_ra + seg_ra
                    else:
                        total_ra = snap_ra

                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True
                    else:
                        lock_to_depot = snap_lock

                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    # 防止到达这个 CS 后再次重复充电
                    skip_charge_once = True
                    skip_charge_cs_id = cs_id
                    break

                if not inserted:
                    return penalty_exit("REPAIR_FAILED")

                continue

            # ---------- 可达：累加该段 RA ----------
            if getattr(self, "ra_model", None) is not None:
                Ttr0 = t_since_charge
                hour0 = (departure_time / 60.0) % 24.0
                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed
                seg_ra = (
                    self.ra_model.SUMR(Qtr_after, Ttr0, hour0)
                    - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                )
                total_ra += seg_ra

            # ---------- 推进 prev -> curr ----------
            arrival_time = departure_time + travel_time
            Q -= energy_needed
            if Q + EPS < 0:
                return penalty_exit("NEGATIVE_Q")

            total_travel_time += travel_time
            total_dist += dist
            t_since_charge += travel_time

            # ---------- 到达 depot：补货（仅在 detour 中） ----------
            if detour_active and (curr_node in self.problem.depots):
                if detour_case == "case2":
                    curr_real = float(demand[detour_curr_customer]) if detour_curr_customer is not None else 0.0
                    tail_exp = _remaining_expected_demand(i + 1)
                    target_load = curr_real + tail_exp
                else:
                    target_load = _remaining_expected_demand(i + 1)

                load = min(CAP, max(0.0, target_load))
                num_restock += 1

                service_time_depot = self.problem.service_time[curr_node]
                departure_time = arrival_time + service_time_depot
                total_service_time += service_time_depot

                i += 1
                continue

            # ---------- Case2：到达客户发现 load 不足 ----------
            if (curr_node in self.problem.customers) and (load + EPS < float(demand[curr_node])):
                depot_id = _nearest_depot(curr_node)

                # curr -> depot -> curr
                route.insert(i + 1, depot_id)
                route.insert(i + 2, curr_node)

                detour_active = True
                detour_end_node = curr_node
                detour_depot_id = depot_id
                detour_case = "case2"
                detour_curr_customer = curr_node

                # 第一次到 curr 不服务，立刻转去 depot
                departure_time = arrival_time
                i += 1
                continue

            # ---------- 服务客户 / 节点 ----------
            service_time = self.problem.service_time[curr_node]
            departure_time = arrival_time + service_time
            total_service_time += service_time

            load -= float(demand[curr_node])
            if load < 0:
                load = 0.0

            # ---------- Case3：服务后空载且后面还有客户 ----------
            if (curr_node in self.problem.customers) and (abs(load) <= EPS) and (i + 1 < len(route)):
                next_node = route[i + 1]
                if next_node in self.problem.customers:
                    depot_id = _nearest_depot(curr_node)

                    # curr -> depot -> next
                    route.insert(i + 1, depot_id)

                    detour_active = True
                    detour_end_node = next_node
                    detour_depot_id = depot_id
                    detour_case = "case3"
                    detour_curr_customer = None

                    i += 1
                    continue

            # ---------- 到达 CS：充电 ----------
            if curr_node in self.charging_nodes:
                if skip_charge_once and (curr_node == skip_charge_cs_id):
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    required_charge = max(0.0, self.problem.soc_max - Q)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)

                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.problem.soc_max, Q + required_charge)
                        departure_time += charge_time
                        total_charging_time += charge_time
                        num_charge += 1

                    t_since_charge = 0.0
                    Q_ref = Q

                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True

            i += 1

        return (
            route,
            total_dist,
            total_travel_time,
            total_charging_time,
            total_service_time,
            num_charge,
            Q,
            departure_time,
            load,
            total_ra,
            num_restock,
        )


    def decode_spr_detail1(self, route, load, demand=None, departure_time=0):
        """
        SPR decode（带简化统计版）

        功能：
        - demand: 传入情景需求（不传则用 self.problem.demand）
        - 保留 EV 执行解码：
            1) 电量推进
            2) 焦虑驱动插桩
            3) 不可达时回退插入可达充电站
            4) 若修复后仍不可达，则返回惩罚解
        - 保留随机需求补货 detour：
            Case2: 到达客户发现 load < demand[curr] -> curr->depot->curr
            Case3: 服务后 load==0 且后面还有客户 -> curr->depot->next
        - 不再区分 plan / recourse，统一累计总量
        """

        if demand is None:
            demand = self.problem.demand

        route = route[:]  # 避免直接改外部传入对象

        Q = self.problem.soc_max
        EPS = self.EPS
        lock_to_depot = False

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])

        # ---------- RA 状态 ----------
        t_since_charge = 0.0
        Q_ref = Q
        total_ra = 0.0

        # ---------- 总成本 ----------
        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0
        num_restock = 0

        # ---------- 统计 ----------
        stats = {
            # A) 充电事件
            "num_charge_total": 0,
            "num_charge_opportunistic": 0,
            "num_charge_risk": 0,
            "num_charge_repair": 0,
            "num_charge_natural": 0,

            "charging_time_total": 0.0,
            "charging_time_opportunistic": 0.0,
            "charging_time_risk": 0.0,
            "charging_time_repair": 0.0,
            "charging_time_natural": 0.0,

            "charged_energy_total": 0.0,
            "charged_energy_opportunistic": 0.0,
            "charged_energy_risk": 0.0,
            "charged_energy_repair": 0.0,
            "charged_energy_natural": 0.0,

            # # B) repair 发生时所处 mode
            # "num_charge_repair_when_safe": 0,
            # "num_charge_repair_when_opportunity": 0,
            # "num_charge_repair_when_risk": 0,

            # "charging_time_repair_when_safe": 0.0,
            # "charging_time_repair_when_opportunity": 0.0,
            # "charging_time_repair_when_risk": 0.0,

            # "charged_energy_repair_when_safe": 0.0,
            # "charged_energy_repair_when_opportunity": 0.0,
            # "charged_energy_repair_when_risk": 0.0,

            # # C) 插桩次数
            # "num_insert_opportunistic": 0,
            # "num_insert_risk": 0,
            # "num_insert_repair": 0,

            # # D) 回溯统计
            # "num_backtrack_trigger": 0,
            # "num_backtrack_success": 0,
            # "backtrack_depth_list": [],
            # "max_backtrack_depth": 0,

            # "num_backtrack_when_safe": 0,
            # "num_backtrack_when_opportunity": 0,
            # "num_backtrack_when_risk": 0,

            # # E) 其他
            # "num_restock": 0,
            # "min_soc": 1.0,
            # "status": "ok",
            # "penalty_exit": 0,
        }

        # 记录“这个 CS 是因为什么被插进 route 的”
        opportunistic_cs_set = set()
        risk_cs_set = set()

        # 记录最近一次 backtrack 触发时的 mode
        trigger_mode = None

        state_log = []
        i = 1

        BIG_PENALTY = 1e6
        WATCHDOG_LIMIT = 200
        watchdog = 0
        tried_pairs = set()

        # 防止“回退插桩分支已充电”后，到达同一CS又重复充电/计数
        skip_charge_once = False
        skip_charge_cs_id = None

        # ---------- detour（补货绕行）标记 ----------
        detour_active = False
        detour_end_node = None
        detour_depot_id = None
        detour_case = None
        detour_curr_customer = None

        CAP = getattr(self.problem, "vehicle_capacity", None)
        if CAP is None:
            CAP = getattr(self.problem, "cap", None)
        if CAP is None:
            raise AttributeError("problem 中找不到 vehicle_capacity/cap")

        def _clone_stats(s):
            out = dict(s)
            # out["backtrack_depth_list"] = list(s.get("backtrack_depth_list", []))
            return out

        def _nearest_depot(node_id: int) -> int:
            depots = list(self.problem.depots)
            if not depots:
                raise AttributeError("problem.depots 为空")
            dm = self.problem.distance_matrix
            best = depots[0]
            best_d = dm[node_id, best]
            for d in depots[1:]:
                dd = dm[node_id, d]
                if dd < best_d:
                    best_d = dd
                    best = d
            return best

        def _remaining_expected_demand(start_index: int) -> float:
            """从 route[start_index:] 中取后续客户的期望需求（用 self.problem.demand 当 μ）"""
            mu = self.problem.demand
            total = 0.0
            for n in route[start_index:]:
                if n in self.problem.customers:
                    total += float(mu[n])
            return total

        def penalty_exit(reason: str):
            # stats["status"] = reason
            # stats["penalty_exit"] = 1
            # stats["num_restock"] = int(num_restock)
            self.last_decode_stats = stats

            return (
                route,
                BIG_PENALTY,  # total_dist
                BIG_PENALTY,  # total_travel_time
                BIG_PENALTY,  # total_charging_time
                BIG_PENALTY,  # total_service_time
                999,          # num_charge
                0.0,          # Q
                departure_time,
                load,
                BIG_PENALTY,  # total_ra
                999,          # num_restock
                stats,
            )

        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                return penalty_exit("WATCHDOG")

            prev_node = route[i - 1]
            curr_node = route[i]

            # ---------- detour 结束判定 ----------
            if (
                detour_active
                and (detour_end_node is not None)
                and (curr_node == detour_end_node)
                and (prev_node == detour_depot_id)
            ):
                detour_active = False
                detour_end_node = None
                detour_depot_id = None
                detour_case = None
                detour_curr_customer = None

            # ---------- RA & mode ----------
            if getattr(self, "ra_model", None) is not None:
                Qtr = max(0.0, Q_ref - Q)
                Ttr = t_since_charge
                hour_of_day = (departure_time / 60.0) % 24.0
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                mode = "safe_mode"

            # ---------- risk_mode：强制去最近可达 CS ----------
            if (not lock_to_depot) and (mode == "risk_mode"):
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    if curr_node != cs_id:
                        route.insert(i, cs_id)

                        # stats["num_insert_risk"] += 1
                        risk_cs_set.add(cs_id)

                        continue

            # ---------- opportunity_mode：顺路插桩 ----------
            if (not lock_to_depot) and (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                already_before = (route[i - 1] == cs_id) if i - 1 >= 0 else False
                already_after = (i + 1 < len(route) and route[i + 1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i + 1] if i + 1 < len(route) else None

                    d_prev_cs = self.problem.distance_matrix[prev_node, cs_id]
                    d_cs_curr = self.problem.distance_matrix[cs_id, curr_node]
                    d_prev_curr = self.problem.distance_matrix[prev_node, curr_node]
                    d_curr_cs = self.problem.distance_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.problem.distance_matrix[curr_node, next_node]
                        d_cs_next = self.problem.distance_matrix[cs_id, next_node]
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost = d_prev_curr + d_curr_cs

                    if insert_before_cost + EPS < insert_after_cost:
                        route.insert(i, cs_id)
                    else:
                        route.insert(i + 1, cs_id)
                    inserted_now = True

                if inserted_now:
                    # stats["num_insert_opportunistic"] += 1
                    opportunistic_cs_set.add(cs_id)
                    continue

            # ---------- 记录快照（用于回退修复） ----------
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "Q": Q,
                "Q_ref": Q_ref,
                "t_since_charge": t_since_charge,
                "departure_time": departure_time,
                "load": load,
                "total_travel_time": total_travel_time,
                "total_service_time": total_service_time,
                "total_charging_time": total_charging_time,
                "total_dist": total_dist,
                "total_ra": total_ra,
                "num_charge": num_charge,
                "num_restock": num_restock,
                "lock_to_depot": lock_to_depot,

                "stats": _clone_stats(stats),
                "opportunistic_cs_set": set(opportunistic_cs_set),
                "risk_cs_set": set(risk_cs_set),

                "detour_active": detour_active,
                "detour_end_node": detour_end_node,
                "detour_depot_id": detour_depot_id,
                "detour_case": detour_case,
                "detour_curr_customer": detour_curr_customer,

                "skip_charge_once": skip_charge_once,
                "skip_charge_cs_id": skip_charge_cs_id,
            })

            # ---------- 常规前进 prev -> curr ----------
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(
                dist, departure_time, load
            )

            # ---------- 不可达：回退插桩修复 ----------
            if Q + EPS < energy_needed:
                # stats["num_backtrack_trigger"] += 1
                trigger_mode = mode

                # if mode == "risk_mode":
                #     stats["num_backtrack_when_risk"] += 1
                # elif mode == "opportunity_mode":
                #     stats["num_backtrack_when_opportunity"] += 1
                # else:
                #     stats["num_backtrack_when_safe"] += 1

                inserted = False
                backtrack_depth = 0

                while state_log:
                    backtrack_depth += 1
                    snap = state_log.pop()

                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]

                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]
                    snap_ra = snap["total_ra"]
                    snap_nchg = snap["num_charge"]
                    snap_nrest = snap["num_restock"]
                    snap_lock = snap["lock_to_depot"]

                    snap_stats = snap["stats"]
                    snap_opp_set = snap["opportunistic_cs_set"]
                    snap_risk_set = snap["risk_cs_set"]

                    snap_detour_active = snap["detour_active"]
                    snap_detour_end_node = snap["detour_end_node"]
                    snap_detour_depot_id = snap["detour_depot_id"]
                    snap_detour_case = snap["detour_case"]
                    snap_detour_curr_customer = snap["detour_curr_customer"]

                    snap_skip_charge_once = snap["skip_charge_once"]
                    snap_skip_charge_cs_id = snap["skip_charge_cs_id"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result

                    if cs_id == prev_node2:
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        continue
                    if (prev_i, cs_id) in tried_pairs:
                        return penalty_exit("TRIED_PAIRS")
                    tried_pairs.add((prev_i, cs_id))

                    # 恢复到 snap 对应状态
                    stats = _clone_stats(snap_stats)
                    opportunistic_cs_set = set(snap_opp_set)
                    risk_cs_set = set(snap_risk_set)

                    detour_active = snap_detour_active
                    detour_end_node = snap_detour_end_node
                    detour_depot_id = snap_detour_depot_id
                    detour_case = snap_detour_case
                    detour_curr_customer = snap_detour_curr_customer

                    skip_charge_once = snap_skip_charge_once
                    skip_charge_cs_id = snap_skip_charge_cs_id

                    # 插入 repair CS
                    snap_route.insert(prev_i + 1, cs_id)
                    # stats["num_insert_repair"] += 1

                    # 到达 CS
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy

                    # stats["min_soc"] = min(stats["min_soc"], max(0.0, Q_arrive_cs / self.problem.soc_max))

                    # full charging
                    required_charge = max(0.0, self.problem.soc_max - Q_arrive_cs)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)

                    Q = self.problem.soc_max
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load

                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_dist = snap_dist + to_cs_dist
                    total_charging_time = snap_ct + charge_time

                    if required_charge > 0 or charge_time > 0:
                        stats["num_charge_total"] += 1
                        stats["num_charge_repair"] += 1

                        stats["charging_time_total"] += charge_time
                        stats["charging_time_repair"] += charge_time

                        stats["charged_energy_total"] += required_charge
                        stats["charged_energy_repair"] += required_charge

                        # if trigger_mode == "risk_mode":
                        #     stats["num_charge_repair_when_risk"] += 1
                        #     stats["charging_time_repair_when_risk"] += charge_time
                        #     stats["charged_energy_repair_when_risk"] += required_charge
                        # elif trigger_mode == "opportunity_mode":
                        #     stats["num_charge_repair_when_opportunity"] += 1
                        #     stats["charging_time_repair_when_opportunity"] += charge_time
                        #     stats["charged_energy_repair_when_opportunity"] += required_charge
                        # else:
                        #     stats["num_charge_repair_when_safe"] += 1
                        #     stats["charging_time_repair_when_safe"] += charge_time
                        #     stats["charged_energy_repair_when_safe"] += required_charge

                    num_charge = stats["num_charge_total"]
                    num_restock = snap_nrest

                    # RA：prev_node2 -> cs 的增量
                    t_since_charge = 0.0
                    Q_ref = Q
                    if getattr(self, "ra_model", None) is not None:
                        Ttr0 = snap_t_since
                        hour0 = (snap_t / 60.0) % 24.0
                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy
                        seg_ra = (
                            self.ra_model.SUMR(Qtr_after, Ttr0, hour0)
                            - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        )
                        total_ra = snap_ra + seg_ra
                    else:
                        total_ra = snap_ra

                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True
                    else:
                        lock_to_depot = snap_lock

                    route = snap_route
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    # stats["num_backtrack_success"] += 1
                    # stats["backtrack_depth_list"].append(backtrack_depth)
                    # stats["max_backtrack_depth"] = max(stats["max_backtrack_depth"], backtrack_depth)

                    # 防止到达这个 CS 后再次重复充电
                    skip_charge_once = True
                    skip_charge_cs_id = cs_id

                    # 修复后立即检查下一跳可达性
                    if i < len(route):
                        next_node = route[i]
                        dist2 = self.problem.distance_matrix[route[i - 1], next_node]
                        energy2, _, _ = self.energy_model.calculate_one_node_energy_time(
                            dist2, departure_time, load
                        )
                        if Q + EPS < energy2:
                            return penalty_exit("REPAIR_STILL_INFEASIBLE")

                    break

                if not inserted:
                    return penalty_exit("REPAIR_FAILED")

                continue

            # ---------- 可达：累加该段 RA ----------
            if getattr(self, "ra_model", None) is not None:
                Ttr0 = t_since_charge
                hour0 = (departure_time / 60.0) % 24.0
                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed
                seg_ra = (
                    self.ra_model.SUMR(Qtr_after, Ttr0, hour0)
                    - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                )
                total_ra += seg_ra

            # ---------- 推进 prev -> curr ----------
            arrival_time = departure_time + travel_time
            Q -= energy_needed
            if Q + EPS < 0:
                return penalty_exit("NEGATIVE_Q")

            total_travel_time += travel_time
            total_dist += dist
            t_since_charge += travel_time

            # stats["min_soc"] = min(stats["min_soc"], max(0.0, Q / self.problem.soc_max))

            # ---------- 到达 depot：补货（仅在 detour 中） ----------
            if detour_active and (curr_node in self.problem.depots):
                if detour_case == "case2":
                    curr_real = float(demand[detour_curr_customer]) if detour_curr_customer is not None else 0.0
                    tail_exp = _remaining_expected_demand(i + 1)
                    target_load = curr_real + tail_exp
                else:
                    target_load = _remaining_expected_demand(i + 1)

                load = min(CAP, max(0.0, target_load))
                num_restock += 1
                # stats["num_restock"] = int(num_restock)

                service_time_depot = self.problem.service_time[curr_node]
                departure_time = arrival_time + service_time_depot
                total_service_time += service_time_depot

                i += 1
                continue

            # ---------- Case2：到达客户发现 load 不足 ----------
            if (curr_node in self.problem.customers) and (load + EPS < float(demand[curr_node])):
                depot_id = _nearest_depot(curr_node)

                # curr -> depot -> curr
                route.insert(i + 1, depot_id)
                route.insert(i + 2, curr_node)

                detour_active = True
                detour_end_node = curr_node
                detour_depot_id = depot_id
                detour_case = "case2"
                detour_curr_customer = curr_node

                # 第一次到 curr 不服务，立刻转去 depot
                departure_time = arrival_time
                i += 1
                continue

            # ---------- 服务客户 / 节点 ----------
            service_time = self.problem.service_time[curr_node]
            departure_time = arrival_time + service_time
            total_service_time += service_time

            load -= float(demand[curr_node])
            if load < 0:
                load = 0.0

            # ---------- Case3：服务后空载且后面还有客户 ----------
            if (curr_node in self.problem.customers) and (abs(load) <= EPS) and (i + 1 < len(route)):
                next_node = route[i + 1]
                if next_node in self.problem.customers:
                    depot_id = _nearest_depot(curr_node)

                    # curr -> depot -> next
                    route.insert(i + 1, depot_id)

                    detour_active = True
                    detour_end_node = next_node
                    detour_depot_id = depot_id
                    detour_case = "case3"
                    detour_curr_customer = None

                    i += 1
                    continue

            # ---------- 到达 CS：充电 ----------
            if curr_node in self.charging_nodes:
                if skip_charge_once and (curr_node == skip_charge_cs_id):
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    required_charge = max(0.0, self.problem.soc_max - Q)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)

                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.problem.soc_max, Q + required_charge)
                        departure_time += charge_time
                        total_charging_time += charge_time

                        stats["num_charge_total"] += 1
                        stats["charging_time_total"] += charge_time
                        stats["charged_energy_total"] += required_charge

                        if curr_node in opportunistic_cs_set:
                            stats["num_charge_opportunistic"] += 1
                            stats["charging_time_opportunistic"] += charge_time
                            stats["charged_energy_opportunistic"] += required_charge
                        elif curr_node in risk_cs_set:
                            stats["num_charge_risk"] += 1
                            stats["charging_time_risk"] += charge_time
                            stats["charged_energy_risk"] += required_charge
                        else:
                            stats["num_charge_natural"] += 1
                            stats["charging_time_natural"] += charge_time
                            stats["charged_energy_natural"] += required_charge

                        num_charge = stats["num_charge_total"]

                    t_since_charge = 0.0
                    Q_ref = Q

                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True

            i += 1

        # stats["num_restock"] = int(num_restock)

        # if stats["num_charge_total"] != (
        #     stats["num_charge_opportunistic"]
        #     + stats["num_charge_risk"]
        #     + stats["num_charge_repair"]
        #     + stats["num_charge_natural"]
        # ):
        #     stats["status"] = "charge_count_mismatch"

        self.last_decode_stats = stats
        num_charge = int(stats["num_charge_total"])

        return (
            route,
            total_dist,
            total_travel_time,
            total_charging_time,
            total_service_time,
            num_charge,
            Q,
            departure_time,
            load,
            total_ra,
            num_restock,
            stats,
        )
   


    def decode_spr_detail(self, route, load, demand=None, departure_time=0):
        """
        SPR decode（带简化统计版，单次访问精确分类）

        功能：
        - demand: 传入情景需求（不传则用 self.problem.demand）
        - 保留 EV 执行解码：
            1) 电量推进
            2) 焦虑驱动插桩
            3) 不可达时回退插入可达充电站
            4) 若修复后仍不可达，则返回惩罚解
        - 保留随机需求补货 detour：
            Case2: 到达客户发现 load < demand[curr] -> curr->depot->curr
            Case3: 服务后 load==0 且后面还有客户 -> curr->depot->next
        - 不再区分 plan / recourse，统一累计总量

        充电分类规则（按“单次访问”）：
        - opportunity_mode 下主动插入的 CS：opportunistic
        - risk_mode 下主动插入的 CS：risk
        - 回退修复插入的 CS：repair
        - 原路线自带或未标记的 CS：natural
        """

        if demand is None:
            demand = self.problem.demand

        route = route[:]  # 避免直接改外部传入对象
        route_charge_tag = [None] * len(route)  # 与 route 对齐，记录每次节点访问的充电标签

        Q = self.problem.soc_max
        EPS = self.EPS
        lock_to_depot = False

        self.cs_in_3km_node = self._build_anchor_cs_map(route[1:-1])

        # ---------- RA 状态 ----------
        t_since_charge = 0.0
        Q_ref = Q
        total_ra = 0.0

        # ---------- 总成本 ----------
        total_charging_time = 0.0
        total_travel_time = 0.0
        total_service_time = 0.0
        total_dist = 0.0
        num_charge = 0
        num_restock = 0

        # ---------- 统计 ----------
        stats = {
            # A) 充电事件
            "num_charge_total": 0,
            "num_charge_opportunistic": 0,
            "num_charge_risk": 0,
            "num_charge_repair": 0,
            "num_charge_natural": 0,

            "charging_time_total": 0.0,
            "charging_time_opportunistic": 0.0,
            "charging_time_risk": 0.0,
            "charging_time_repair": 0.0,
            "charging_time_natural": 0.0,

            "charged_energy_total": 0.0,
            "charged_energy_opportunistic": 0.0,
            "charged_energy_risk": 0.0,
            "charged_energy_repair": 0.0,
            "charged_energy_natural": 0.0,
        }

        # 记录最近一次 backtrack 触发时的 mode
        trigger_mode = None

        state_log = []
        i = 1

        BIG_PENALTY = 1e6
        WATCHDOG_LIMIT = 200
        watchdog = 0
        tried_pairs = set()

        # 防止“回退插桩分支已充电”后，到达同一CS又重复充电/计数
        skip_charge_once = False
        skip_charge_cs_id = None

        # ---------- detour（补货绕行）标记 ----------
        detour_active = False
        detour_end_node = None
        detour_depot_id = None
        detour_case = None
        detour_curr_customer = None

        CAP = getattr(self.problem, "vehicle_capacity", None)
        if CAP is None:
            CAP = getattr(self.problem, "cap", None)
        if CAP is None:
            raise AttributeError("problem 中找不到 vehicle_capacity/cap")

        def _clone_stats(s):
            return dict(s)

        def _nearest_depot(node_id: int) -> int:
            depots = list(self.problem.depots)
            if not depots:
                raise AttributeError("problem.depots 为空")
            dm = self.problem.distance_matrix
            best = depots[0]
            best_d = dm[node_id, best]
            for d in depots[1:]:
                dd = dm[node_id, d]
                if dd < best_d:
                    best_d = dd
                    best = d
            return best

        def _remaining_expected_demand(start_index: int) -> float:
            """从 route[start_index:] 中取后续客户的期望需求（用 self.problem.demand 当 μ）"""
            mu = self.problem.demand
            total = 0.0
            for n in route[start_index:]:
                if n in self.problem.customers:
                    total += float(mu[n])
            return total

        def _add_charge_stats(bucket: str, charge_time: float, charged_energy: float):
            """统一累计充电统计"""
            if charged_energy <= 0.0 and charge_time <= 0.0:
                return

            stats["num_charge_total"] += 1
            stats["charging_time_total"] += charge_time
            stats["charged_energy_total"] += charged_energy

            if bucket == "opportunistic":
                stats["num_charge_opportunistic"] += 1
                stats["charging_time_opportunistic"] += charge_time
                stats["charged_energy_opportunistic"] += charged_energy
            elif bucket == "risk":
                stats["num_charge_risk"] += 1
                stats["charging_time_risk"] += charge_time
                stats["charged_energy_risk"] += charged_energy
            elif bucket == "repair":
                stats["num_charge_repair"] += 1
                stats["charging_time_repair"] += charge_time
                stats["charged_energy_repair"] += charged_energy
            elif bucket == "natural":
                stats["num_charge_natural"] += 1
                stats["charging_time_natural"] += charge_time
                stats["charged_energy_natural"] += charged_energy
            else:
                raise ValueError(f"Unknown charge bucket: {bucket}")

        def penalty_exit(reason: str):
            self.last_decode_stats = stats
            return (
                route,
                BIG_PENALTY,  # total_dist
                BIG_PENALTY,  # total_travel_time
                BIG_PENALTY,  # total_charging_time
                BIG_PENALTY,  # total_service_time
                999,          # num_charge
                0.0,          # Q
                departure_time,
                load,
                BIG_PENALTY,  # total_ra
                999,          # num_restock
                stats,
            )

        while i < len(route):
            watchdog += 1
            if watchdog > WATCHDOG_LIMIT:
                return penalty_exit("WATCHDOG")

            prev_node = route[i - 1]
            curr_node = route[i]

            # ---------- detour 结束判定 ----------
            if (
                detour_active
                and (detour_end_node is not None)
                and (curr_node == detour_end_node)
                and (prev_node == detour_depot_id)
            ):
                detour_active = False
                detour_end_node = None
                detour_depot_id = None
                detour_case = None
                detour_curr_customer = None

            # ---------- RA & mode ----------
            if getattr(self, "ra_model", None) is not None:
                Qtr = max(0.0, Q_ref - Q)
                Ttr = t_since_charge
                hour_of_day = (departure_time / 60.0) % 24.0
                ra = self.ra_model.R_instant(Qtr, Ttr, hour_of_day)
                mode = self._ra_mode(ra)
            else:
                mode = "safe_mode"

            # ---------- risk_mode：强制去最近可达 CS ----------
            if (not lock_to_depot) and (mode == "risk_mode"):
                cs_result = self._to_nearest_cs(prev_node, Q, departure_time, load)
                if cs_result is not None:
                    cs_id, _, _, _ = cs_result
                    if curr_node != cs_id:
                        route.insert(i, cs_id)
                        route_charge_tag.insert(i, "risk")
                        continue

            # ---------- opportunity_mode：顺路插桩 ----------
            if (not lock_to_depot) and (mode == "opportunity_mode") and (curr_node in self.cs_in_3km_node):
                cs_id = self.cs_in_3km_node[curr_node]
                already_before = (route[i - 1] == cs_id) if i - 1 >= 0 else False
                already_after = (i + 1 < len(route) and route[i + 1] == cs_id)

                inserted_now = False
                if (not already_before) and (not already_after):
                    next_node = route[i + 1] if i + 1 < len(route) else None

                    d_prev_cs = self.problem.distance_matrix[prev_node, cs_id]
                    d_cs_curr = self.problem.distance_matrix[cs_id, curr_node]
                    d_prev_curr = self.problem.distance_matrix[prev_node, curr_node]
                    d_curr_cs = self.problem.distance_matrix[curr_node, cs_id]

                    if next_node is not None:
                        d_curr_next = self.problem.distance_matrix[curr_node, next_node]
                        d_cs_next = self.problem.distance_matrix[cs_id, next_node]
                        insert_before_cost = d_prev_cs + d_cs_curr + d_curr_next
                        insert_after_cost = d_prev_curr + d_curr_cs + d_cs_next
                    else:
                        insert_before_cost = d_prev_cs + d_cs_curr
                        insert_after_cost = d_prev_curr + d_curr_cs

                    if insert_before_cost + EPS < insert_after_cost:
                        route.insert(i, cs_id)
                        route_charge_tag.insert(i, "opportunistic")
                    else:
                        route.insert(i + 1, cs_id)
                        route_charge_tag.insert(i + 1, "opportunistic")
                    inserted_now = True

                if inserted_now:
                    continue

            # ---------- 记录快照（用于回退修复） ----------
            state_log.append({
                "prev_i": i - 1,
                "route": route[:],
                "route_charge_tag": route_charge_tag[:],
                "Q": Q,
                "Q_ref": Q_ref,
                "t_since_charge": t_since_charge,
                "departure_time": departure_time,
                "load": load,
                "total_travel_time": total_travel_time,
                "total_service_time": total_service_time,
                "total_charging_time": total_charging_time,
                "total_dist": total_dist,
                "total_ra": total_ra,
                "num_charge": num_charge,
                "num_restock": num_restock,
                "lock_to_depot": lock_to_depot,

                "stats": _clone_stats(stats),

                "detour_active": detour_active,
                "detour_end_node": detour_end_node,
                "detour_depot_id": detour_depot_id,
                "detour_case": detour_case,
                "detour_curr_customer": detour_curr_customer,

                "skip_charge_once": skip_charge_once,
                "skip_charge_cs_id": skip_charge_cs_id,
            })

            # ---------- 常规前进 prev -> curr ----------
            dist = self.problem.distance_matrix[prev_node, curr_node]
            energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(
                dist, departure_time, load
            )

            # ---------- 不可达：回退插桩修复 ----------
            if Q + EPS < energy_needed:
                trigger_mode = mode

                inserted = False
                backtrack_depth = 0

                while state_log:
                    backtrack_depth += 1
                    snap = state_log.pop()

                    prev_i = snap["prev_i"]
                    snap_route = snap["route"]
                    snap_route_charge_tag = snap["route_charge_tag"]
                    snap_Q = snap["Q"]
                    snap_Q_ref = snap["Q_ref"]
                    snap_t_since = snap["t_since_charge"]
                    snap_t = snap["departure_time"]
                    snap_load = snap["load"]

                    snap_tt = snap["total_travel_time"]
                    snap_st = snap["total_service_time"]
                    snap_ct = snap["total_charging_time"]
                    snap_dist = snap["total_dist"]
                    snap_ra = snap["total_ra"]
                    snap_nchg = snap["num_charge"]
                    snap_nrest = snap["num_restock"]
                    snap_lock = snap["lock_to_depot"]

                    snap_stats = snap["stats"]

                    snap_detour_active = snap["detour_active"]
                    snap_detour_end_node = snap["detour_end_node"]
                    snap_detour_depot_id = snap["detour_depot_id"]
                    snap_detour_case = snap["detour_case"]
                    snap_detour_curr_customer = snap["detour_curr_customer"]

                    snap_skip_charge_once = snap["skip_charge_once"]
                    snap_skip_charge_cs_id = snap["skip_charge_cs_id"]

                    prev_node2 = snap_route[prev_i]
                    cs_result = self._to_nearest_cs(prev_node2, snap_Q, snap_t, snap_load)
                    if cs_result is None:
                        continue

                    cs_id, to_cs_dist, to_cs_energy, to_cs_time = cs_result

                    if cs_id == prev_node2:
                        continue
                    if prev_i + 1 < len(snap_route) and snap_route[prev_i + 1] == cs_id:
                        continue
                    if (prev_i, cs_id) in tried_pairs:
                        return penalty_exit("TRIED_PAIRS")
                    tried_pairs.add((prev_i, cs_id))

                    # 恢复到 snap 对应状态
                    stats = _clone_stats(snap_stats)
                    detour_active = snap_detour_active
                    detour_end_node = snap_detour_end_node
                    detour_depot_id = snap_detour_depot_id
                    detour_case = snap_detour_case
                    detour_curr_customer = snap_detour_curr_customer
                    skip_charge_once = snap_skip_charge_once
                    skip_charge_cs_id = snap_skip_charge_cs_id

                    # 插入 repair CS
                    snap_route.insert(prev_i + 1, cs_id)
                    snap_route_charge_tag.insert(prev_i + 1, "repair")

                    # 到达 CS
                    t_arrive_cs = snap_t + to_cs_time
                    Q_arrive_cs = snap_Q - to_cs_energy

                    # full charging
                    required_charge = max(0.0, self.problem.soc_max - Q_arrive_cs)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)

                    Q = self.problem.soc_max
                    departure_time = t_arrive_cs + charge_time
                    load = snap_load

                    total_travel_time = snap_tt + to_cs_time
                    total_service_time = snap_st
                    total_dist = snap_dist + to_cs_dist
                    total_charging_time = snap_ct + charge_time

                    _add_charge_stats("repair", charge_time, required_charge)

                    num_charge = stats["num_charge_total"]
                    num_restock = snap_nrest

                    # RA：prev_node2 -> cs 的增量
                    t_since_charge = 0.0
                    Q_ref = Q
                    if getattr(self, "ra_model", None) is not None:
                        Ttr0 = snap_t_since
                        hour0 = (snap_t / 60.0) % 24.0
                        Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                        Qtr_after = Qtr_before + to_cs_energy
                        seg_ra = (
                            self.ra_model.SUMR(Qtr_after, Ttr0, hour0)
                            - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                        )
                        total_ra = snap_ra + seg_ra
                    else:
                        total_ra = snap_ra

                    if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                        lock_to_depot = True
                    else:
                        lock_to_depot = snap_lock

                    route = snap_route
                    route_charge_tag = snap_route_charge_tag[:]
                    i = prev_i + 2
                    state_log = state_log[:prev_i + 1]
                    inserted = True

                    # 防止到达这个 CS 后再次重复充电
                    skip_charge_once = True
                    skip_charge_cs_id = cs_id

                    # 修复后立即检查下一跳可达性
                    if i < len(route):
                        next_node = route[i]
                        dist2 = self.problem.distance_matrix[route[i - 1], next_node]
                        energy2, _, _ = self.energy_model.calculate_one_node_energy_time(
                            dist2, departure_time, load
                        )
                        if Q + EPS < energy2:
                            return penalty_exit("REPAIR_STILL_INFEASIBLE")

                    break

                if not inserted:
                    return penalty_exit("REPAIR_FAILED")

                continue

            # ---------- 可达：累加该段 RA ----------
            if getattr(self, "ra_model", None) is not None:
                Ttr0 = t_since_charge
                hour0 = (departure_time / 60.0) % 24.0
                Qtr_before = max(0.0, Q_ref - Q)
                Qtr_after = Qtr_before + energy_needed
                seg_ra = (
                    self.ra_model.SUMR(Qtr_after, Ttr0, hour0)
                    - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                )
                total_ra += seg_ra

            # ---------- 推进 prev -> curr ----------
            arrival_time = departure_time + travel_time
            Q -= energy_needed
            if Q + EPS < 0:
                return penalty_exit("NEGATIVE_Q")

            total_travel_time += travel_time
            total_dist += dist
            t_since_charge += travel_time

            # ---------- 到达 depot：补货（仅在 detour 中） ----------
            if detour_active and (curr_node in self.problem.depots):
                if detour_case == "case2":
                    curr_real = float(demand[detour_curr_customer]) if detour_curr_customer is not None else 0.0
                    tail_exp = _remaining_expected_demand(i + 1)
                    target_load = curr_real + tail_exp
                else:
                    target_load = _remaining_expected_demand(i + 1)

                load = min(CAP, max(0.0, target_load))
                num_restock += 1

                service_time_depot = self.problem.service_time[curr_node]
                departure_time = arrival_time + service_time_depot
                total_service_time += service_time_depot

                i += 1
                continue

            # ---------- Case2：到达客户发现 load 不足 ----------
            if (curr_node in self.problem.customers) and (load + EPS < float(demand[curr_node])):
                depot_id = _nearest_depot(curr_node)

                # curr -> depot -> curr
                route.insert(i + 1, depot_id)
                route_charge_tag.insert(i + 1, None)

                route.insert(i + 2, curr_node)
                route_charge_tag.insert(i + 2, None)

                detour_active = True
                detour_end_node = curr_node
                detour_depot_id = depot_id
                detour_case = "case2"
                detour_curr_customer = curr_node

                # 第一次到 curr 不服务，立刻转去 depot
                departure_time = arrival_time
                i += 1
                continue

            # ---------- 服务客户 / 节点 ----------
            service_time = self.problem.service_time[curr_node]
            departure_time = arrival_time + service_time
            total_service_time += service_time

            load -= float(demand[curr_node])
            if load < 0:
                load = 0.0

            # ---------- Case3：服务后空载且后面还有客户 ----------
            if (curr_node in self.problem.customers) and (abs(load) <= EPS) and (i + 1 < len(route)):
                next_node = route[i + 1]
                if next_node in self.problem.customers:
                    depot_id = _nearest_depot(curr_node)

                    # curr -> depot -> next
                    route.insert(i + 1, depot_id)
                    route_charge_tag.insert(i + 1, None)

                    detour_active = True
                    detour_end_node = next_node
                    detour_depot_id = depot_id
                    detour_case = "case3"
                    detour_curr_customer = None

                    i += 1
                    continue

            # ---------- 到达 CS：充电 ----------
            if curr_node in self.charging_nodes:
                if skip_charge_once and (curr_node == skip_charge_cs_id):
                    skip_charge_once = False
                    skip_charge_cs_id = None
                else:
                    required_charge = max(0.0, self.problem.soc_max - Q)
                    charge_time = self.energy_model.calculate_charging_time(required_charge)

                    if required_charge > 0 or charge_time > 0:
                        Q = min(self.problem.soc_max, Q + required_charge)
                        departure_time += charge_time
                        total_charging_time += charge_time

                        visit_tag = route_charge_tag[i]

                        if visit_tag == "opportunistic":
                            _add_charge_stats("opportunistic", charge_time, required_charge)
                        elif visit_tag == "risk":
                            _add_charge_stats("risk", charge_time, required_charge)
                        elif visit_tag == "repair":
                            # 正常不应走到这里；repair 已在 backtrack 分支统计过，
                            # 且这里通常会被 skip_charge_once 跳过。
                            return penalty_exit("REPAIR_DOUBLE_COUNT")
                        else:
                            _add_charge_stats("natural", charge_time, required_charge)

                        num_charge = stats["num_charge_total"]

                    t_since_charge = 0.0
                    Q_ref = Q

                    if self._can_reach_final_depot(i, route, Q, departure_time, load):
                        lock_to_depot = True

            i += 1

        classified_total = (
            stats["num_charge_opportunistic"]
            + stats["num_charge_risk"]
            + stats["num_charge_repair"]
            + stats["num_charge_natural"]
        )

        if stats["num_charge_total"] != classified_total:
            raise ValueError(
                f"charge count mismatch: total={stats['num_charge_total']}, "
                f"opp={stats['num_charge_opportunistic']}, "
                f"risk={stats['num_charge_risk']}, "
                f"repair={stats['num_charge_repair']}, "
                f"natural={stats['num_charge_natural']}"
            )

        self.last_decode_stats = stats
        num_charge = int(stats["num_charge_total"])

        return (
            route,
            total_dist,
            total_travel_time,
            total_charging_time,
            total_service_time,
            num_charge,
            Q,
            departure_time,
            load,
            total_ra,
            num_restock,
            stats,
        )











