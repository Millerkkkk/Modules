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
                 margin_energy=0.5, radius_km=3.0):
        
        self.problem = problem

        self.energy_model = energy_model
        self.margin_energy = margin_energy
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
        margin = self.margin_energy      # 安全余量（若你的业务允许“刚好到达”即可，可将 margin 设为 0）

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
            nearest_cs, dist = self.problem.nearest_cs(prev_node)
            e, dt, _ = self.energy_model.calculate_one_node_energy_time(dist, t_depart_prev, load_prev)
            if Q_at_prev + self.EPS >= e:
                return (nearest_cs, dist, e, dt)
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

















