def decode_spr(self, route, load, departure_time=0, demand=None):
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
    total_ra = 0.0

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

    def penalty_exit():
        return (
            route,
            BIG_PENALTY, BIG_PENALTY,
            BIG_PENALTY, BIG_PENALTY,
            999,
            0.0,
            departure_time,
            load,
            BIG_PENALTY,
            BIG_PENALTY, BIG_PENALTY, BIG_PENALTY,
            999, 999
        )

    def _move_with_recourse_repair(a, b, Q_local, t_local, load_local):
        """
        仅用于补货 detour 的移动修复：a->b
        若不可达：a->nearest_cs(可达)->充电->再尝试到 b
        所有新增 travel/charging 记入 recourse_*。
        同时同步 RA 状态：行驶会累加 t_since_charge；充电会重置 t_since_charge 和 Q_ref。
        返回 (Q_new, t_new) 或 None
        """
        nonlocal recourse_dist, recourse_travel_time, recourse_charging_time, recourse_num_charge
        nonlocal t_since_charge, Q_ref

        local_watch = 0
        while True:
            local_watch += 1
            if local_watch > 20:
                return None

            dist_ab = self.problem.distance_matrix[a, b]
            e_ab, t_ab, _ = self.energy_model.calculate_one_node_energy_time(dist_ab, t_local, load_local)

            if Q_local + EPS >= e_ab:
                recourse_dist += dist_ab
                recourse_travel_time += t_ab
                Q_local -= e_ab
                t_local += t_ab
                t_since_charge += t_ab
                return Q_local, t_local

            cs_result = self._to_nearest_cs(a, Q_local, t_local, load_local)
            if cs_result is None:
                return None

            cs_id, d_cs, e_cs, t_cs = cs_result
            if cs_id == a:
                return None

            # a -> cs
            recourse_dist += d_cs
            recourse_travel_time += t_cs
            Q_local -= e_cs
            t_local += t_cs
            t_since_charge += t_cs

            # 在 cs 充电（repair，记入 recourse）
            required_charge, charge_time = self._estimate_partial_charge(i, route, Q_local, t_local, load_local)
            if required_charge > 0 or charge_time > 0:
                Q_local = min(self.problem.soc_max, Q_local + required_charge)
                t_local += charge_time
                recourse_charging_time += charge_time
                recourse_num_charge += 1

                # 充电后 RA 参考重置
                t_since_charge = 0.0
                Q_ref = Q_local

            a = cs_id

    while i < len(route):
        watchdog += 1
        if watchdog > WATCHDOG_LIMIT:
            return penalty_exit()

        prev_node = route[i - 1]
        curr_node = route[i]

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
            "total_ra": total_ra,
            "lock_to_depot": lock_to_depot,

            "recourse_charging_time": recourse_charging_time,
            "recourse_travel_time": recourse_travel_time,
            "recourse_dist": recourse_dist,
            "recourse_num_restock": recourse_num_restock,
            "recourse_num_charge": recourse_num_charge,
        })

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

        # ---------- 常规前进 prev → curr ----------
        dist = self.problem.distance_matrix[prev_node, curr_node]
        energy_needed, travel_time, _ = self.energy_model.calculate_one_node_energy_time(dist, departure_time, load)

        # 不可达：回退插桩（repair 充电 -> recourse）
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
                snap_ra_total = snap["total_ra"]
                snap_lock = snap["lock_to_depot"]

                snap_rct = snap["recourse_charging_time"]
                snap_rtt = snap["recourse_travel_time"]
                snap_rdist = snap["recourse_dist"]
                snap_rrest = snap["recourse_num_restock"]
                snap_rnchg = snap["recourse_num_charge"]

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
                    return penalty_exit()
                tried_pairs.add((prev_i, cs_id))

                snap_route.insert(prev_i + 1, cs_id)

                # 先到 CS
                t_arrive_cs = snap_t + to_cs_time
                Q_arrive_cs = snap_Q - to_cs_energy

                required_charge, charge_time = self._estimate_partial_charge(
                    prev_i + 1, snap_route, Q_arrive_cs, t_arrive_cs, snap_load
                )

                Q = min(self.problem.soc_max, Q_arrive_cs + required_charge)
                departure_time = t_arrive_cs + charge_time
                load = snap_load

                # plan：回到快照并加到 CS
                plan_travel_time = snap_tt + to_cs_time
                plan_service_time = snap_st
                plan_dist = snap_dist + to_cs_dist
                plan_charging_time = snap_ct
                plan_num_charge = snap_nchg

                # recourse：repair 充电
                recourse_charging_time = snap_rct + charge_time
                recourse_num_charge = snap_rnchg + (1 if (charge_time > 0 or required_charge > 0) else 0)
                recourse_travel_time = snap_rtt
                recourse_dist = snap_rdist
                recourse_num_restock = snap_rrest

                # RA 重置
                t_since_charge = 0.0
                Q_ref = Q

                if getattr(self, "ra_model", None) is not None:
                    Ttr0 = snap_t_since
                    hour0 = (snap_t / 60.0) % 24.0
                    Qtr_before = max(0.0, snap_Q_ref - snap_Q)
                    Qtr_after = Qtr_before + to_cs_energy
                    seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
                    total_ra = snap_ra_total + seg_ra
                else:
                    total_ra = snap_ra_total

                if self._can_reach_final_depot(prev_i + 1, snap_route, Q, departure_time, load):
                    lock_to_depot = True
                else:
                    lock_to_depot = snap_lock

                route = snap_route
                i = prev_i + 2
                state_log = state_log[:prev_i + 1]
                inserted = True

                skip_charge_once = True
                skip_charge_cs_id = cs_id

                # 插桩后立即检查下一跳可达
                if i < len(route):
                    next_node = route[i]
                    dist2 = self.problem.distance_matrix[route[i - 1], next_node]
                    energy2, _, _ = self.energy_model.calculate_one_node_energy_time(dist2, departure_time, load)
                    if Q + EPS < energy2:
                        return penalty_exit()

                break

            if not inserted:
                return penalty_exit()

            continue

        # ---------- 可达：推进 prev→curr（计划行驶） ----------
        if getattr(self, "ra_model", None) is not None:
            Ttr0 = t_since_charge
            hour0 = (departure_time / 60.0) % 24.0
            Qtr_before = max(0.0, Q_ref - Q)
            Qtr_after = Qtr_before + energy_needed
            seg_ra = self.ra_model.SUMR(Qtr_after, Ttr0, hour0) - self.ra_model.SUMR(Qtr_before, Ttr0, hour0)
            total_ra += seg_ra

        # 先走 prev->curr
        arrival_time = departure_time + travel_time
        Q -= energy_needed
        if Q + EPS < 0:
            return penalty_exit()

        plan_travel_time += travel_time
        plan_dist += dist
        t_since_charge += travel_time

        # ----------------------------
        # Case 2: 到达客户发现 load 不足 -> curr->depot->curr（detour，recourse）
        # ----------------------------
        if (curr_node in self.problem.customers) and (load + EPS < float(demand[curr_node])):
            depot_id = _nearest_depot(curr_node)

            res = _move_with_recourse_repair(curr_node, depot_id, Q, arrival_time, load)
            if res is None:
                return penalty_exit()
            Q, t_at_depot = res

            need_exp = _remaining_expected_demand(i)
            load = min(CAP, need_exp)
            recourse_num_restock += 1

            res = _move_with_recourse_repair(depot_id, curr_node, Q, t_at_depot, load)
            if res is None:
                return penalty_exit()
            Q, arrival_time = res

        # 服务（一次）
        service_time = self.problem.service_time[curr_node]
        departure_time = arrival_time + service_time
        plan_service_time += service_time

        # 扣减载重（按情景需求）
        load -= float(demand[curr_node])

        # ----------------------------
        # Case 3: 服务后 load==0 且下一个是客户 -> curr->depot->next（detour，recourse）
        # 完成 detour 后跳过下一轮 curr->next 的计划行驶（避免重复计费）
        # ----------------------------
        if (curr_node in self.problem.customers) and (abs(load) <= EPS) and (i + 1 < len(route)):
            next_node = route[i + 1]
            if next_node in self.problem.customers:
                depot_id = _nearest_depot(curr_node)

                res = _move_with_recourse_repair(curr_node, depot_id, Q, departure_time, load)
                if res is None:
                    return penalty_exit()
                Q, t_at_depot = res

                need_exp = _remaining_expected_demand(i + 1)
                load = min(CAP, need_exp)
                recourse_num_restock += 1

                res = _move_with_recourse_repair(depot_id, next_node, Q, t_at_depot, load)
                if res is None:
                    return penalty_exit()
                Q, departure_time = res

                # 我们已经抵达 next_node：跳过下一轮对 curr->next 的计划行驶统计
                i += 1
                continue

        # ---------- 到达CS：计划充电 ----------
        if curr_node in self.problem.css:
            if skip_charge_once and (curr_node == skip_charge_cs_id):
                skip_charge_once = False
                skip_charge_cs_id = None
            else:
                required_charge, charge_time = self._estimate_partial_charge(i, route, Q, departure_time, load)
                if required_charge > 0 or charge_time > 0:
                    Q = min(self.problem.soc_max, Q + required_charge)
                    departure_time += charge_time
                    plan_charging_time += charge_time
                    plan_num_charge += 1

                t_since_charge = 0.0
                Q_ref = Q

                if self._can_reach_final_depot(i, route, Q, departure_time, load):
                    lock_to_depot = True

        i += 1

    return (
        route,
        plan_dist, plan_travel_time,
        plan_charging_time, plan_service_time,
        plan_num_charge,
        Q, departure_time, load, total_ra,
        recourse_dist, recourse_travel_time, recourse_charging_time,
        recourse_num_restock, recourse_num_charge
    )
