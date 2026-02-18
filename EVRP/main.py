import os

from EVRP.loader import load_problem


def main():
    # ----------------------------
    # 1. 实例路径
    # ----------------------------
    instance_name = "rc103_21"
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    print("Instance file:", file_path)

    # ----------------------------
    # 2. EV 参数（必须提供）
    # ----------------------------
    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    # ----------------------------
    # 3. 读取 Problem
    # ----------------------------
    problem = load_problem(file_path, ev_params, return_mappings=False)

    # ----------------------------
    # 4. 基本信息输出
    # ----------------------------
    print("\n===== Problem Loaded Successfully =====")
    print("Total nodes:", len(problem.nodes))
    print("Depots:", problem.depots)
    print("Number of customers:", len(problem.customers))
    print("Number of charging stations:", len(problem.css))

    print("\nDistance matrix size:",
          len(problem.distance_matrix), "x", len(problem.distance_matrix[0]))

    # ----------------------------
    # 5. 测试访问一个客户节点
    # ----------------------------
    if problem.customers:
        c = problem.customers[0]
        node = problem.nodes[c]

        print("\n===== First Customer Example =====")
        print("Customer idx:", c)
        print("Location:", (node.x, node.y))
        print("Demand:", problem.demand[c])
        print("Time window:", (node.ready_time, node.due_time))

        # 最近 depot
        nearest = problem.nearest_depot(c)
        print("Nearest depot idx:", nearest)
        print("Distance to depot:", problem.distance_matrix[c][nearest])

    print("\n===== Done =====")
    print(dir(problem))



if __name__ == "__main__":
    main()
