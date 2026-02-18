import os
import sys

import numpy as np

from Modules.Clustering.plot import plot_clusters

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)




from Modules.EVRP.loader import load_problem      # 按你实际路径改
from Modules.Clustering.api import clustering               # 你重构后的统一入口


def main():
    # -----------------------------
    # 1. Load instance
    # -----------------------------
    instance_name = 'rc103_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    problem, mappings = load_problem(file_path, ev_params, return_mappings=True)

    print("Loaded problem:")
    print("customers =", len(problem.customers))
    print("stations  =", len(problem.css))
    print("depots    =", len(problem.depots))

    # -----------------------------
    # 2. Run clustering (EVRP API)
    # -----------------------------
    clusters = clustering(
        problem,
        method="kmeans",          # kmeans / kmedoids
        feature="time_window",         # location -> problem.distance;time_window
        random_state=0,
    )

    # clusters: dict[int, set[int]]
    print("\nClustering result:")
    print("n_clusters =", len(clusters))

    for cid, members in clusters.items():
        print(f"Cluster {cid}: size={len(members)}")
    print(clusters)
    
    # -----------------------------
    # 3. Plot clustering result
    # -----------------------------


    plot_clusters(problem, clusters, feature="time_window")
    plot_clusters(problem, clusters, feature="location")
    plot_clusters(problem, clusters, feature="time_window_mid_width")





if __name__ == "__main__":
    main()


