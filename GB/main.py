import os
import sys

import numpy as np

from plot import plot_clusters

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)




from Modules.EVRP.loader import load_problem      # 按你实际路径改
from Modules.GB.api import gb_clustering               # 你重构后的统一入口


def main():
    # instance_name = 'c103_21'
    # base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"

    instance_name = 'pr02_evrp'
    base_path = r"D:\02_Research\DataSet\C-mdvrptw-improved"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }
    
    instance = load_problem(file_path, ev_params)

    # 1) 用 EVRP 接口取客户节点与坐标
    customer_ids = list(instance.customers)
    points = np.asarray([(instance.nodes[i].x, instance.nodes[i].y) for i in customer_ids], dtype=float)

    # （可选）只对某个 cluster 的客户做 GB
    # customer_ids = list(cluster[1])
    # points = np.array([instance.nodes[i].location() for i in customer_ids], dtype=float)

    # # 2) 跑 GB（返回 clusters 的索引是 0..len(points)-1 的“局部索引”）
    # clusters_local, meta = cluster(
    #     points,
    #     method="gb_kmeans",      # 或 "gb_dbscan"
    #     return_meta=True,
    #     split_k=1.0,             # kmeans: split_k
    #     merge=False,              # kmeans: 是否做 overlap/contain
    #     random_state=0,
    #     # dbscan 的话参数通常是 split_k / core_ratio
    # )

    clusters_local = gb_clustering(
        points,
        method="gb_kmedoids",
        split_k=0.6,
        merge=False,
        random_state=0,
    )


    # clusters_local, meta = cluster(
    #     points,
    #     method="gb_dbscan", 
    #     return_meta=True,
    #     split_k=0.4,             # kmeans: split_k
    #     core_ratio=0.92,
    # )

    # 3) 局部索引 -> EVRP customer_id
    clusters_customer_id = [[customer_ids[j] for j in cl] for cl in clusters_local.gbs]

    print("n_customers =", len(customer_ids))
    print("n_clusters  =", len(clusters_customer_id))
    print("clusters(customer_id) =", clusters_customer_id)

    # meta 里有 centers/radii，可用于 plot
    # centers = meta["centers"]
    # radii   = meta["radii"]

    plot_clusters(points, clusters_local.gbs)



if __name__ == "__main__":
    main()
