from __future__ import annotations

from typing import Dict, Set, List, Literal, Any, Optional

from .methods.k_means import ImprovedKMeans
from .methods.k_medoids import ImprovedKMedoids


ClusterMethod = Literal["kmeans", "kmedoids"]
ClusterFeature = Literal["location", "time_window", "time_window_mid_width"]


def clustering(
    problem,
    *,
    method: ClusterMethod = "kmeans",
    feature: ClusterFeature = "location",
    vehicle_capacity: Optional[float] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[int, Set[int]]:

    customer_ids: List[int] = list(problem.customers)

    if method == "kmeans":
        model = ImprovedKMeans(
            problem,
            customer_ids,
            feature=feature,
            vehicle_capacity=vehicle_capacity,
            random_state=random_state,
            **kwargs,
        )
        return model.fit()

    elif method == "kmedoids":
        model = ImprovedKMedoids(
            problem,
            customer_ids,
            feature=feature,
            vehicle_capacity=vehicle_capacity,
            random_state=random_state,
            **kwargs,
        )
        return model.fit()

    else:
        raise ValueError(f"Unknown clustering method: {method}")




