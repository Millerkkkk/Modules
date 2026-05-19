import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from comparison.run_pymoo.problem import load_instance, plot_pareto, solve, termination, print_summary, print_decoded_solutions





def build_algorithm_moead(N_PARTITIONS, N_NEIGHBORS, PROB_NEIGHBOR_MATING):
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=N_PARTITIONS)

    return MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=N_NEIGHBORS,
        prob_neighbor_mating=PROB_NEIGHBOR_MATING
    )


def main():
    begin_time = time.time()

    md_evrp_problem = load_instance(
        instance_name=INSTANCE_NAME,
        base_path=BASE_PATH,
        beta=0.1,
        n_scenarios=10,
        seed=42
    )

    algorithm = build_algorithm_moead()

    res = solve(md_evrp_problem, algorithm)

    run_time = time.time() - begin_time

    print_summary(res, INSTANCE_NAME, run_time)
    print_decoded_solutions(md_evrp_problem, res, max_solutions=5)
    plot_pareto(res, INSTANCE_NAME)


if __name__ == "__main__":

    INSTANCE_NAME = "rc103_21"
    BASE_PATH = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    SEED = 1
    N_PARTITIONS = 99
    N_NEIGHBORS = 15
    PROB_NEIGHBOR_MATING = 0.7

    main()
