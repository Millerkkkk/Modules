import os
import sys
import time

from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from comparison.run_pymoo.problem import load_instance, plot_pareto, solve, termination, print_summary, print_decoded_solutions





def build_algorithm_nsga3(N_PARTITIONS):
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=N_PARTITIONS)

    return NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(),
        mutation=InversionMutation()
    )


def main():
    begin_time = time.time()

    md_evrp_problem = load_instance(
        instance_name=INSTANCE_NAME,
        base_path=BASE_PATH,
        beta=0.,
        n_scenarios=1,
        seed=42
    )

    algorithm = build_algorithm_nsga3()

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

    main()
