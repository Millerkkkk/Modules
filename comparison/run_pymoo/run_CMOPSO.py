import os
import numpy as np
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.optimize import minimize

from comparison.run_pymoo.problem import load_instance, plot_pareto, solve, termination, print_summary, print_decoded_solutions





def run_gb_cmopso(
    instance_path: str,
    beta: float,
    pop_size: int = 50,
    n_gen: int = 200,
    base_seed: int = 42,
    n_seed_solutions: int = 10,
    n_scenarios: int = 100,
    n_reduced_scenarios: int = 10,
):
    problem = load_problem(instance_path, ev_params)

    # 场景
    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=n_scenarios,
        random_state=base_seed,
    )

    _, reduced_scenarios, reduced_probs = fast_forward_selection(
        scenarios=scenarios,
        n_select=n_reduced_scenarios,
    )

    # evaluator
    evaluator_obj = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    def evaluator_fn(routes_clusters):
        cend = evaluator_obj.evaluate_ffs(
            routes_clusters,
            reduced_scenarios,
            reduced_probs,
        )
        # evaluate_ffs 有时返回 (eval_result, repaired_routes)
        if isinstance(cend, tuple) and len(cend) == 2:
            return cend[0]
        return cend

    # 生成多个 GB 初始解
    clusters, seed_solutions = generate_gb_seed_solutions(
        problem=problem,
        base_seed=base_seed,
        n_seed_solutions=n_seed_solutions,
    )

    # 取第一个解作为骨架模板
    base_routes_clusters = seed_solutions[0]
    state = build_block_state(base_routes_clusters)

    # problem / sampling / algorithm
    mo_problem = GBClusterRoutingProblem(
        state=state,
        evaluator_fn=evaluator_fn,
    )

    sampling = GBHeuristicSampling(
        state=state,
        seed_solutions=seed_solutions,
    )

    algorithm = CMOPSO(
        pop_size=pop_size,
        sampling=sampling,
    )

    res = minimize(
        mo_problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=base_seed,
        verbose=True,
    )

    pareto_routes = [decode_routes_clusters(x, state) for x in res.X]

    return {
        "X": res.X,
        "F": res.F,
        "routes_clusters_list": pareto_routes,
        "seed_solutions": seed_solutions,
        "clusters": clusters,
    }




def build_algorithm_cmopso(pop_size, sampling):
    return CMOPSO(pop_size=pop_size, sampling=sampling,)


def main():
    begin_time = time.time()

    md_evrp_problem = load_instance(
        instance_name=INSTANCE_NAME,
        base_path=BASE_PATH,
        beta=0.1,
        n_scenarios=10,
        seed=42
    )

    algorithm = build_algorithm_cmopso()

    res = solve(md_evrp_problem, algorithm)

    run_time = time.time() - begin_time

    print_summary(res, INSTANCE_NAME, run_time)
    print_decoded_solutions(md_evrp_problem, res, max_solutions=5)
    plot_pareto(res, INSTANCE_NAME)


if __name__ == "__main__":

    INSTANCE_NAME = "rc103_21"
    BASE_PATH = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    SEED = 1
    POP_SIZE = 100

    main()
