import os
from pathlib import Path
import sys





sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from matplotlib import pyplot as plt
import numpy as np

from EVRP.loader import load_problem
from comparison.run_SPEA.initializer import HybridInitializer, InitialPopulationConfig
from evaluate import Evaluator
from params import ev_params
from stochastic_demand import generate_normal_scenarios

from comparison.run_SPEA.heuristic import RLSHConstructor
from comparison.run_SPEA.operators import CrossoverManager, MutationManager
from comparison.run_SPEA.spea_engine import SPEAOptimizer


def check_solution(solution, all_customer_ids):
    flat = []
    for r in solution:
        flat.extend(r.customers)

    unique_flat = set(flat)
    all_customer_ids = set(all_customer_ids)

    print("total assigned =", len(flat))
    print("unique assigned =", len(unique_flat))
    print("expected =", len(all_customer_ids))
    print("duplicates =", len(flat) - len(unique_flat))
    print("missing =", len(all_customer_ids - unique_flat))
    print("extra =", len(unique_flat - all_customer_ids))

    if len(flat) != len(unique_flat):
        dup = []
        seen = set()
        for x in flat:
            if x in seen:
                dup.append(x)
            seen.add(x)
        print("duplicated customers:", dup[:20])

    if all_customer_ids - unique_flat:
        print("missing customers sample:", list(all_customer_ids - unique_flat)[:20])

def check_capacity(solution, demand, capacity):
    for i, r in enumerate(solution):
        load = sum(demand[c] for c in r.customers)
        print(f"route {i}: load={load}, capacity={capacity}, feasible={load <= capacity}")


def plot_archive(archive):
    xs = [ind.objectives[0] for ind in archive.individuals]
    ys = [ind.objectives[1] for ind in archive.individuals]

    plt.figure()
    plt.scatter(xs, ys)

    plt.xlabel("Total Cost")
    plt.ylabel("Total RA")
    plt.title("Pareto Front (SPEA2)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    instance_name = 'rc103_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    problem = load_problem(file_path, ev_params)

    
    rng = np.random.default_rng(42)

    rlsh = RLSHConstructor(
        problem,
    )

    config = InitialPopulationConfig(
        population_size=100,
        heuristic_ratio=0.3,          # 论文实验推荐 30%
        shuffle_customers_before_rlsh=True,
        randomize_rlsh_lambda=False,  # 先关掉，便于复现
    )

    initializer = HybridInitializer(
        instance=problem,
        rlsh_constructor=rlsh,
        config=config,
    )

    population = initializer.initialize_population(rng)

    print("population size =", len(population))
    print("first individual routes =", len(population[0]))
    print(population[0])

    check_solution(population[0], problem.customers)
    check_capacity(population[0], problem.demand, problem.vehicle_capacity)


    mutation_manager = MutationManager(
        rlsh,
        rrm_times=1,
        rnem_times=10,
        rntm_times=10,
        raem_times=10,
        ratm_times=10,
    )

    crossover_manager = CrossoverManager(
        rlsh,
        hic_times=2,
        ric_times=2,
        inherit_num=2,
    )

    # parent1 = population[0]
    # parent2 = population[1]
    # child_after_mut = mutation_manager.apply_all(population[0], rng)

    # child1, child2 = crossover_manager.apply_one(parent1, parent2, rng)



    beta=0.1
    n_scenarios=1
    seed=42
    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=n_scenarios,
        random_state=seed,
    )

    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)


    optimizer = SPEAOptimizer(
        initializer=initializer,
        evaluator=evaluator,
        crossover_manager=crossover_manager,
        mutation_manager=mutation_manager,
        scenarios=scenarios,
        pop_size=100,
        archive_size=30,
        max_generations=2000,
        rng=rng,
        crossover_prob=0.9,
        mutation_prob=0.2,
        tournament_size=2,
        verbose=True,
    )

    archive = optimizer.run()

    print("Archive size:", archive.size())
    for ind in archive.individuals:
        print(ind.objectives)

    plot_archive(archive)