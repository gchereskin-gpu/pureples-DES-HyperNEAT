"""
An experiment using a variable-sized Adaptive DES-HyperNEAT network to perform a pole balancing task.
"""

import os
import pickle
import logging
import neat
import gymnasium as gym
import matplotlib.pyplot as plt
from pureples.experiments.t_maze.multi_t_maze import MultiTMazeEnv, save_run_gif
from pureples.experiments.t_maze import study
from pureples.shared.visualize import draw_net, plot_fitness_stats
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_adaptive_des
from pureples.des_hyperneat import AdaptiveDESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "L"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

MAZE_LENGTH = 1
MAZE_LENGTH_TEXT = "1_turn" if MAZE_LENGTH == 1 else str(MAZE_LENGTH) + "_turns"

NUM_DEPLOYMENTS = 4

NUM_RUNS = 2

# Maximum number of generations per evolutionary run (also the length the
# per-run fitness series are padded to for the multi-run study).
MAX_GENERATIONS = 1000

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = []

for i in range(0, 5):
    INPUT_COORDINATES.append((-1. + (1.0*i/2), -1.))
INPUT_COORDINATES.append((0., -1.2))

OUTPUT_COORDINATES = [(-1., 1.), (0., 1.), (1., 1.)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def params(version):
    """
    Adaptive DES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 8.0,
            "activation": "sigmoid"}


# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.AdaptiveDesGenome, neat.reproduction.DesReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/t_maze/config_cppn_t_maze')


def run(gens, env, version):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_adaptive_des(gens, env, 500, CONFIG, params(version), SUBSTRATE, NUM_DEPLOYMENTS,
                                     num_workers=12)
    print(f"adaptive_des_hyperneat_t_maze_{MAZE_LENGTH_TEXT}_{VERSION_TEXT} done")
    return winner, stats


def std_connection_an_products(net):
    """
    Yield the A*n product for every standard connection in an Adaptive DES
    phenotype network. Each link in ``net.node_evals`` is
    ``(in_id, weight, branch_id, a, b, c, d, n, mod_weights)``, so ``a`` and
    ``n`` are at indices 3 and 7.
    """
    return [link[3] * link[7]
            for _, _, _, _, _, links in net.node_evals
            for link in links]


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = MultiTMazeEnv(MAZE_LENGTH)

    BASE_NAME = f"adaptive_des_hyperneat_t_maze_{MAZE_LENGTH_TEXT}_{VERSION_TEXT}"
    RESULTS_DIR = "pureples/experiments/t_maze/results"
    DATA_DIR = f"{RESULTS_DIR}/{BASE_NAME}_data"

    def save_standard_outputs(winner, stats, prefix, network_title):
        """
        Save the files produced for every winning network -- the fitness plot,
        the phenotype network image (built with show=False so the interactive
        window is not opened here), the CPPN drawing + pickle, and the behaviour
        gif -- under ``prefix``, and return the (net, cppn) pair for analysis.
        """
        plot_fitness_stats(stats, f"{prefix}_fitness.png",
                           title=f"{network_title} — fitness over generations")

        cppn = neat.nn.DesFeedForwardNetwork.create(winner, CONFIG)
        network = AdaptiveDESNetwork(SUBSTRATE, cppn, params(VERSION))
        net = network.create_phenotype_network(CONFIG, filename=f"{prefix}_winner", show=False)

        draw_net(cppn, filename=f"{prefix}_cppn")
        with open(f"{prefix}_cppn.pkl", 'wb') as output:
            pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

        save_run_gif(net, ENVIRONMENT, f"{prefix}_run.gif", title=network_title, fps=4)
        return net, cppn

    if NUM_RUNS <= 1:
        # Single run: keep the original interactive workflow -- save everything,
        # then pop up the interactive network window last.
        WINNER, STATS = run(MAX_GENERATIONS, ENVIRONMENT, VERSION)
        print(WINNER)
        save_standard_outputs(WINNER, STATS[0], f"{RESULTS_DIR}/{BASE_NAME}",
                              "Adaptive DES T-Maze")
        plt.show()
    else:
        # Multi-run study: run NUM_RUNS evolutions, saving per-run data only for
        # runs that solve the task, then a summary across the successful runs.
        os.makedirs(DATA_DIR, exist_ok=True)
        records = []
        for run_number in range(1, NUM_RUNS + 1):
            print(f"=== Adaptive DES T-Maze: run {run_number} of {NUM_RUNS} ===")
            winner, stats_tuple = run(MAX_GENERATIONS, ENVIRONMENT, VERSION)
            stats = stats_tuple[0]

            if not study.solved(winner, CONFIG):
                print(f"Run {run_number} did not reach the fitness threshold "
                      f"within {MAX_GENERATIONS} generations; no files saved.")
                continue

            run_prefix = f"{DATA_DIR}/{BASE_NAME}_run{run_number}"
            net, _cppn = save_standard_outputs(
                winner, stats, run_prefix, f"Adaptive DES T-Maze run {run_number}")

            mean_series, best_series = study.padded_fitness_series(stats, MAX_GENERATIONS)
            mean_an, inhibitory, excitatory = study.connection_sign_metrics(
                std_connection_an_products(net))
            record = study.RunRecord(
                run_number=run_number,
                generations_to_solution=len(stats.most_fit_genomes),
                mean_fitness_per_gen=mean_series,
                best_fitness_per_gen=best_series,
                mean_an_product=mean_an,
                inhibitory_count=inhibitory,
                excitatory_count=excitatory,
                cppn_connection_count=study.count_cppn_connections(winner),
                num_branches=net.num_branches,
            )
            study.write_run_report(f"{run_prefix}_data.txt", record, MAX_GENERATIONS,
                                   title="Adaptive DES T-Maze")
            records.append(record)

            # Release the (never-shown) network figure before the next run.
            plt.close('all')

        study.write_summary(
            graph_path=f"{DATA_DIR}/{BASE_NAME}_summary_fitness.png",
            text_path=f"{DATA_DIR}/{BASE_NAME}_summary.txt",
            records=records, total_runs=NUM_RUNS,
            max_generations=MAX_GENERATIONS, title="Adaptive DES T-Maze")

        print(f"Study complete: {len(records)} of {NUM_RUNS} run(s) solved the "
              f"task and were saved to {DATA_DIR}.")