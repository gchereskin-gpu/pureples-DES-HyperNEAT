"""
An experiment using a variable-sized Adaptive DES-HyperNEAT network to perform a pole balancing task.
"""

# Force matplotlib's non-interactive Agg backend BEFORE any pyplot import. The
# default Tk backend spawns Tcl GUI objects; when those are garbage-collected
# while the multiprocessing evaluation pool is alive, Tcl aborts the whole
# process with "Tcl_AsyncDelete: async handler deleted by the wrong thread"
# (preceded by "main thread is not in main loop"), killing the run and losing
# all evolution progress. Agg creates no Tcl interpreter, so this cannot happen.
# Every figure in this study is saved to disk, never shown interactively.
import matplotlib
matplotlib.use("Agg")

import os
import pickle
import logging
import neat
import gymnasium as gym
import matplotlib.pyplot as plt
from pureples.experiments.t_maze.multi_t_maze import (
    MultiTMazeEnv, save_run_gif, sample_generation_schedule,
    evaluate_all_single_switch, evaluate_double_switch, evaluate_delayed_feedback)
from pureples.experiments.t_maze import study
from pureples.shared.visualize import draw_net, plot_fitness_stats
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_adaptive_es
from pureples.es_hyperneat import AdaptiveESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

MAZE_LENGTH = 1
MAZE_LENGTH_TEXT = "1_turn" if MAZE_LENGTH == 1 else str(MAZE_LENGTH) + "_turns"

NUM_DEPLOYMENTS = 4

NUM_RUNS = 6

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
CONFIG = neat.config.Config(neat.genome.AdaptiveDefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/t_maze/config_cppn_t_maze')


def run(gens, env, version):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_adaptive_es(gens, env, 500, CONFIG, params(version), SUBSTRATE, NUM_DEPLOYMENTS,
                                    num_workers=12, schedule_sampler=sample_generation_schedule)
    print(f"adaptive_es_hyperneat_t_maze_{MAZE_LENGTH_TEXT}_{VERSION_TEXT} done")
    return winner, stats


def std_connection_an_products(net):
    """
    Yield the A*n product for every standard connection in an Adaptive ES
    phenotype network. Each node_eval is
    ``(node, activation, aggregation, bias, response, std_links, mod_links)``
    and each standard link is ``(in_id, weight, a, b, c, d, n)`` (std_links may
    be None), so ``a`` and ``n`` are at indices 2 and 6.
    """
    products = []
    for node_eval in net.node_evals:
        std_links = node_eval[5]
        if not std_links:
            continue
        for link in std_links:
            products.append(link[2] * link[6])
    return products


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = MultiTMazeEnv(MAZE_LENGTH)

    BASE_NAME = f"adaptive_es_hyperneat_t_maze_{MAZE_LENGTH_TEXT}_{VERSION_TEXT}"
    RESULTS_DIR = "pureples/experiments/t_maze/results"
    DATA_DIR = f"{RESULTS_DIR}/{BASE_NAME}_data"

    def save_standard_outputs(winner, stats, prefix, network_title, save_gif=False):
        """
        Save the files produced for every winning network -- the fitness plot,
        the phenotype network image (built with show=False so the interactive
        window is not opened here), and the CPPN drawing + pickle -- under
        ``prefix``, run the three post-evolution tests, and return the
        ``(net, cppn, test_means)`` triple for analysis.

        ``test_means`` is a ``(test1, test2, test3)`` tuple of mean per-deployment
        fitnesses:
          Test 1 -- every training-range single-switch schedule (10 deployments);
          Test 2 -- double-switch generalization over 20 episodes (6 deployments);
          Test 3 -- Test 1's schedules with delayed reward feedback (10 deployments).
        All three run post-evolution only and never affect training.

        A behaviour gif is written only when ``save_gif`` is True (single-run
        mode); multi-run studies save no gifs.
        """
        plot_fitness_stats(stats, f"{prefix}_fitness.png",
                           title=f"{network_title} — fitness over generations")

        cppn = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
        network = AdaptiveESNetwork(SUBSTRATE, cppn, params(VERSION))
        net = network.create_phenotype_network(filename=f"{prefix}_winner", show=False)

        draw_net(cppn, filename=f"{prefix}_cppn")
        with open(f"{prefix}_cppn.pkl", 'wb') as output:
            pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

        test_means = []
        for label, evaluate in (
                ("Test 1 (all single-switch schedules)", evaluate_all_single_switch),
                ("Test 2 (double-switch generalization)", evaluate_double_switch),
                ("Test 3 (delayed reward feedback)", evaluate_delayed_feedback)):
            fitnesses = evaluate(net, ENVIRONMENT)
            mean_fitness = sum(fitnesses) / len(fitnesses)
            test_means.append(mean_fitness)
            print(f"{network_title} — {label}: mean over {len(fitnesses)} deployments "
                  f"= {mean_fitness:.4f}  per-deployment: "
                  f"{[round(f, 4) for f in fitnesses]}")

        if save_gif:
            save_run_gif(net, ENVIRONMENT, f"{prefix}_run.gif", title=network_title, fps=4)
        return net, cppn, tuple(test_means)

    if NUM_RUNS <= 1:
        # Single run: keep the original interactive workflow -- save everything,
        # then pop up the interactive network window last.
        WINNER, STATS = run(MAX_GENERATIONS, ENVIRONMENT, VERSION)
        print(WINNER)
        save_standard_outputs(WINNER, STATS[0], f"{RESULTS_DIR}/{BASE_NAME}",
                              "Adaptive ES T-Maze", save_gif=True)
        plt.show()
    else:
        # Multi-run study: run NUM_RUNS evolutions, saving per-run data only for
        # runs that solve the task, then a summary across the successful runs.
        os.makedirs(DATA_DIR, exist_ok=True)
        records = []
        for run_number in range(1, NUM_RUNS + 1):
            print(f"=== Adaptive ES T-Maze: run {run_number} of {NUM_RUNS} ===")
            winner, stats_tuple = run(MAX_GENERATIONS, ENVIRONMENT, VERSION)
            stats = stats_tuple[0]

            if not study.solved(winner, CONFIG):
                print(f"Run {run_number} did not reach the fitness threshold "
                      f"within {MAX_GENERATIONS} generations; no files saved.")
                continue

            run_prefix = f"{DATA_DIR}/{BASE_NAME}_run{run_number}"
            # No gif: multi-run studies (NUM_RUNS > 1) never write .gif files.
            net, _cppn, test_means = save_standard_outputs(
                winner, stats, run_prefix, f"Adaptive ES T-Maze run {run_number}")

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
                all_switch_fitness=test_means[0],
                double_switch_fitness=test_means[1],
                delayed_feedback_fitness=test_means[2],
            )
            study.write_run_report(f"{run_prefix}_data.txt", record, MAX_GENERATIONS,
                                   title="Adaptive ES T-Maze")
            records.append(record)

            # Release the (never-shown) network figure before the next run.
            plt.close('all')

        study.write_summary(
            graph_path=f"{DATA_DIR}/{BASE_NAME}_summary_fitness.png",
            text_path=f"{DATA_DIR}/{BASE_NAME}_summary.txt",
            records=records, total_runs=NUM_RUNS,
            max_generations=MAX_GENERATIONS, title="Adaptive ES T-Maze")

        print(f"Study complete: {len(records)} of {NUM_RUNS} run(s) solved the "
              f"task and were saved to {DATA_DIR}.")