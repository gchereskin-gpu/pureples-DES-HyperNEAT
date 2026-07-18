"""
Helpers for multi-run data collection across the adaptive ES/DES T-Maze studies.

Used when an experiment's ``NUM_RUNS`` is greater than 1. Each *successful* run
(one that reaches the fitness threshold within the generation budget) records
its per-generation fitness and a handful of network statistics; once every run
has finished, a summary fitness graph and a summary text file aggregate the
successful runs. Runs that fail to solve the task contribute nothing.
"""

import os
import statistics as _stats
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class RunRecord:
    """Metrics collected from a single successful evolutionary run."""
    run_number: int
    generations_to_solution: int
    mean_fitness_per_gen: List[float]   # padded to the generation budget
    best_fitness_per_gen: List[float]   # padded to the generation budget
    mean_an_product: float              # mean of A*n over standard connections
    inhibitory_count: int               # # std connections with A*n < 0
    excitatory_count: int               # # std connections with A*n > 0
    cppn_connection_count: int
    num_branches: int = None            # # branches (Adaptive DES only; None for ES)
    # Post-evolution test means (see multi_t_maze). None until measured.
    all_switch_fitness: float = None       # Test 1: mean over all training-range single-switch schedules
    double_switch_fitness: float = None    # Test 2: mean over the double-switch generalization deployments
    delayed_feedback_fitness: float = None  # Test 3: mean over the delayed-feedback deployments

    @property
    def connection_ratio(self):
        """Inhibitory/excitatory connection ratio (inf when no excitatory connections)."""
        return _safe_ratio(self.inhibitory_count, self.excitatory_count)


def _safe_ratio(numerator, denominator):
    """``numerator/denominator``, or ``float('inf')`` when the denominator is 0."""
    return numerator / denominator if denominator else float("inf")


def solved(winner, config):
    """
    Return True when ``winner`` reached the configured fitness threshold.

    Mirrors neat's own termination test: a run is successful only when the best
    genome's fitness meets ``fitness_threshold`` (runs that exhaust the
    generation budget without doing so end with a below-threshold best genome).
    """
    return winner.fitness is not None and winner.fitness >= config.fitness_threshold


def padded_fitness_series(stats, max_generations):
    """
    Return ``(mean_per_gen, best_per_gen)`` for one run, each padded to
    ``max_generations`` entries.

    A run that solves the task in G < ``max_generations`` generations only has G
    recorded generations; the remaining entries are filled with the final
    generation's value so every run contributes a full-length series.
    """
    mean_per_gen = list(stats.get_fitness_mean())
    best_per_gen = [g.fitness for g in stats.most_fit_genomes]

    def pad(series):
        if not series:
            return [0.0] * max_generations
        if len(series) >= max_generations:
            return series[:max_generations]
        return series + [series[-1]] * (max_generations - len(series))

    return pad(mean_per_gen), pad(best_per_gen)


def connection_sign_metrics(an_products):
    """
    Given the A*n product for every standard connection, return
    ``(mean_product, inhibitory_count, excitatory_count)``.

    A connection is inhibitory when A*n < 0 and excitatory when A*n > 0
    (products of exactly 0 are counted as neither).
    """
    products = list(an_products)
    mean_product = sum(products) / len(products) if products else 0.0
    inhibitory = sum(1 for p in products if p < 0.0)
    excitatory = sum(1 for p in products if p > 0.0)
    return mean_product, inhibitory, excitatory


def count_cppn_connections(genome):
    """Number of expressed (enabled) connections in the evolved CPPN genome."""
    return sum(1 for cg in genome.connections.values() if cg.enabled)


def write_run_report(path, record, max_generations, title):
    """
    Write the per-run text report described in the study spec: the padded
    per-generation mean/best fitness series, the generations-to-solution count,
    the two A*n ratios, and the CPPN connection count.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ratio = record.connection_ratio
    ratio_text = "undefined (no excitatory connections)" if ratio == float("inf") else f"{ratio:.6f}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title} — run {record.run_number} report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Generations to reach fitness threshold: {record.generations_to_solution}\n\n")

        if (record.all_switch_fitness is not None
                or record.double_switch_fitness is not None
                or record.delayed_feedback_fitness is not None):
            f.write("Post-evolution tests (mean fitness per deployment; not seen in training):\n")
            if record.all_switch_fitness is not None:
                f.write(f"  Test 1 -- all training-range single-switch schedules: "
                        f"{record.all_switch_fitness:.6f}\n")
            if record.double_switch_fitness is not None:
                f.write(f"  Test 2 -- double-switch generalization (20 episodes): "
                        f"{record.double_switch_fitness:.6f}\n")
            if record.delayed_feedback_fitness is not None:
                f.write(f"  Test 3 -- delayed reward feedback: "
                        f"{record.delayed_feedback_fitness:.6f}\n")
            f.write("\n")

        f.write("Excitatory/inhibitory analysis (sign of A*n per standard connection):\n")
        f.write(f"  Inhibition/excitation ratio (mean of A*n over standard connections): "
                f"{record.mean_an_product:.6f}\n")
        f.write(f"  Inhibitory/excitatory connection ratio: {ratio_text}\n")
        f.write(f"    inhibitory (A*n < 0) standard connections: {record.inhibitory_count}\n")
        f.write(f"    excitatory (A*n > 0) standard connections: {record.excitatory_count}\n\n")

        f.write(f"Number of connections in the evolved CPPN: {record.cppn_connection_count}\n")
        if record.num_branches is not None:
            f.write(f"Number of branches in the network: {record.num_branches}\n")
        f.write("\n")

        f.write(f"Per-generation fitness (padded to {max_generations} generations):\n")
        f.write("generation,mean_fitness,best_fitness\n")
        for gen in range(max_generations):
            mean_v = record.mean_fitness_per_gen[gen]
            best_v = record.best_fitness_per_gen[gen]
            f.write(f"{gen},{mean_v:.6f},{best_v:.6f}\n")


def write_summary(graph_path, text_path, records, total_runs, max_generations, title):
    """
    Write the across-runs summary graph and text file from the successful runs.

    The graph plots, per generation, the mean over runs of each run's mean
    fitness (line) and the mean over runs of each run's best fitness (dots); no
    standard deviation is drawn. The text file reports the mean/std dev of the
    generations-to-solution, the run-averaged A*n ratios, and the mean CPPN
    connection count.
    """
    if not records:
        return

    generations = list(range(max_generations))
    mean_of_means = [
        _stats.mean(r.mean_fitness_per_gen[g] for r in records) for g in generations
    ]
    mean_of_bests = [
        _stats.mean(r.best_fitness_per_gen[g] for r in records) for g in generations
    ]

    # --- Summary graph -----------------------------------------------------
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(generations, mean_of_means, color="tab:blue", linewidth=1.5,
            label="Mean of per-run mean fitness")
    ax.scatter(generations, mean_of_bests, color="tab:red", s=16, zorder=3,
               label="Mean of per-run best fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{title} — fitness over {len(records)} successful run(s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(graph_path, bbox_inches="tight")
    plt.close(fig)

    # --- Summary text ------------------------------------------------------
    gens_to_solution = [r.generations_to_solution for r in records]
    gens_mean = _stats.mean(gens_to_solution)
    gens_std = _stats.stdev(gens_to_solution) if len(gens_to_solution) > 1 else 0.0

    an_ratio_mean = _stats.mean(r.mean_an_product for r in records)

    finite_conn_ratios = [r.connection_ratio for r in records
                          if r.connection_ratio != float("inf")]
    excluded = len(records) - len(finite_conn_ratios)
    conn_ratio_mean = _stats.mean(finite_conn_ratios) if finite_conn_ratios else float("inf")
    conn_ratio_text = ("undefined (no run had excitatory connections)"
                       if conn_ratio_mean == float("inf") else f"{conn_ratio_mean:.6f}")

    cppn_conn_mean = _stats.mean(r.cppn_connection_count for r in records)

    branch_counts = [r.num_branches for r in records if r.num_branches is not None]
    branch_mean = _stats.mean(branch_counts) if branch_counts else None

    def test_stats(attr):
        """(mean, std dev) of a post-evolution test metric over the runs that have it."""
        values = [getattr(r, attr) for r in records if getattr(r, attr) is not None]
        if not values:
            return None, 0.0
        return _stats.mean(values), (_stats.stdev(values) if len(values) > 1 else 0.0)

    test_summaries = [
        ("Test 1 -- all training-range single-switch schedules", "all_switch_fitness"),
        ("Test 2 -- double-switch generalization (20 episodes)", "double_switch_fitness"),
        ("Test 3 -- delayed reward feedback", "delayed_feedback_fitness"),
    ]

    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{title} — summary over successful runs\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Successful runs: {len(records)} of {total_runs}\n")
        f.write(f"  (only successful runs contribute to the statistics below)\n\n")

        f.write("Generations to find a solution:\n")
        f.write(f"  mean    = {gens_mean:.4f}\n")
        f.write(f"  std dev = {gens_std:.4f}\n\n")

        written_any_test = False
        for label, attr in test_summaries:
            mean_v, std_v = test_stats(attr)
            if mean_v is None:
                continue
            if not written_any_test:
                f.write("Post-evolution test fitness (per-run mean over that test's "
                        "deployments; never seen in training):\n")
                written_any_test = True
            f.write(f"  {label}:\n")
            f.write(f"    mean    = {mean_v:.4f}\n")
            f.write(f"    std dev = {std_v:.4f}\n")
        if written_any_test:
            f.write("\n")

        f.write("Run-averaged excitatory/inhibitory analysis:\n")
        f.write(f"  inhibition/excitation ratio (mean of per-run mean A*n): {an_ratio_mean:.6f}\n")
        f.write(f"  inhibitory/excitatory connection ratio (mean of per-run ratios): {conn_ratio_text}\n")
        if excluded:
            f.write(f"    ({excluded} run(s) excluded from the connection ratio: no excitatory connections)\n")
        f.write("\n")

        f.write(f"Mean number of connections in the evolved CPPN: {cppn_conn_mean:.4f}\n")
        if branch_mean is not None:
            f.write(f"Mean number of branches in the network: {branch_mean:.4f}\n")
        f.write("\n")

        f.write("Per-generation summary fitness (mean over successful runs; "
                "matches the summary graph):\n")
        f.write("generation,mean_of_mean_fitness,mean_of_best_fitness\n")
        for gen in generations:
            f.write(f"{gen},{mean_of_means[gen]:.6f},{mean_of_bests[gen]:.6f}\n")
