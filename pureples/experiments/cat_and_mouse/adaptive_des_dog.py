"""
An experiment using a variable-sized Adaptive DES-HyperNEAT network to perform a pole balancing task.
"""

import os
import pickle
import logging
import math
import neat
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym
from pureples.experiments.cat_and_mouse.dog import DogEnv
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_adaptive_des
from pureples.des_hyperneat import AdaptiveDESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

NUM_DEPLOYMENTS = 10

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = []

for i in range(7):
    x = -1.0 + (2.0 * i / 6.0)
    INPUT_COORDINATES.append((x, -1.0))
INPUT_COORDINATES.append((0., -1.2))
INPUT_COORDINATES.append((0., -1.4))

OUTPUT_COORDINATES = [(-1., 1.), (1., 1.)]
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
                            'pureples/experiments/cat_and_mouse/config_cppn_dog')


def draw_environment_frame(state, ax):
    """Render the current room state for a single frame."""
    ax.clear()

    room_points = state.get("room") if isinstance(state, dict) else getattr(state, "room", None)
    if not room_points:
        return

    xs = [p[0] for p in room_points]
    ys = [p[1] for p in room_points]

    min_x, max_x = min(xs) - 2, max(xs) + 2
    min_y, max_y = min(ys) - 2, max(ys) + 2

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)

    for x, y in room_points:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white", edgecolor="black"))

    current_position = state.get("current_position") if isinstance(state, dict) else getattr(state, "current_position", None)
    if current_position is not None:
        ax.scatter(current_position[0], current_position[1], color="blue", s=120, label="Robot")
        rotation = state.get("rotation") if isinstance(state, dict) else getattr(state, "rotation", 0.0)
        heading_x = math.sin(math.radians(rotation))
        heading_y = math.cos(math.radians(rotation))
        ax.plot(
            [current_position[0], current_position[0] + heading_x * 0.5],
            [current_position[1], current_position[1] + heading_y * 0.5],
            color="red",
            linewidth=2,
        )

    dog_position = state.get("dog_position") if isinstance(state, dict) else getattr(state, "dog_position", None)
    if dog_position is not None:
        ax.scatter(dog_position[0], dog_position[1], color="red", s=120, label="Dog")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Adaptive DES Dog Run")
    ax.legend(loc="upper right")


def run(gens, env, version):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_adaptive_des(gens, env, 500, CONFIG, params(version), SUBSTRATE, NUM_DEPLOYMENTS)
    print(f"adaptive_des_hyperneat_dog_{VERSION_TEXT} done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = DogEnv()

    # Run! Only relevant to look at the winner.
    WINNER = run(250, ENVIRONMENT, VERSION)[0]

    print(WINNER)

    # Save CPPN if wished reused and draw it + winner to file.
    CPPN = neat.nn.DesFeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = AdaptiveDESNetwork(SUBSTRATE, CPPN, params(VERSION))
    NET = NETWORK.create_phenotype_network(
        CONFIG,
        filename=f"pureples/experiments/cat_and_mouse/adaptive_des_hyperneat_dog_{VERSION_TEXT}_winner")
    draw_net(
        CPPN, filename=f"pureples/experiments/cat_and_mouse/adaptive_des_hyperneat_dog_{VERSION_TEXT}_cppn")
    with open(f'pureples/experiments/cat_and_mouse/adaptive_des_hyperneat_dog_{VERSION_TEXT}_cppn.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)

    video_path = "pureples/experiments/cat_and_mouse/adaptive_des_dog_run.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ob, _ = ENVIRONMENT.reset(seed=0)
    NET.reset()

    states = []
    step_count = 0
    for _ in range(1000):
        action = NET.activate(ob)
        ob, reward, terminated, truncated, _ = ENVIRONMENT.step(action)
        states.append({
            "room": list(ENVIRONMENT.room),
            "current_position": list(ENVIRONMENT.current_position),
            "rotation": ENVIRONMENT.rotation,
            "dog_position": list(ENVIRONMENT.dog_position),
        })
        step_count += 1

        if terminated or truncated or step_count % 200 == 0:
            ob, _ = ENVIRONMENT.reset()
            NET.reset()

    try:
        from matplotlib.animation import FuncAnimation
        from PIL import Image

        def update(frame):
            draw_environment_frame(states[frame], ax)
            return ax

        animation = FuncAnimation(fig, update, frames=len(states), interval=1000 / 8, blit=False)
        gif_path = video_path.replace(".mp4", ".gif")
        animation.save(gif_path, writer="pillow", fps=8)
        print(f"Saved animation to {gif_path}")
    except Exception as exc:
        fallback_path = video_path.replace(".mp4", ".png")
        if states:
            draw_environment_frame(states[-1], ax)
        fig.savefig(fallback_path)
        print(f"Could not save animation ({exc}); saved a still image to {fallback_path}")
    finally:
        plt.close(fig)
