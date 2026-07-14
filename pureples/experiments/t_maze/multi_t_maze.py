import os
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym

class MultiTMazeEnv(gym.Env):
    def __init__(self, num_turns=1):
        super().__init__()
        self.num_turns = num_turns
        self.ray_step = 0.05  # Distance increment for ray marching
        self.max_ray_distance = 2.0  # Maximum distance for each sensor ray
        self.reward_process_steps = 5 #TODO: make this an argument to be set in higher level code
        self.reward = 0.0
        self.switch_reward_episode = 0
        self.reward_process_timestep = 0

        # Switch episodes already used by earlier deployments in the current
        # evaluation. Each reset() draws a switch episode not in this set so a
        # genome is tested against distinct switch timings across its
        # deployments; call reset_switch_history() to start a fresh cycle.
        self.switch_episode_range = (1, 5)
        self._used_switch_episodes = set()


    def reset(self, seed = None, options = None):

        self.maze_points, self.high_reward, self.low_rewards = self.generate_t_maze()

        self.switch_reward_episode = self._choose_switch_episode()

        self.current_position = [0.0, 0.0]
        self.rotation = 0.0

        self.current_episode = 0
        self.episode_timestep = 0
        self.reward_process_timestep = 0

        ob = self.observe()

        self.hit_wall = False
        self.last_reward = 0.0
        self.reward = 0.0

        self.info = None

        return ob, self.info

    def reset_switch_history(self):
        """
        Forget which switch episodes earlier deployments used.

        Call this at the start of each genome's deployment sequence so the
        "no repeated switch episode across deployments" guarantee applies per
        evaluation rather than globally (a single env instance is reused for
        many genomes within a worker).
        """
        self._used_switch_episodes = set()

    def _choose_switch_episode(self):
        """
        Pick the episode at which the reward arms swap, without reusing a switch
        episode already drawn by an earlier deployment in this evaluation.

        Switch episodes are drawn without replacement from
        ``switch_episode_range`` (inclusive). Once every episode in the range has
        been used, the history is cleared and a fresh cycle begins so sampling
        can continue when there are more deployments than distinct episodes.
        """
        low, high = self.switch_episode_range
        options = [e for e in range(low, high + 1)
                   if e not in self._used_switch_episodes]
        if not options:
            # Every switch episode has been used since the last history reset;
            # start a new cycle so distinct sampling can continue.
            self._used_switch_episodes = set()
            options = list(range(low, high + 1))

        choice = random.choice(options)
        self._used_switch_episodes.add(choice)
        return choice

    def step(self, action):
        """Advance the agent one step through the maze using the supplied action."""

        terminated, truncated = False, False
        reward = 0.0

        if self.reward_process_timestep == 0:
            # The action is a 3-tuple: [left, forward, right].
            left, forward, right = action

            # Rotate the agent by 17 degrees for each unit of right-minus-left.
            rotation_delta = 20.0 * (right - left)
            self.rotation = (self.rotation + rotation_delta) % 360.0

            # Move forward in the new heading by forward/4 units.
            old_position = self.current_position.copy()
            heading_x = math.sin(math.radians(self.rotation))
            heading_y = math.cos(math.radians(self.rotation))
            self.current_position = [
                self.current_position[0] + heading_x * (forward / 4.0),
                self.current_position[1] + heading_y * (forward / 4.0),
            ] #TODO: Normalize the forward so that the robot can move backwards or stop when forward is 0.5 or less

            # If the new position leaves the maze path, undo the move and end the episode.
            if not self.point_in_maze(self.current_position):
                self.current_position = old_position
                self.hit_wall = True
            else:
                self.hit_wall = False

        # check if reached end of maze
        high_reward = any(in_bounds(self.current_position, p) for p in self.high_reward)
        low_reward = any(in_bounds(self.current_position, p) for p in self.low_rewards)

        # process reward and/or end episode
        if self.episode_timestep >= 30 or high_reward or low_reward:
            # process reward
            if self.reward_process_timestep < self.reward_process_steps:
                self.reward = 1.0 if high_reward else 0.1 if low_reward else 0.0
                ob = self.observe()
                self.reward_process_timestep += 1

            # move on to next episode if done processing reward
            if self.reward_process_timestep >= self.reward_process_steps:
                # end deployment if appropriate
                if self.current_episode >= 9:
                    terminated = True
                
                # reset agent
                self.current_position = [0.0, 0.0]
                self.rotation = 0.0

                # calculate fitness
                if self.current_episode == 0:
                    self.last_reward = self.reward
                if self.current_episode > 0:
                    if self.last_reward == 1.0 and self.reward == 1.0:
                        reward = 1.0
                    elif self.last_reward == 0.1 and self.reward == 0.1:
                        reward = 0.1
                    elif self.last_reward == 0.0:
                        reward = 0.0
                        self.last_reward = self.reward
                    else:
                        reward = 0.0
                        self.last_reward = 0.0
                self.reward = 0.0

                # penalize wall collision
                if self.hit_wall:
                    reward -= 0.05
                    self.hit_wall = False
                
                # switch reward sides if appropriate (always flip to the opposite arm)
                if self.current_episode == self.switch_reward_episode:
                    self.high_reward, self.low_rewards = self.low_rewards, self.high_reward #TODO: This will not work as intended if using longer than a single t-maze

                self.current_episode += 1
                self.episode_timestep = 0
                self.reward_process_timestep = 0
        else:
            ob = self.observe()
            self.episode_timestep += 1
            
        return ob, reward, terminated, truncated, self.info


    def observe(self):
        """Return a 7-element observation with five distance sensors, one mode value, and a reward value."""
        ob = []

        # Cast five rays from the agent's current position.
        # The middle sensor uses angle 0, which corresponds to the +y direction.
        for angle_offset in (-90.0, -45.0, 0.0, 45.0, 90.0):
            ray_angle = self.rotation + angle_offset
            ray_x = math.sin(math.radians(ray_angle))
            ray_y = math.cos(math.radians(ray_angle))

            # Ray march outward in 0.05-unit increments until the ray hits the maze.
            sensed_distance = self.max_ray_distance
            for distance in [i * self.ray_step for i in range(int(self.max_ray_distance / self.ray_step) + 1)]:
                sample_x = self.current_position[0] + ray_x * distance
                sample_y = self.current_position[1] + ray_y * distance
                if not self.point_in_maze([sample_x, sample_y]):
                    sensed_distance = distance
                    break

            ob.append(sensed_distance)

        # Append the reward value.
        ob.append(self.reward)

        return ob

    def point_in_maze(self, point):
        """Return True when the point lies inside any maze-path square."""
        return any(in_bounds(point, maze_point) for maze_point in self.maze_points)

    def generate_t_maze(self):
        """
        Returns a single, static T-maze.

        The maze is a stem rising from the origin to a T-junction, with a left
        arm and a right arm. One of two reward layouts is chosen at random:
        high reward on the left (low on the right), or high reward on the right
        (low on the left).

        Returns:
            maze_points : list[(x, y)]
                The centers of the seven 1x1 blocks making up the maze.

            high_reward : list[(x, y)]
                A single coordinate for the high-reward block (coincides with
                one arm's endpoint).

            low_rewards : list[(x, y)]
                A single coordinate for the low-reward block on the opposite arm.
        """

        # Stem (0,0)->(0,1)->(0,2), then left arm and right arm off the junction.
        maze_points = [
            (0, 0),
            (0, 1),
            (0, 2),
            (-1, 2),
            (-2, 2),
            (1, 2),
            (2, 2),
        ]

        left_end = (-2, 2)
        right_end = (2, 2)

        if random.random() < 0.5:
            # High reward on the left, low reward on the right.
            high_reward = [left_end]
            low_rewards = [right_end]
        else:
            # High reward on the right, low reward on the left.
            high_reward = [right_end]
            low_rewards = [left_end]

        return maze_points, high_reward, low_rewards


    def visualize_maze(self, maze_points, goal_endpoint=None):
        """
        Visualizes maze points as 1x1 white squares centered at each coordinate.
        """

        xs = [p[0] for p in maze_points]
        ys = [p[1] for p in maze_points]

        min_x, max_x = min(xs) - 2, max(xs) + 2
        min_y, max_y = min(ys) - 2, max(ys) + 2

        fig, ax = plt.subplots(figsize=(8, 8))

        # "Gaussian space" / coordinate background
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)

        # Draw each maze point as a 1x1 square centered on its coordinate
        for x, y in maze_points:
            square = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="white",
                edgecolor="black"
            )
            ax.add_patch(square)

        # Mark start
        ax.scatter(0, 0, s=100, marker="o", label="Start")

        # Mark goal, if given
        if goal_endpoint is not None:
            ax.scatter(
                goal_endpoint[0],
                goal_endpoint[1],
                s=120,
                marker="*",
                label="Goal"
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Generated Multiple T-Maze")
        ax.legend()

        plt.show()

def in_bounds(current_position, maze_point):
    """
    Check if the current position is within the bounds (of +-0.5 in x and y directions) of the given maze point.

    Returns T/F
    """
    return (
        maze_point[0] - 0.5 < current_position[0] < maze_point[0] + 0.5
        and maze_point[1] - 0.5 < current_position[1] < maze_point[1] + 0.5
    )


def draw_tmaze_frame(state, ax, title):
    """Render a single captured T-maze state onto ``ax`` (one animation frame)."""
    ax.clear()

    maze_points = state["maze"]
    xs = [p[0] for p in maze_points]
    ys = [p[1] for p in maze_points]
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)

    # Maze blocks.
    for x, y in maze_points:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white", edgecolor="black"))

    # Reward locations: green star = high reward, orange star = low reward.
    # These swap arms mid-deployment, so watching the agent track the green star
    # is what verifies it adapts after the switch.
    for x, y in state["high_reward"]:
        ax.scatter(x, y, color="green", s=220, marker="*", zorder=3, label="High reward")
    for x, y in state["low_rewards"]:
        ax.scatter(x, y, color="orange", s=140, marker="*", zorder=3, label="Low reward")

    # Agent position and heading.
    px, py = state["current_position"]
    ax.scatter(px, py, color="blue", s=120, zorder=4, label="Agent")
    heading_x = math.sin(math.radians(state["rotation"]))
    heading_y = math.cos(math.radians(state["rotation"]))
    ax.plot([px, px + heading_x * 0.5], [py, py + heading_y * 0.5],
            color="red", linewidth=2, zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} — episode {state['episode']}")
    ax.legend(loc="upper right")


def save_run_gif(net, env, gif_path, title, fps=4, max_steps=600):
    """
    Roll out ``net`` on ``env`` for a single deployment and save a gif of the run.

    Uses one activation per step, matching how the adaptive networks are
    evaluated during evolution. Each step captures the maze, the agent's pose,
    and the (switching) reward locations so the whole deployment -- including the
    behaviour across the mid-deployment reward switch -- is visible.
    """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    ob, _ = env.reset()
    net.reset()

    states = []
    for _ in range(max_steps):
        ob, _reward, terminated, truncated, _ = env.step(net.activate(ob))
        states.append({
            "maze": list(env.maze_points),
            "current_position": list(env.current_position),
            "rotation": env.rotation,
            "high_reward": list(env.high_reward),
            "low_rewards": list(env.low_rewards),
            "episode": env.current_episode,
        })
        if terminated or truncated:
            break

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        from matplotlib.animation import FuncAnimation

        def update(frame):
            draw_tmaze_frame(states[frame], ax, title)
            return ax

        animation = FuncAnimation(fig, update, frames=len(states), interval=1000 / fps, blit=False)
        animation.save(gif_path, writer="pillow", fps=fps)
        print(f"Saved animation to {gif_path}")
    except Exception as exc:  # visualization is best-effort; fall back to a still
        fallback_path = gif_path.replace(".gif", ".png")
        if states:
            draw_tmaze_frame(states[-1], ax, title)
        fig.savefig(fallback_path)
        print(f"Could not save animation ({exc}); saved a still image to {fallback_path}")
    finally:
        plt.close(fig)