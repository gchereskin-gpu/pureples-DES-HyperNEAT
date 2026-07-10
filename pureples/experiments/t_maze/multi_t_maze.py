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
        self.reward_process_steps = 5
        self.reward = 0.0
        self.switch_reward_episode = 0

    def reset(self, seed = None, options = None):

        self.maze_points, self.high_reward, self.low_rewards = self.generate_t_maze()

        self.switched_reward_episode = random.randint(1, 8)

        self.current_position = [0.0, 0.0]
        self.rotation = 0.0

        self.current_episode = 0
        self.episode_timestep = 0

        ob = self.observe()

        self.hit_wall = False
        self.last_reward = 0.0
        self.reward = 0.0

        self.info = None

        return ob, self.info

    def step(self, action):
        """Advance the agent one step through the maze using the supplied action."""

        terminated, truncated = False, False
        reward = 0.0

        # The action is a 3-tuple: [left, forward, right].
        left, forward, right = action

        # Rotate the agent by 17 degrees for each unit of right-minus-left.
        rotation_delta = 17.0 * (right - left)
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

        # move on to next episode
        if self.episode_timestep >= 18 or high_reward or low_reward:
            # end deployment if appropriate
            if self.current_episode >= 10:
                terminated = True
            
            # reset agent
            self.current_position = [0.0, 0.0]
            self.rotation = 0.0

            # process reward
            self.reward = 1.0 if high_reward else 0.1 if low_reward else 0.0
            for _ in range(self.reward_process_steps):
                ob = self.observe()

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
            
            # penalize wall collision
            if self.hit_wall:
                reward -= 0.05
                self.hit_wall = False
            
            # switch reward sides if appropiate
            if self.current_episode == self.switch_reward_episode:
                self.maze_points, self.high_reward, self.low_rewards = self.generate_t_maze()

            self.current_episode += 1
            self.episode_timestep = 0
        else:
            ob = self.observe()
            
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