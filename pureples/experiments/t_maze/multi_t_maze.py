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
        self.solved = False

    def reset(self, seed = None, options = None):

        self.maze_points, self.goal_endpoint = self.generate_t_maze(self.num_turns)

        self.current_position = [0.0, 0.0]
        self.rotation = 0.0

        self.exploration = True  # Start in exploration mode
        self.current_timestep = 0

        ob = self.observe()

        self.info = None

        return ob, self.info

    def step(self, action):
        """Advance the agent one step through the maze using the supplied action."""

        terminated, truncated = False, False
        reward = 0.0
        self.solved = False

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
        ]

        # If the new position leaves the maze path, undo the move and end the episode.
        if not self.point_in_maze(self.current_position):
            self.current_position = old_position
            hit_wall = True
        else:
            hit_wall = False

        self.current_timestep += 1

        # Exploration and navigation trials use different time limits and reset rules.
        if self.exploration:
            # in exploration trial, max number of timesteps is 4 * 8n where n is number of turns
            max_timesteps = 4 * 10 * self.num_turns + 1
            if in_bounds(self.current_position, self.goal_endpoint) or self.current_timestep >= max_timesteps:
                self.exploration = False
                self.current_position = [0.0, 0.0]
                self.rotation = 0.0
                self.current_timestep = 0
                if in_bounds(self.current_position, self.goal_endpoint):
                    self.solved = True
            reward = 0.0
        else:
            # in navigation trial, max number of timesteps is 4 * 4n where n is number of turns
            max_timesteps = 4 * 4 * self.num_turns + 3
            if self.current_timestep >= max_timesteps:
                terminated = True
            # check if agent reached end of maze
            if in_bounds(self.current_position, self.goal_endpoint):
                reward = 1.0
                terminated = True
                self.solved = True
            if hit_wall:
                reward -= 0.1  # Penalize hitting walls

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

        # The final mode value reports whether the next step belongs to exploration or navigation.
        ob.append(0.0 if self.exploration else 1.0)

        # Append the reward value.
        if self.solved:
            ob.append(1.0)
        else:
            ob.append(0.0)

        return ob

    def point_in_maze(self, point):
        """Return True when the point lies inside any maze-path square."""
        return any(in_bounds(point, maze_point) for maze_point in self.maze_points)

    def generate_t_maze(self, num_turns):
        """
        Returns:
            maze_points : list[(x, y)]
                Every coordinate in the maze.

            goal_endpoint : (x, y)
                Randomly selected endpoint from the final T-junction.
        """

        while True:  # restart if maze corners itself

            points = {(0, 0)}

            # Cardinal directions
            UP = (0, 1)
            DOWN = (0, -1)
            LEFT = (-1, 0)
            RIGHT = (1, 0)

            DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

            def add(p, d):
                return (p[0] + d[0], p[1] + d[1])

            def scale(d, k):
                return (d[0] * k, d[1] * k)

            def left_of(d):
                return (-d[1], d[0])

            def right_of(d):
                return (d[1], -d[0])

            def extend_two(start, direction):
                """
                Returns:
                    [point1, point2]
                """
                p1 = add(start, direction)
                p2 = add(p1, direction)
                return [p1, p2]

            def can_place_segment(start, direction):
                """
                Check whether the two-point extension would collide.
                """
                p1 = add(start, direction)
                p2 = add(p1, direction)

                if p1 in points:
                    return False
                if p2 in points:
                    return False

                return True

            #
            # FIRST T-JUNCTION
            #

            stem_dir = UP  # The first stem always points up.

            if not can_place_segment((0, 0), stem_dir):
                continue

            stem = extend_two((0, 0), stem_dir)

            points.update(stem)

            junction = stem[-1]

            perp1 = left_of(stem_dir)
            perp2 = right_of(stem_dir)

            if not can_place_segment(junction, perp1):
                continue

            if not can_place_segment(junction, perp2):
                continue

            arm1 = extend_two(junction, perp1)
            arm2 = extend_two(junction, perp2)

            points.update(arm1)
            points.update(arm2)

            active_endpoints = [
                (arm1[-1], perp1),
                (arm2[-1], perp2)
            ]

            #
            # REMAINING T-JUNCTIONS
            #

            for _ in range(num_turns - 1):

                possible_extensions = []

                for endpoint, endpoint_dir in active_endpoints:

                    for next_dir in (
                        left_of(endpoint_dir),
                        right_of(endpoint_dir)
                    ):

                        stem_ok = can_place_segment(endpoint, next_dir)

                        if not stem_ok:
                            continue

                        future_junction = add(
                            add(endpoint, next_dir),
                            next_dir
                        )

                        branch1 = left_of(next_dir)
                        branch2 = right_of(next_dir)

                        if not can_place_segment(
                            future_junction,
                            branch1
                        ):
                            continue

                        if not can_place_segment(
                            future_junction,
                            branch2
                        ):
                            continue

                        possible_extensions.append(
                            (endpoint, endpoint_dir, next_dir)
                        )

                #
                # No legal move => restart entire maze
                #

                if not possible_extensions:
                    break

                endpoint, endpoint_dir, next_dir = random.choice(
                    possible_extensions
                )

                stem = extend_two(endpoint, next_dir)

                points.update(stem)

                junction = stem[-1]

                branch1 = left_of(next_dir)
                branch2 = right_of(next_dir)

                arm1 = extend_two(junction, branch1)
                arm2 = extend_two(junction, branch2)

                points.update(arm1)
                points.update(arm2)

                active_endpoints = [
                    (arm1[-1], branch1),
                    (arm2[-1], branch2)
                ]

            else:
                #
                # Successfully built all turns
                #

                goal_endpoint, _ = random.choice(active_endpoints)

                return list(points), goal_endpoint


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
    if current_position[0] >= maze_point[0] + 0.5 or current_position[1] >= maze_point[1] + 0.5 or current_position[0] <= maze_point[0] - 0.5 or current_position[1] <= maze_point[1] - 0.5:
        return False

    return True