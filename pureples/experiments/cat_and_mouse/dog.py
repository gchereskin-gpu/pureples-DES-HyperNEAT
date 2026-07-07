import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym

class DogEnv(gym.Env):
    def __init__(self, num_turns=1):
        super().__init__()
        self.ray_step = 0.2  # Distance increment for ray marching
        self.max_ray_distance = 3.0  # Maximum distance for each sensor ray
        points = set()
        for x in range(-8, 8):
            for y in range(-8, 8):
                points.add((x, y))
        self.default_room = list(points)
        self.max_timesteps = 150

    def reset(self, seed = None, options = None):

        self.room = self.generate_room()

        self.current_position = random.choice(self.room)
        self.rotation = random.uniform(0, 360)  # Random initial rotation in degrees

        self.dog_position = self.current_position
        self.dog_rotation = random.uniform(0, 360)

        self.current_timestep = 0

        ob = self.observe()

        self.info = None

        return ob, self.info

    def step(self, action):
        """Advance the agent one step through the maze using the supplied action."""

        terminated, truncated = False, False
        reward = 0.0

        # The action is a 3-tuple: [left, forward, right].
        left, right = action

        reward += 5 * (left + right)

        reward -= 5 * (abs(left - right))

        # Rotate the agent by 30 degrees for each unit of right-minus-left.
        rotation_delta = 30.0 * (right - left)
        self.rotation = (self.rotation + rotation_delta) % 360.0

        # TODO: make the movement take momentum into consideration to emulate robosphere's movement
        # Move forward in the new heading by forward/2 units.
        old_position = list(self.current_position).copy()
        heading_x = math.sin(math.radians(self.rotation))
        heading_y = math.cos(math.radians(self.rotation))
        self.current_position = [
            self.current_position[0] + heading_x * (left + right / 2.0),
            self.current_position[1] + heading_y * (left + right / 2.0),
        ]

        # rotate dog towards robot
        dx = self.current_position[0] - self.dog_position[0]
        dy = self.current_position[1] - self.dog_position[1]
        self.dog_rotation = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

        # move dog towards robot
        distance = math.hypot(dx, dy)
        if distance > 0:
            move_x = dx / distance * 0.5
            move_y = dy / distance * 0.5
            self.dog_position = [self.dog_position[0] + move_x * 0.8, self.dog_position[1] + move_y * 0.8]

        # if the dog hits a wall, place it somewhere else randomly and reward the robot
        if not self.point_in_room(self.dog_position):
            reward += 1.0
            self.dog_position = random.choice(self.room)

        # figure out if the robot hit the wall
        if not self.point_in_room(self.current_position):
            self.current_position = old_position
            hit_wall = True
            reward -= 50.0
        else:
            hit_wall = False
        
        # terminate if trial is over
        if self.current_timestep >= self.max_timesteps:
            terminated = True
        self.current_timestep += 1

        # update observation
        ob = self.observe()

        # update reward
        reward += ob[7]
        if hit_wall:
            reward -= 5.0

        return ob, reward, terminated, truncated, self.info

    def observe(self):
        """Return a 9-element observation with five distance sensors, one mode value, and a reward value."""
        ob = []

        # Cast five rays from the agent's current position.
        # The middle sensor uses angle 0, which corresponds to the +y direction.
        for angle_offset in (-90.0, -35.0, -15.0, 0.0, 15.0, 30.0, 90.0):
            ray_angle = self.rotation + angle_offset
            ray_x = math.sin(math.radians(ray_angle))
            ray_y = math.cos(math.radians(ray_angle))

            # Ray march outward in 0.05-unit increments until the ray hits the maze.
            sensed_distance = self.max_ray_distance
            for distance in [i * self.ray_step for i in range(int(self.max_ray_distance / self.ray_step) + 1)]:
                sample_x = self.current_position[0] + ray_x * distance
                sample_y = self.current_position[1] + ray_y * distance
                if not self.point_in_room([sample_x, sample_y]):
                    sensed_distance = distance
                    break

            ob.append(sensed_distance)

        # append the distance between the robot and the dog
        dog_distance = math.hypot(
            self.current_position[0] - self.dog_position[0],
            self.current_position[1] - self.dog_position[1],
        )
        ob.append(dog_distance)

        # append the direction in which the dog is relative to the robot's rotation
        dx = self.dog_position[0] - self.current_position[0]
        dy = self.dog_position[1] - self.current_position[1]
        dog_angle = math.atan2(dx, dy)
        robot_angle = math.radians(self.rotation)
        relative_angle = dog_angle - robot_angle
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        ob.append(relative_angle)

        return ob # of the format (-90.0, -35.0, -15.0, 0.0, 15.0, 30.0, 90.0, dog distance, dog direction)

    def point_in_room(self, point):
        """Return True when the point lies inside any maze-path square."""
        return any(in_bounds(point, room_point) for room_point in self.room)

    def generate_room(self):
        """
        Returns:
            room_points : list[(x, y)]
                Every coordinate in the room.
        """
        room_points = set()

        room_points = set(self.default_room)

        for _ in range(6):
            block_size = random.choice([[1, 3],[2, 2], [3, 1], [1, 1], [5, 3], [1, 5], [5, 1], [3, 5]])
            block_start = random.choice(self.default_room)
            for x in range(block_start[0], block_start[0] + block_size[0]):
                for y in range(block_start[1], block_start[1] + block_size[1]):
                    room_points.discard((x, y))

        return list(room_points)

    def visualize_step(self):
        """
        Visualizes the current room with the robot and dog positions.
        """

        room_points = getattr(self, "room", None)
        if not room_points:
            return

        xs = [p[0] for p in room_points]
        ys = [p[1] for p in room_points]

        min_x, max_x = min(xs) - 2, max(xs) + 2
        min_y, max_y = min(ys) - 2, max(ys) + 2

        fig, ax = plt.subplots(figsize=(8, 8))

        # "Gaussian space" / coordinate background
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)

        # Draw each room point as a 1x1 square centered on its coordinate
        for x, y in room_points:
            square = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="white",
                edgecolor="black"
            )
            ax.add_patch(square)

        # Mark robot and dog positions
        current_position = getattr(self, "current_position", None)
        if current_position is not None:
            ax.scatter(current_position[0], current_position[1], color="blue", s=120, label="Robot")
            rotation = getattr(self, "rotation", 0.0)
            heading_x = math.sin(math.radians(rotation))
            heading_y = math.cos(math.radians(rotation))
            ax.plot(
                [current_position[0], current_position[0] + heading_x * 0.5],
                [current_position[1], current_position[1] + heading_y * 0.5],
                color="red",
                linewidth=2,
            )

        dog_position = getattr(self, "dog_position", None)
        if dog_position is not None:
            ax.scatter(dog_position[0], dog_position[1], color="red", s=120, label="Dog")

        goal_endpoint = getattr(self, "goal_endpoint", None)
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
        ax.set_title("Current Room")
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