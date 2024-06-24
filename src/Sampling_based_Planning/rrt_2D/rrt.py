"""
RRT_2D
@author: huiming zhou
"""

import json
import os
import sys
import math
import numpy as np

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)
from rrt_2D import env, plotting, utils

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")


class DynamicObj:
    def __init__(self) -> None:
        self.velocity = []
        self.size = []
        self.known = False
        self.current_pos = []
        self.index = 0
        self.init_pos = None

    def update_pos(self):
        """
        TODO improve
        """
        velocity = self.velocity
        new_pos = [
            self.current_pos[0] + (velocity[0]),
            self.current_pos[1] + (velocity[1]),
        ]

        self.current_pos = new_pos
        return new_pos


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.coords = np.array(n)
        self.edge = None
        self.cost = 0
        self.time_waited = 0

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "x", None) == self.x
            and getattr(other, "y", None) == self.y
        )

    def __hash__(self):
        return hash(str(self.x) + "," + str(self.y))


class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.name = "RRT"
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env.Env()
        # self.env.x_range = (0, 1000)
        # self.env.y_range = (0, 1000)
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        # Data associated with traversal of the found path
        self.initial_path = []
        self.current_index = 0
        self.dynamic_objects = []
        self.invalidated_nodes = set()
        self.invalidated_edges = set()
        self.speed = 60
        self.time_steps = 0
        self.agent_positions = [self.s_start.coords]
        self.agent_pos = self.s_start.coords

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len and not self.utils.is_collision(
                    node_new, self.s_goal
                ):
                    self.new_state(node_new, self.s_goal)
                    return self.extract_path(node_new)

        return None

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node(
                (
                    np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                    np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                )
            )

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[
            int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y) for nd in node_list]))
        ]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node(
            (
                node_start.x + dist * math.cos(theta),
                node_start.y + dist * math.sin(theta),
            )
        )
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    def extract_path_a_to_b(self, node_start, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None and node_now is not node_start:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def change_env(self, map_name, obs_name=None):
        """
        Method which changes the env based on custom map input.
        """
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.s_start = Node(data["agent"])
            self.s_goal = Node(data["goal"])
            self.vertex = [self.s_start]

            # Initialize the new custom environment
            self.env = env.CustomEnv(data)

            # Update plotting with new environment details
            self.plotting = plotting.Plotting(data["agent"], data["goal"])
            self.plotting.env = self.env
            self.plotting.xI = data["agent"]
            self.plotting.xG = data["goal"]
            self.plotting.obs_bound = self.env.obs_boundary
            self.plotting.obs_circle = self.env.obs_circle
            self.plotting.obs_rectangle = self.env.obs_rectangle

            # Update utilities with new environment details
            self.utils = utils.Utils()
            self.utils.env = self.env
            self.utils.obs_boundary = self.env.obs_boundary
            self.utils.obs_circle = self.env.obs_circle
            self.utils.obs_rectangle = self.env.obs_rectangle

            # Update environment properties
            self.x_range = self.env.x_range
            self.y_range = self.env.y_range
            self.obs_circle = self.env.obs_circle
            self.obs_rectangle = self.env.obs_rectangle
            self.obs_boundary = self.env.obs_boundary

            if obs_name:
                # TODO
                self.set_dynamic_obs(obs_name)

        else:
            print("Error, map not found")

    def set_dynamic_obs(self, filename):
        """
        Adds dynamic objects to the environment given a JSON filename
        containing the data of the dynamic objects.
        """
        # Loads the objects
        obj_json = None
        with open(filename) as f:
            obj_json = json.load(f)

        # Adds each object to the environment
        if obj_json:
            for obj in obj_json:
                new_obj = DynamicObj()
                new_obj.velocity = obj["velocity"]
                new_obj.current_pos = obj["position"]
                new_obj.size = obj["size"]

                new_obj.index = len(self.env.obs_rectangle) - 1
                self.dynamic_objects.append(new_obj)

        else:
            print("Error, dynamic objects could not be loaded")

    def run(self):
        """
        Attempts to run the algorithm to intitially find a global path and
        then traverse the environment while avoiding dynamic objects.
        TODO.
        """
        global_path = self.planning()[::-1]
        self.initial_path = global_path

        # TODO init dynamic obs

        if global_path:
            current = global_path[self.current_index]
            GOAL = global_path[-1]

            while current != GOAL:
                current = global_path[self.current_index]


def main():
    x_start = (466, 270)
    x_goal = (967, 963)

    rrt = Rrt(x_start, x_goal, 0.5, 0.15, 10000)
    path = rrt.planning()

    if path:
        print(f"Nodes: {len(rrt.vertex)}")
        print(f"Path cost: {utils.Utils.path_cost(path)}")
        rrt.plotting.animation(rrt.vertex, path, "RRT", True)
    else:
        print("No Path Found!")


if __name__ == "__main__":
    main()
