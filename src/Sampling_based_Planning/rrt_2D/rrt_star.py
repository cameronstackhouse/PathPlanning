"""
RRT_star 2D
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

from rrt_2D import env, plotting, utils, queue


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


class RrtStar:
    def __init__(
        self, x_start, x_goal, step_len, goal_sample_rate, search_radius, iter_max
    ):
        self.name = "RRT*"
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []
        self.edges = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
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
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

        index = self.search_goal_parent()
        self.path = self.extract_path(self.vertex[index])
        return self.path

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node(
            (
                node_start.x + dist * math.cos(theta),
                node_start.y + dist * math.sin(theta),
            )
        )

        node_new.parent = node_start

        return node_new

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_list = [
            math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex
        ]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        if len(node_index) > 0:
            cost_list = [
                dist_list[i] + self.cost(self.vertex[i])
                for i in node_index
                if not self.utils.is_collision(self.vertex[i], self.s_goal)
            ]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

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

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [
            math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex
        ]
        dist_table_index = [
            ind
            for ind in range(len(dist_table))
            if dist_table[ind] <= r
            and not self.utils.is_collision(node_new, self.vertex[ind])
        ]

        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[
            int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y) for nd in node_list]))
        ]

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def update_cost(self, parent_node):
        OPEN = queue.QueueFIFO()
        OPEN.put(parent_node)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.child) == 0:
                continue

            for node_c in node.child:
                node_c.Cost = self.get_new_cost(node, node_c)
                OPEN.put(node_c)

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

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

            self.agent_pos = data["agent"]

            # Add dynamic obs if needed
            if obs_name:
                self.set_dynamic_obs(obs_name)

        else:
            print("Error, map not found")

    def move(self, path, mps=6) -> bool:
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.s_goal.coords

        current_pos = self.agent_pos
        next_node = path[self.current_index + 1]

        # Checks for collision between current point and the waypoint node
        # TODO need to make a one-step-ahead check Might need to change based on implementation
        if self.utils.is_collision(Node(current_pos), Node(next_node)):
            return [None, None]

        seg_distance = self.utils.euclidian_distance(current_pos, next_node)

        direction = (
            (next_node[0] - current_pos[0]) / seg_distance,
            (next_node[1] - current_pos[1]) / seg_distance,
        )

        new_pos = (
            current_pos[0] + direction[0] * mps,
            current_pos[1] + direction[1] * mps,
        )

        # Checks for overshoot
        if self.utils.euclidian_distance(current_pos, new_pos) >= seg_distance:
            self.agent_pos = next_node
            self.current_index += 1
            return next_node

        return new_pos

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

    def init_dynamic_obs(self, n_obs):
        """
        TODO, tidy
        """
        for _ in range(n_obs):
            new_obj = DynamicObj()
            new_obj.velocity = [
                1000,
                1000,
            ]
            new_obj.size = [50, 50]
            new_obj.current_pos = [707, 610]
            new_obj.init_pos = new_obj.current_pos

            self.env.add_rect(
                new_obj.current_pos[0],
                new_obj.current_pos[1],
                new_obj.size[0],
                new_obj.size[1],
            )

            # TODO THIS IS THE KEY!
            self.utils.env.add_rect(
                new_obj.current_pos[0],
                new_obj.current_pos[1],
                new_obj.size[0],
                new_obj.size[1],
            )

            new_obj.index = len(self.env.obs_rectangle) - 1
            self.dynamic_objects.append(new_obj)

        new_obj = DynamicObj()
        new_obj.velocity = [
            1000,
            1000,
        ]
        new_obj.size = [100, 100]
        new_obj.current_pos = [787, 635]
        new_obj.init_pos = new_obj.current_pos

        self.env.add_rect(
            new_obj.current_pos[0],
            new_obj.current_pos[1],
            new_obj.size[0],
            new_obj.size[1],
        )

        self.utils.env.add_rect(
            new_obj.current_pos[0],
            new_obj.current_pos[1],
            new_obj.size[0],
            new_obj.size[1],
        )

        new_obj.index = len(self.env.obs_rectangle) - 1
        self.dynamic_objects.append(new_obj)

    def in_dynamic_obj(self, node, obj):
        """
        TODO
        """
        x, y = node.coords
        x0, y0 = obj.current_pos
        width, height = obj.size
        return (x0 <= x < x0 + width) and (y0 <= y < y0 + height)

    def invalidate(self, obj):
        """
        Invalidates nodes and edges which have been blocked by a dynamic object.

        """
        # Check if nodes exist lie within the current object
        for node in self.vertex:
            if self.in_dynamic_obj(node, obj):
                self.invalidated_nodes.add(node)

        # Check if any part of an edge lies within the object
        for edge in self.edges:
            n1 = edge.node_1
            n2 = edge.node_2

            if self.utils.is_collision(n1, n2):
                self.invalidated_edges.add(edge)

    def revalidate(self):
        """
        Assesses invalidated nodes and edges to see if they are no longer invalidated by
        dynamic objects.
        """
        # Check nodes
        nodes_to_remove = []
        for node in self.invalidated_nodes:
            # Check if a node is still inside one of the dynamic objects
            blocked = False
            for object in self.dynamic_objects:
                if self.in_dynamic_obj(node, object):
                    blocked = True
                    break

            if not blocked:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.invalidated_nodes.remove(node)

        # Check edges
        edges_to_remove = []
        for edge in self.invalidated_edges:
            n1 = edge.node_1
            n2 = edge.node_2

            if not self.utils.is_collision(n1, n2):
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            self.invalidated_edges.remove((edge))

    def update_object_positions(self, time_steps=1):
        """
        Updates the position of dynamic objects over one timestep.
        The object moves in a fixed direction and comes to an immediate stop
        if a fixed object is detected.

        :param: time_steps: the number of time steps in the future to predict the object positions.
        """
        for object in self.dynamic_objects:
            # Attempt to move in direction of travel
            new_pos = object.update_pos()

            self.env.update_obj_pos(object.index, new_pos[0], new_pos[1])
            self.utils.env.update_obj_pos(object.index, new_pos[0], new_pos[1])

    def update_world_view(self):
        """
        Discovers dynamic objects within the vicinity of the UAV and updates
        validated and invalidated edges.
        """
        self.revalidate()
        VISION = 30
        pos = self.agent_pos

        for obj in self.dynamic_objects:
            self.invalidate(obj)
            obj_pos, obj_size = obj.current_pos, obj.size
            obj_center_x = obj_pos[0]
            obj_center_y = obj_pos[1]
            obj_half_height = obj_size[0] / 2
            obj_half_width = obj_size[1] / 2

            # Calculate the distance from the UAV to the edges of the object's bounding box
            nearest_x = max(
                obj_center_x - obj_half_width,
                min(pos[0], obj_center_x + obj_half_width),
            )
            nearest_y = max(
                obj_center_y - obj_half_height,
                min(pos[1], obj_center_y + obj_half_height),
            )

            distance = math.sqrt((nearest_x - pos[0]) ** 2 + (nearest_y - pos[1]) ** 2)

            # Check if the object is within the UAV's vision radius
            if distance <= VISION:
                obj.known = True
            else:
                obj.known = False

    def run(self):
        """
        Attempts to run the algorithm to intitially find a global path and
        then traverse the environment while avoiding dynamic objects.
        TODO.
        """
        global_path = self.planning()[::-1]
        self.initial_path = global_path

        # self.init_dynamic_obs(1)

        if global_path:
            current = global_path[self.current_index]
            GOAL = global_path[-1]

            while current != GOAL:
                print(self.agent_pos)
                
                current = global_path[self.current_index]
                self.update_object_positions()
                self.update_world_view()
                new_coords = self.move(global_path, self.speed)

                if new_coords == [None, None]:
                    # Rerun rrt from the current position
                    self.vertex = [Node(current)]
                    self.s_start = Node(current)
                    
                    global_path = self.planning()[::-1]
                    self.current_index = 0
                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords
                self.time_steps += 1

            self.path = global_path
            return global_path
        else:
            return None


def main():
    x_start = (18, 8)  # Starting node
    x_goal = (37, 18)  # Goal node

    rrt_star = RrtStar(x_start, x_goal, 5, 0.10, 20, 5000)
    rrt_star.change_env("Evaluation/Maps/2D/block_map_25/20.json")
    path = rrt_star.planning() # .run()

    if path:
        print(f"Number of nodes: {len(rrt_star.vertex)}")
        print(f"Path length: {rrt_star.utils.path_cost(path)}")

        rrt_star.plotting.animation(
            rrt_star.vertex, rrt_star.path, "rrt*, N = " + str(rrt_star.iter_max)
        )


if __name__ == "__main__":
    main()
