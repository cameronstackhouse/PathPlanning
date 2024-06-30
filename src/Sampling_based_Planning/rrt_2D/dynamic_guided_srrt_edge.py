import math
import os
import sys

import numpy as np

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.plotting import DynamicPlotting
from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge
from rrt_2D.rrt import Node, DynamicObj


class DynamicGuidedSRrtEdge(MBGuidedSRrtEdge):
    def __init__(
        self,
        start,
        end,
        goal_sample_rate,
        global_time=4.0,
        local_time=0.05,
        mem=100000,
        min_edge_length=4,
    ):
        super().__init__(
            start, end, goal_sample_rate, global_time, mem, min_edge_length
        )
        self.path = []
        self.speed = 100

    def run(self):
        """
        Attempts to run the algorithm to initially find a global path
        and then traverse the environment while avoiding dynamic objects
        """
        taken_path = []  # TODO!
        # Find initial global path without knowledge of dynamic objects
        global_path = self.planning()
        self.initial_path = global_path
        self.init_dynamic_obs(1)

        if global_path:
            global_path = global_path[::-1]
            current = global_path[self.current_index]
            GOAL = global_path[-1]

            current = np.array(current)
            GOAL = np.array(GOAL)
            # While the final node has not been reached
            while not np.array_equal(current, GOAL):
                current = global_path[self.current_index]
                self.update_object_positions()
                self.update_world_view()
                new_coords = self.move(global_path, self.speed)
                # If the UAV can't move to the next position
                if new_coords[0] is None:
                    # Attempt to reconnect
                    if not self.reconnect(global_path):
                        new_path = self.regrow()
                        if not new_path:
                            return False
                        else:
                            global_path = new_path
                            self.agent_positions.append(self.agent_pos)
                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords
                self.time_steps += 1
            # TODO update
            self.path = global_path
            return True
        else:
            return False

    def move(self, path, mps=6):
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.s_goal.coords

        current_pos = self.agent_pos
        next_node = path[self.current_index + 1]

        # Checks for collision between current point and the waypoint node
        # TODO change, need to make sure object which is blocking is known
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

    def invalidated_graph_after_n_steps(self, n=1):
        pass

    def regrow(self):
        """
        TODO:
        Regrows the tree to try and find path from current to end node.
        """
        self.vertex.clear()
        self.edges.clear()
        self.ellipsoid = None
        current_pos_node = Node(self.agent_pos)

        for node in self.invalidated_nodes:
            if node in self.vertex:
                self.vertex.remove(node)

        for edge in self.invalidated_edges:
            if edge in self.edges:
                self.edges.remove(edge)

        self.invalidated_nodes.clear()
        self.invalidated_edges.clear()

        self.s_start = current_pos_node
        self.vertex = [current_pos_node]
        self.edges = []

        new_path = self.planning()

        if new_path:
            self.current_index = 0
            return new_path[::-1]
        else:
            return None

    def reconnect(self, path):
        """ """
        # TODO see if wating for one time period would clear it, AKA the edge is valid
        current_pos = self.agent_pos
        goal_pos = path[self.current_index + 1]
        # TODO go one timestep ahead
        return not self.utils.is_collision(Node(current_pos), Node(goal_pos))

    def change_env(self, map_name):
        super().change_env(map_name)

    def plot(self):
        dynamic_objects = self.dynamic_objects

        nodelist = self.vertex
        path = self.path

        plotter = DynamicPlotting(
            self.s_start.coords,
            self.s_goal.coords,
            dynamic_objects,
            self.time_steps,
            self.agent_positions,
            self.initial_path,
        )

        plotter.env = self.env
        plotter.obs_bound = self.env.obs_boundary
        plotter.obs_circle = self.env.obs_circle
        plotter.obs_rectangle = self.env.obs_rectangle

        plotter.animation(nodelist, path, "Test", animation=False)


if __name__ == "__main__":
    start = (906, 2)
    end = (10, 505)
    goal_sample_rate = 0.05
    rrt = DynamicGuidedSRrtEdge(start, end, goal_sample_rate)
    rrt.change_env("Evaluation/Maps/2D/block_map_25/0.json")

    success = rrt.run()

    rrt.plot()
