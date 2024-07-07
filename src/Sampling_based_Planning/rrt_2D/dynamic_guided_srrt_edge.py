import copy
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
        global_time=3.0,
        local_time=1.0,
        mem=100000,
        min_edge_length=4,
        obj_dir=None,
    ):
        super().__init__(
            start, end, goal_sample_rate, global_time, mem, min_edge_length
        )
        self.path = []
        self.speed = 6
        self.distance_travelled = 0
        self.obj_dir = obj_dir
        self.start_rect = None

    def run(self):
        """
        Attempts to run the algorithm to initially find a global path
        and then traverse the environment while avoiding dynamic objects
        """
        self.initial_start = self.s_start
        self.start_rect = copy.deepcopy(self.env.obs_rectangle)
        prev_coords = self.s_start.coords
        # Find initial global path without knowledge of dynamic objects
        global_path = self.planning()
        self.initial_path = global_path

        if self.obj_dir:
            self.set_dynamic_obs(self.obj_dir)

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

                if not self.utils.is_collision(Node(self.agent_pos), Node(GOAL)):
                    global_path = [self.agent_pos, self.s_goal.coords]
                    self.current_index = 0

                new_coords = self.move(global_path, self.speed)
                # If the UAV can't move to the next position
                if new_coords[0] is None:
                    # Attempt to reconnect
                    if not self.reconnect(global_path):
                        new_path = self.regrow()
                        if not new_path:
                            self.agent_positions.append(self.agent_pos)
                            return False
                        else:
                            global_path = new_path

                    self.agent_positions.append(self.agent_pos)

                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords

                    self.distance_travelled += self.utils.euclidian_distance(
                        prev_coords, new_coords
                    )
                    prev_coords = new_coords

                self.time_steps += 1
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

        # TODO: Check for collision within next x amount of time (maybe based on speed)
        future_uav_positions = []
        PREDICTION_HORIZON = 4
        for t in range(1, PREDICTION_HORIZON):
            future_pos = (
                current_pos[0] + direction[0] * mps * t,
                current_pos[1] + direction[1] * mps * t,
            )
            future_uav_positions.append(future_pos)

        for future_pos in future_uav_positions:
            for dynamic_object in self.dynamic_objects:
                dynamic_future_pos = dynamic_object.predict_future_positions(
                    PREDICTION_HORIZON
                )

                # TODO, check for future collisions
                for pos in dynamic_future_pos:
                    original_pos = dynamic_object.current_pos
                    dynamic_object.current_pos = pos

                    if self.in_dynamic_obj(Node(future_pos), dynamic_object):
                        dynamic_object.current_pos = original_pos
                        return [None, None]

                    dynamic_object.current_pos = original_pos

        return new_pos

    def invalidated_graph_after_n_steps(self, n=1):
        pass

    def regrow(self):
        """
        Regrows the tree to try and find path from current to end node.
        """
        # Clears the current tree
        self.vertex.clear()
        self.edges.clear()
        self.ellipsoid = None

        current_pos_node = Node(self.agent_pos)

        self.invalidated_nodes.clear()
        self.invalidated_edges.clear()

        self.s_start = current_pos_node
        self.vertex = [current_pos_node]
        self.edges = []

        # Attempts to find a new path from the current position
        new_path = self.planning()

        if new_path:
            self.current_index = 0
            return new_path[::-1]
        else:
            return None

    def reconnect(self, path):
        """
        Waits up to t time steps to see if waiting would clear the object
        in a faster time than replanning.
        """
        current_pos = self.agent_pos
        goal_pos = path[self.current_index + 1]

        time_steps = int(self.time)

        # Checks if its faster to wait than replan
        for t in range(1, time_steps + 1):
            collision_detected = False
            for obj in self.dynamic_objects:
                future_pos = [
                    obj.current_pos[0] + obj.velocity[0] * t,
                    obj.current_pos[1] + obj.velocity[1] * t,
                ]

                original_pos = obj.current_pos
                obj.current_pos = future_pos

                # TODO might need to change
                if self.in_dynamic_obj(Node(current_pos), obj) or self.in_dynamic_obj(
                    Node(goal_pos), obj
                ):
                    collision_detected = True

                # Restore the original position
                obj.current_pos = original_pos

                if collision_detected:
                    break

            if not collision_detected:
                return True

        return False

    def change_env(self, map_name, obj_dir=None):
        super().change_env(map_name, obj_dir)
        self.agent_positions = [self.s_start.coords]

    def plot(self):
        dynamic_objects = self.dynamic_objects

        nodelist = self.vertex
        path = self.path

        plotter = DynamicPlotting(
            self.initial_start.coords,
            self.s_goal.coords,
            dynamic_objects,
            self.time_steps,
            self.agent_positions,
            self.initial_path,
        )

        plotter.env = self.env
        plotter.obs_bound = self.env.obs_boundary
        plotter.obs_circle = self.env.obs_circle
        plotter.obs_rectangle = self.start_rect

        plotter.animation(nodelist, path, "Test", animation=True)


if __name__ == "__main__":
    start = (906, 2)
    end = (10, 505)
    goal_sample_rate = 0.05
    rrt = DynamicGuidedSRrtEdge(
        start,
        end,
        goal_sample_rate,
        global_time=3,
        obj_dir="Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json",
    )
    rrt.change_env("Evaluation/Maps/2D/block_map_25/block_23.json")

    success = rrt.run()

    print(success)

    rrt.plot()
