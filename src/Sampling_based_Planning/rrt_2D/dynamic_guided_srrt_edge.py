import copy
import os
import sys
import time

import numpy as np

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.plotting import DynamicPlotting
from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge
from rrt_2D.rrt import Node


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
    ):
        super().__init__(
            start, end, goal_sample_rate, global_time, mem, min_edge_length
        )
        self.path = []
        self.speed = 6
        self.distance_travelled = 0
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

        start_time = time.time()
        global_path = self.planning()
        end_time = time.time() - start_time

        self.compute_time = end_time

        self.initial_path = global_path

        start_time = time.time()
        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        if global_path:
            global_path = global_path[::-1]
            GOAL = global_path[-1]

            GOAL = np.array(GOAL)
            # While the final node has not been reached
            while not np.array_equal(self.agent_pos, GOAL):
                self.update_object_positions()
                self.update_world_view()

                if not self.utils.is_collision(Node(self.agent_pos), Node(GOAL)):
                    global_path = [self.agent_pos, self.s_goal.coords]
                    self.current_index = 0

                new_coords = self.move(global_path, self.speed)
                # If the UAV can't move to the next position
                if new_coords[0] is None:
                    replan_time = time.time()
                    # Attempt to reconnect
                    if not self.reconnect(global_path):
                        new_path = self.regrow()
                        replan_time = time.time() - replan_time
                        self.replan_time.append(replan_time)
                        if not new_path:
                            self.agent_positions.append(self.agent_pos)
                            return None
                        else:
                            global_path = new_path
                    replan_time = time.time() - replan_time
                    self.agent_positions.append(self.agent_pos)

                else:
                    self.agent_positions.append(new_coords)
                    self.agent_pos = new_coords

                    self.distance_travelled += self.utils.euclidian_distance(
                        prev_coords, new_coords
                    )
                    prev_coords = new_coords

                self.time_steps += 1
            self.path = global_path
            self.total_time = time.time() - start_time
            return self.agent_positions
        else:
            self.total_time = time.time() - start_time
            return None

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

                future_agent_pos = [
                    current_pos[0] + (goal_pos[0] - current_pos[0]) * (t / time_steps),
                    current_pos[1] + (goal_pos[1] - current_pos[1]) * (t / time_steps),
                ]

                seg_distance = self.utils.euclidian_distance(
                    current_pos, future_agent_pos
                )

                if (
                    self.utils.euclidian_distance(current_pos, future_pos)
                    >= seg_distance
                ):
                    break

                original_pos = obj.current_pos
                obj.current_pos = future_pos

                if self.in_dynamic_obj(
                    Node(future_agent_pos), obj
                ) or self.in_dynamic_obj(Node(goal_pos), obj):
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
        self.dobs_dir = obj_dir
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

        plotter.animation(nodelist, self.agent_positions, "Test", animation=True)


if __name__ == "__main__":
    start = (906, 2)
    end = (10, 505)
    goal_sample_rate = 0.05
    rrt = DynamicGuidedSRrtEdge(
        start,
        end,
        goal_sample_rate,
        global_time=3,
    )
    rrt.change_env(
        "Evaluation/Maps/2D/block_map_25/block_9.json",
        "Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json",
    )

    rrt.run()
    rrt.plot()
