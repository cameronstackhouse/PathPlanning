import math
import os
import sys

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
        global_time=2.0,
        local_time=0.05,
        mem=100000,
        min_edge_length=4,
    ):
        super().__init__(
            start, end, goal_sample_rate, global_time, mem, min_edge_length
        )
        self.path = []

    def run(self):
        """
        Attempts to run the algorithm to initially find a global path
        and then traverse the environment while avoiding dynamic objects
        """
        taken_path = [] # TODO!
        # Find initial global path without knowledge of dynamic objects
        global_path = self.planning()[::-1]
        self.initial_path = global_path
        self.init_dynamic_obs(1)

        if global_path:
            current = global_path[self.current_index]
            GOAL = global_path[-1]
            # While the final node has not been reached
            while current != GOAL:
                # print(
                #     f"Timestep: {self.time_steps}\ninvlaidated edges: {len(self.invalidated_edges)}\nInvalidated nodes: {len(self.invalidated_nodes)}"
                # )
                current = global_path[self.current_index]
                self.update_object_positions()
                self.update_world_view()
                new_coords = self.move(global_path, self.speed)
                # If the UAV can't move to the next position
                if new_coords == [None, None]:
                    # Attempt to reconnect
                    if not self.reconnect():
                        new_path = self.regrow()
                        if not new_path:
                            return False
                        else:
                            global_path = new_path[::-1]
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

    def invalidated_graph_after_n_steps(self, n=1):
        pass

    def regrow(self):
        """
        TODO:
        Regrows the tree to try and find path from current to end node.
        """
        self.vertex.clear()
        self.edges.clear()
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
            return new_path
        else:
            return None

    def reconnect(self):
        """ """
        # TODO see if wating for one time period would clear it, AKA the edge is valid
        current_pos = self.agent_pos
        goal_pos = self.initial_path[self.current_index + 1]
        # TODO go one timestep ahead
        return not self.utils.is_collision(Node(current_pos), Node(goal_pos))


if __name__ == "__main__":
    start = (582, 230)
    end = (901, 900)
    goal_sample_rate = 0.05
    rrt = DynamicGuidedSRrtEdge(start, end, goal_sample_rate)

    success = rrt.run()

    dynamic_objects = rrt.dynamic_objects
    nodelist = rrt.vertex
    path = rrt.path

    plotter = DynamicPlotting(
        start,
        end,
        dynamic_objects,
        rrt.time_steps,
        rrt.agent_positions,
        rrt.initial_path,
    )

    plotter.env = rrt.env

    plotter.animation(nodelist, path, "Test", animation=False)
