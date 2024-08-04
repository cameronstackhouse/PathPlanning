import cProfile
import json
import os
import pstats
import sys
import time
import psutil

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import utils
from rrt_2D.rrt import Node
from rrt_2D.rrt_edge import Edge
from rrt_2D.guided_srrt_edge import GuidedSRrtEdge


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")
from load_map import create_custom_env


class MBGuidedSRrtEdge(GuidedSRrtEdge):
    def __init__(
        self, start, end, goal_sample_rate, time=0.05, mem=100000, min_edge_length=4
    ):
        super().__init__(start, end, goal_sample_rate, float("inf"), min_edge_length)
        self.name = "MB-SRRT-Edge-" + str(time)
        self.mem = mem
        self.time = time
        self.peak_cpu = 0
        self.dobs_dir = None

    def clear_tree(self):
        self.vertex = [self.s_start]
        self.edges = []

    def update_ellipsoid(self, path):
        super().update_ellipsoid(path)
        self.clear_tree()

    def planning(self):
        b_path = None
        path_cost = float("inf")
        start_time = time.time()
        while True:
            cpu_usage = psutil.cpu_percent(interval=None)
            self.peak_cpu = max(self.peak_cpu, cpu_usage)
            elapsed_time = time.time() - start_time

            process = psutil.Process(os.getpid())
            memory_usage = (process.memory_info().rss) / (1024 * 1024)

            if elapsed_time > self.time or memory_usage > self.mem:
                break

            node_rand = self.generate_random_node()
            node_near = self.nearest_neighbour(self.vertex, self.edges, node_rand)
            node_new = self.new_state(node_near, node_rand)

            # Checks if the new node can be added to the tree without a collision
            if node_new and not self.utils.is_collision(node_near, node_new):
                # Adds the node to the tree and stores the newly created edge
                self.vertex.append(node_new)
                new_edge = Edge(node_near, node_new)
                self.edges.append(new_edge)

                if not self.utils.is_collision(node_new, self.s_goal):
                    final_node = self.new_state(node_new, self.s_goal)
                    if final_node and not self.utils.is_collision(node_new, final_node):
                        path = self.extract_path(final_node)
                        cost = self.utils.path_cost(path)
                        if cost < path_cost:
                            if self.first_success is None:
                                self.first_success = time.time() - start_time
                            b_path = path
                            path_cost = cost
                            self.update_ellipsoid(path)

                # Checks for direct paths from points along the added edge to the goal
                k = self.calculate_k(new_edge)
                partition_points = self.get_k_partitions(k, new_edge)

                # Checks for a direct path from each line partition to the goal
                for point in partition_points:
                    # Creates a new node
                    point_node = Node(point)
                    point_node.parent = new_edge.node_1
                    point_node.edge = new_edge
                    # Checks if there is a collision free path from the current point to the goal
                    if not self.utils.is_collision(point_node, self.s_goal):
                        final_node = self.new_state(point_node, self.s_goal)
                        if final_node and not self.utils.is_collision(
                            point_node, final_node
                        ):
                            path = self.extract_path(final_node)
                            cost = self.utils.path_cost(path)
                            if cost < path_cost:
                                if self.first_success is None:
                                    self.first_success = time.time() - start_time
                                b_path = path
                                path_cost = cost
                                self.update_ellipsoid(path)

        return b_path

    def change_env(self, map_name, obj_dir=None):
        super().change_env(map_name, obj_dir)
        self.ellipsoid = None


def main():
    srrt_edge = MBGuidedSRrtEdge((0, 0), (0, 0), 0.0, 5)

    srrt_edge.change_env("Evaluation/Maps/2D/block_map_250/200.json")
    
    globals_ = globals().copy() 
    locals_ = locals() 
    
    globals_['srrt_edge'] = srrt_edge
    
    cProfile.runctx('srrt_edge.planning()', globals_, locals_, 'profile_output.prof')    
    
    stats = pstats.Stats("profile_output.prof")
    stats.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(10)


    # print(srrt_edge.peak_cpu)

    # if path:
    #     print(f"Number of nodes: {len(srrt_edge.vertex)}")
    #     print(f"Path length: {srrt_edge.utils.path_cost(path)}")

    #     srrt_edge.plotting.animation(
    #         srrt_edge.vertex, path, "Bounded Guided SRRT-Edge", False
    #     )
    # else:
    #     print("No Path Found!")
    #     srrt_edge.plotting.animation(
    #         srrt_edge.vertex, [], "Bounded Guided SRRT-Edge", False
    #     )
    #     # pass


if __name__ == "__main__":
    main()
