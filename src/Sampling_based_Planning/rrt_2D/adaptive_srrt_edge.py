import math
import os
import sys
import time

import psutil


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import utils
from rrt_2D.dynamic_guided_srrt_edge import DynamicGuidedSRrtEdge
from rrt_2D.rrt import Node
from rrt_2D.rrt_edge import Edge
from rrt_2D.srrt_edge import SRrtEdge


class AdaptiveSRRTEdge(DynamicGuidedSRrtEdge):
    def __init__(
        self,
        start,
        end,
        goal_sample_rate,
        global_time=3,
        local_time=1,
        mem=100000,
        min_edge_length=1,
        x=10,
        t=25,
    ):
        super().__init__(
            start, end, goal_sample_rate, global_time, local_time, mem, min_edge_length
        )
        self.reject_count = 0
        self.b_path = None
        self.path_cost = float("inf")
        self.T = t
        self.x = x

    def steer_collision_free(self, node_new, node_near):
        dist, theta = self.get_distance_and_angle(node_near, node_new)

        dist = min(self.step_len, dist)
        node_new = Node(
            (node_near.x + dist * math.cos(theta), node_near.y + dist * math.cos(theta))
        )

        node_new.parent = node_near

        if not self.utils.is_collision(node_near, node_new):
            return node_new
        else:
            return None

    def accept_sample(self, node_new, node_near):
        # Adds the node to the tree and stores the newly created edge
        self.vertex.append(node_new)
        new_edge = Edge(node_near, node_new)
        self.edges.append(new_edge)

        if not self.utils.is_collision(node_new, self.s_goal):
            final_node = self.new_state(node_new, self.s_goal)
            if final_node and not self.utils.is_collision(node_new, final_node):
                path = self.extract_path(final_node)
                cost = utils.Utils.path_cost(path)
                if cost < self.path_cost:
                    self.b_path = path
                    self.path_cost = cost
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
                if final_node and not self.utils.is_collision(point_node, final_node):
                    path = self.extract_path(final_node)
                    cost = utils.Utils.path_cost(path)
                    if cost < self.path_cost:
                        self.b_path = path
                        self.path_cost = cost
                        self.update_ellipsoid(path)

    def planning(self):
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
                self.accept_sample(node_new, node_near)
                self.reject_count = 0

                if self.step_len + self.x > self.env.x_range[1]:
                    self.step_len = float("inf")
                else:
                    self.step_len += self.x
            elif node_new:
                node = self.steer_collision_free(node_new, node_near)

                if node:
                    self.accept_sample(node, node_near)
                    self.reject_count = 0

                    if self.step_len + self.x > self.env.x_range[1]:
                        self.step_len = float("inf")
                    else:
                        self.step_len += self.x
                else:
                    self.reject_count += 1
                    if self.reject_count == self.T:
                        if self.step_len == float("inf"):
                            self.step_len = self.env.x_range[1] - self.x
                        elif self.step_len - self.x > 0:
                            self.step_len -= self.x
                        else:
                            self.step_len = 1
            else:
                self.reject_count += 1
                if self.reject_count == self.T:
                    if self.step_len == float("inf"):
                        self.step_len = self.env.x_range[1] - self.x
                    elif self.step_len - self.x > 0:
                        self.step_len -= self.x
                    else:
                        self.step_len = 1

        return self.b_path


if __name__ == "__main__":
    start = (906, 2)
    end = (10, 505)
    goal_sample_rate = 0.05
    a = AdaptiveSRRTEdge(start, end, goal_sample_rate, 5, 5, 100000, 1, 100, 1)

    a.change_env(
        "Evaluation/Maps/2D/main/house_12.json",
    )

    a.run()
    a.plot()
