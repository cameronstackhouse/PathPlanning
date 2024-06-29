import os
import sys
import math
import numpy as np

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "../Sampling_based_Planning/"
)

from rrt import Node, Rrt, utils


class Edge:
    """
    An edge in the tree
    """

    def __init__(self, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2

    def __eq__(self, other):
        return isinstance(other, self.__class__) and (
            (
                getattr(other, "node_1", None) == self.node_1
                and getattr(other, "node_2", None) == self.node_2
            )
            or (
                getattr(other, "node_1", None) == self.node_2
                and getattr(other, "node_2", None) == self.node_1
            )
        )

    def __hash__(self):
        return hash(
            str(self.node_1.x)
            + ","
            + str(self.node_1.y)
            + ":"
            + str(self.node_2.x)
            + ","
            + str(self.node_2.y)
        )


class RrtEdge(Rrt):
    """
    Modified implementation of RRT-Edge based on the 2017 paper by Correia et al.
    https://ieeexplore.ieee.org/abstract/document/8215282.

    Key features of the algorithm involve a completley unbounded edge length alongside
    """

    def __init__(self, start, end, goal_sample_rate, iter_max, min_edge_length=4):
        super().__init__(start, end, float("inf"), goal_sample_rate, iter_max)
        self.name = "RRT-Edge"
        self.edges = []
        self.min_edge_length = min_edge_length

        self.env.x_range = (0, 1000)
        self.env.y_range = (0, 1000)

    def planning(self):
        b_path = None
        path_cost = float("inf")
        for _ in range(self.iter_max):
            node_rand = self.generate_random_node()
            node_near = self.nearest_neighbour(self.vertex, self.edges, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.edges.append(Edge(node_near, node_new))

                if node_new.edge is not None:
                    self.split(node_new)

                if not self.utils.is_collision(node_new, self.s_goal):
                    final_node = self.new_state(node_new, self.s_goal)
                    if final_node and not self.utils.is_collision(node_new, final_node):
                        path = self.extract_path(final_node)
                        cost = utils.Utils.path_cost(path)
                        if cost < path_cost:
                            b_path = path
                            path_cost = cost

        return b_path

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        # Scale up the distance if it's shorter than the minimum edge length
        if dist < self.min_edge_length:
            dist = self.min_edge_length

        dist = min(self.step_len, dist)
        node_new = Node(
            (
                node_start.x + dist * math.cos(theta),
                node_start.y + dist * math.sin(theta),
            )
        )
        node_new.parent = node_start

        if (
            node_new.x > self.x_range[1]
            or node_new.x < self.x_range[0]
            or node_new.y > self.y_range[1]
            or node_new.y < self.y_range[0]
        ):
            return None

        return node_new

    def generate_random_node(self):
        delta = self.utils.delta
        return Node(
            (
                np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
            )
        )

    def nearest_neighbour(self, node_list, edge_list, n):
        nearest_node = Rrt.nearest_neighbor(node_list, n)
        nearest_edge_dist, nearest_edge_proj, nearest_edge = (
            self.nearest_edge_projection(edge_list, n)
        )

        node_dist = math.hypot(nearest_node.x - n.x, nearest_node.y - n.y)

        if nearest_edge_proj is not None and nearest_edge_dist < node_dist:
            new_node = Node(nearest_edge_proj)
            new_node.edge = nearest_edge
            new_node.parent = nearest_edge.node_1
            return new_node
        else:
            return nearest_node

    def nearest_edge_projection(self, edge_list, n):
        """
        TODO description
        """
        min_distance = float("inf")
        proj = None
        nearest_edge = None
        for edge in edge_list:
            proj_node_coords = self.orthogonal_projection(edge, n)
            if proj_node_coords is not None:
                distance = math.hypot(
                    proj_node_coords[0] - n.x, proj_node_coords[1] - n.y
                )
                if distance < min_distance:
                    min_distance = distance
                    proj = proj_node_coords
                    nearest_edge = edge

        return min_distance, proj, nearest_edge

    @staticmethod
    def orthogonal_projection(edge, new_node):
        P1 = np.array([edge.node_1.x, edge.node_1.y])
        P2 = np.array([edge.node_2.x, edge.node_2.y])
        A = np.array([new_node.x, new_node.y])

        B = P2 - P1
        B_norm_sq = np.dot(B, B)

        A_shifted = A - P1

        P_A = (np.dot(A_shifted, B) / B_norm_sq) * B

        # If the projection does not lie on the edge
        if np.dot(P_A, B) < 0 or np.dot(P_A, B) > B_norm_sq:
            return None

        proj_coords = P_A + P1

        return proj_coords

    def split(self, node):
        """
        Splits the edge that the node is on into two distinct edges
        """
        edge = node.edge
        node.parent = edge.node_1
        edge.node_2.parent = node
        self.edges.remove(edge)
        self.edges.append(Edge(edge.node_1, node))
        self.edges.append(Edge(node, edge.node_2))
        node.edge = None

    def change_env(self, map_name):
        super().change_env(map_name)
        self.edges = []


def main():
    x_start = (2, 2)
    x_goal = (82, 77)

    rrt_edge = RrtEdge(x_start, x_goal, 0.05, 2000)
    path = rrt_edge.planning()

    if path:
        print(f"Number of nodes: {len(rrt_edge.vertex)}")
        print(f"Path length: {utils.Utils.path_cost(path)}")
        rrt_edge.plotting.animation(rrt_edge.vertex, path, "RRT-Edge", True)
    else:
        print("No Path Found!")


if __name__ == "__main__":
    main()
