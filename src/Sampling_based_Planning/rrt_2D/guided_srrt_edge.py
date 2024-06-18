import os
import sys
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import utils
from rrt_2D.rrt import Node
from rrt_2D.rrt_edge import Edge
from rrt_2D.srrt_edge import SRrtEdge


class GuidedSRrtEdge(SRrtEdge):
    """
    Ellipsoid-guided SRRT with dynamic adjustment of sampling space.
    """

    def __init__(self, start, end, goal_sample_rate, iter_max, min_edge_length=4):
        super().__init__(start, end, goal_sample_rate, iter_max, min_edge_length)
        self.ellipsoid = None

    def update_ellipsoid(self, path):
        """
        Update the parameters of the ellipsoid based on the best path found.
        """

        x1, y1 = path[0]
        x2, y2 = path[-1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        angle = np.arctan2(y2 - y1, x2 - x1)

        # Calculate the semi-major axis length
        semi_major_axis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2

        max_distance = max(
            np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in path
        )

        # Set the semi-minor axis length
        semi_minor_axis = max_distance

        self.ellipsoid = (center_x, center_y, semi_major_axis, semi_minor_axis, angle)

    def generate_random_node(self):
        """
        Generate a random node within the ellipsoid if defined, otherwise in the entire space.
        """
        if self.ellipsoid:
            center_x, center_y, semi_major_axis, semi_minor_axis, angle = self.ellipsoid
            while True:
                # Samples a point within a unit circle
                u, v = np.random.uniform(-1, 1, 2)
                if u**2 + v**2 <= 1:
                    break
            x_scaled = semi_major_axis * u
            y_scaled = semi_minor_axis * v

            # Rotate the point by the given angle
            x_rotated = x_scaled * np.cos(angle) - y_scaled * np.sin(angle)
            y_rotated = x_scaled * np.sin(angle) + y_scaled * np.cos(angle)

            # Translate the point to the ellipse center
            x = center_x + x_rotated
            y = center_y + y_rotated

            return Node((x, y))
        else:
            return super().generate_random_node()

    def planning(self):
        b_path = None
        path_cost = float("inf")
        for _ in range(self.iter_max):
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
                        cost = utils.Utils.path_cost(path)
                        if cost < path_cost:
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
                            cost = utils.Utils.path_cost(path)
                            if cost < path_cost:
                                b_path = path
                                path_cost = cost
                                self.update_ellipsoid(path)

        return b_path


def main():
    x_start = (771, 110)
    x_goal = (62, 968)

    srrt_edge = GuidedSRrtEdge(x_start, x_goal, 0.05, 5000)
    path = srrt_edge.planning()

    if path:
        print(f"Number of nodes: {len(srrt_edge.vertex)}")
        print(f"Path length: {utils.Utils.path_cost(path)}")
        srrt_edge.plotting.animation(srrt_edge.vertex, path, "Guided SRRT-Edge", True)
    else:
        print("No Path Found!")


if __name__ == "__main__":
    main()
