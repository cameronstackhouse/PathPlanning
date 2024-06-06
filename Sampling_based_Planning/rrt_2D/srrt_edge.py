import os
import sys
import math

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import utils
from rrt_2D.rrt import Node
from rrt_2D.rrt_edge import RrtEdge, Edge


class SRrtEdge(RrtEdge):
    """
    Version of RRT Edge which performs K checks along the edge of a newly added
    node to see if the 
    """

    def __init__(self, start, end, goal_sample_rate, iter_max, min_edge_length=5):
        super().__init__(start, end, goal_sample_rate, iter_max, min_edge_length)

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

                # # Checks for direct paths from points along the added edge to the goal
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
                            if cost < path_cost:
                                b_path = path
                                path_cost = cost

        return b_path

    def calculate_k(self, edge):
        """
        TODO
        """
        x1, x2 = edge.node_1.x, edge.node_2.x
        y1, y2 = edge.node_1.y, edge.node_2.y
        
        edge_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        return math.ceil(edge_len)

    def get_k_partitions(self, k, edge):
        x1, y1 = edge.node_1.coords
        x2, y2 = edge.node_2.coords

        dx = (x2 - x1) / k
        dy = (y2 - y1) / k

        midpoints = []
        for i in range(k):
            midpoint_x = x1 + (i + 0.5) * dx
            midpoint_y = y1 + (i + 0.5) * dy
            midpoints.append((midpoint_x, midpoint_y))

        return midpoints


def main():
    x_start = (2, 2)
    x_goal = (29, 91)

    srrt_edge = SRrtEdge(x_start, x_goal, 0.4, 1000)
    path = srrt_edge.planning()

    if path:
        print(f"Path length: {utils.Utils.path_cost(path)}")
        srrt_edge.plotting.animation(srrt_edge.vertex, path, "SRRT-Edge", True)
    else:
        print("No Path Found!")


if __name__ == "__main__":
    main()
