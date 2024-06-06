import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from rrt_2D import utils
from rrt_2D.rrt import Node
from rrt_2D.rrt_edge import RrtEdge, Edge


class SRrtEdge(RrtEdge):
    """
    Version of RRT Edge which performs K checks along the edge of a newly added
    node to see if 
    """
    def __init__(self, start, end, goal_sample_rate, iter_max, min_edge_length=3):
        super().__init__(start, end, goal_sample_rate, iter_max, min_edge_length)

    def planning(self):
        b_path = None
        path_cost = float('inf')
        for _ in range(self.iter_max):
            node_rand = self.generate_random_node()
            node_near = self.nearest_neighbour(self.vertex, self.edges, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                new_edge = Edge(node_near, node_new)
                self.edges.append(new_edge)

                if node_new.edge is not None:
                    self.split(node_new)

                # Checks for direct paths from points along the added edge to the 
                k = self.calculate_k(new_edge)
                partition_points = self.get_k_partitions(k, new_edge)
                
                for point in partition_points:
                    point_node = Node(point)
                    point_node.edge = new_edge
                    if not self.utils.is_collision(point_node, self.s_goal):
                        self.vertex.append(point_node)
                        point_node.parent = new_edge.node_1
                        self.new_state(point_node, self.s_goal)
                        path = self.extract_path(point_node)
                        cost = utils.Utils.path_cost(path)
                        if cost < path_cost:
                            b_path = path
                            path_cost = cost

                if not self.utils.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal)
                    path = self.extract_path(node_new)
                    cost = utils.Utils.path_cost(path)
                    if cost < path_cost:
                        b_path = path
                        path_cost = cost

        return b_path
    
    def calculate_k(self, edge):
        """
        TODO
        """
        return 3
    
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
    x_goal = (80, 10)

    srrt_edge = SRrtEdge(x_start, x_goal, 0.4, 2000)
    path = srrt_edge.planning()

    if path:
        print(f"Path length: {utils.Utils.path_cost(path)}")
        srrt_edge.plotting.animation(srrt_edge.vertex, path, "RRT-Edge", True)

if __name__ == "__main__":
    main()
