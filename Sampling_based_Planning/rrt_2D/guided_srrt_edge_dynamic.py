import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import utils
from rrt_2D.rrt import Node
from rrt_2D.rrt_edge import Edge
from rrt_2D.guided_srrt_edge import GuidedSRrtEdge

class GuidedSRrtEdgeDynamic(GuidedSRrtEdge):
    def __init__(self, start, end, goal_sample_rate, iter_max, min_edge_length=4):
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
    
    def replanning(self):
        """TODO"""
        pass

    def sample_node_replanning(self):
        """TODO"""
        pass

    def on_press(self, event):
        x, y = event.xdata, event.ydata

        # TODO checks

        x, y = int(x), int(y)
        self.obs_circle.append([x, y, 2])
        self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle)
        self.InvalidateNodes()

        if self.is_path_invalid():
            pass

    def InvalidateNodes(self):
        """TODO"""
        pass

    def is_path_invalid(self):
        """TODO"""
        for node in self.path:
            if not node.valid:
                return True
            
        return False


def main():
    pass

if __name__ == "__main__":
    pass

    