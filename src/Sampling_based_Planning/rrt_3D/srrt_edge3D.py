import math
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "../Sampling_based_Planning/"
)

from rrt_edge3D import RrtEdge, Edge
from rrt_3D.env3D import env
from rrt_3D.utils3D import (
    getDist,
    sampleFree,
    steer,
    isCollide,
    near,
    nearest,
    visualization,
    cost,
    path,
)


class SRrtEdge(RrtEdge):
    """
    TODO, make anytime
    """

    def __init__(self):
        super().__init__()

    def run(self):
        self.V.append(self.x0)
        best_path = None
        best_path_dist = float("inf")
        while self.ind < self.maxiter:
            xrand = sampleFree(self)
            xnearest = self.nearest(xrand, self.E)
            xnew, dist = steer(self, xnearest, xrand)
            collide, _ = isCollide(self, xnearest, xnew, dist=dist)
            if not collide:
                new_edge = Edge(xnearest, xnew)
                self.E.append(new_edge)
                self.V.append(xnew)
                self.wireup(tuple(xnew), tuple(xnearest))

                goal_dist = getDist(xnew, self.xt)
                goal_collide, _ = isCollide(self, xnew, self.xt, goal_dist)
                if not goal_collide:
                    self.wireup(tuple(self.xt), tuple(xnew))
                    self.Path, D = path(self)
                    print("Total distance = " + str(D))
                    break

                # Checks for direct path from points along the added edge to the goal
                k = self.calculate_k(new_edge)
                partition_points = self.get_k_partitions(k, new_edge)
                found = False
                for partition_point in partition_points:
                    self.Parent[tuple(partition_point)] = tuple(new_edge.node_1)
                    goal_partition_collide, _ = isCollide(
                        self, partition_point, self.xt, goal_dist
                    )
                    if not goal_partition_collide:
                        found = True
                        self.wireup(tuple(self.xt), tuple(partition_point))
                        self.Path, D = path(self)
                        print("Total distance = " + str(D))
                        break

                if found:
                    break

                visualization(self)
                self.i += 1
            self.ind += 1
        self.done = True
        visualization(self)
        plt.show()

        if self.Path:
            return True
        else:
            return False

    def calculate_k(self, edge):
        """
        TODO
        """
        node_1 = edge.node_1
        node_2 = edge.node_2

        edge_len = getDist(node_1, node_2)

        return min(5, math.ceil(edge_len))

    def get_k_partitions(self, k, edge):
        """
        TODO
        """
        x1, y1, z1 = edge.node_1
        x2, y2, z2 = edge.node_2

        dx = (x2 - x1) / k
        dy = (y2 - y1) / k
        dz = (z2 - z1) / k

        midpoints = []
        for i in range(k):
            midpoint_x = x1 + (i + 0.5) * dx
            midpoint_y = y1 + (i + 0.5) * dy
            midpoint_z = z1 + (i + 0.5) * dz
            midpoints.append((midpoint_x, midpoint_y, midpoint_z))

        return midpoints


if __name__ == "__main__":
    p = SRrtEdge()
    p.run()
