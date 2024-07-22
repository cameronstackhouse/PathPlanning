import math
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "../Sampling_based_Planning/"
)

from rrt3D import rrt
from rrt_3D.env3D import env
from rrt_3D.utils3D import (
    getDist,
    sampleFree,
    steer,
    isCollide,
    nearest,
    visualization,
    cost,
    path,
)


class Edge:
    def __init__(self, p1, p2) -> None:
        self.node_1 = p1
        self.node_2 = p2


class RrtEdge(rrt):
    """
    Implements RRT-Edge in three dimensions.

    TODO make anytime.
    """

    def __init__(self):
        super().__init__()
        self.E = []
        self.stepsize = float("inf")
        self.flag = {}

    def planning(self):
        self.V.append(self.x0)
        best_path = None
        best_path_dist = float("inf")
        while self.ind < self.maxiter:
            xrand = sampleFree(self)
            xnearest = self.nearest(xrand, self.E)
            xnew, dist = steer(self, xnearest, xrand)
            # TODO Getting key errors for parent, could be child stuff
            collide, _ = isCollide(self, xnearest, xnew, dist=dist)
            if not collide:
                new_edge = Edge(xnearest, xnew)
                self.E.append(new_edge)
                self.V.append(xnew)
                self.flag[xnew] = 'Valid'
                self.wireup(tuple(xnew), tuple(xnearest))

                goal_dist = getDist(xnew, self.xt)
                goal_collide, _ = isCollide(self, xnew, self.xt, goal_dist)
                if not goal_collide:
                    self.wireup(tuple(self.xt), tuple(xnew))
                    self.Path, D = path(self)
                    print("Total distance = " + str(D))
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

    def nearest(self, point, edge_list):
        """
        TODO
        """
        nearest_node = nearest(self, point)
        nearest_edge_dist, nearest_edge_proj, nearest_edge = (
            self.nearest_edge_projection(edge_list, point)
        )

        node_dist = getDist(nearest_node, point)

        if nearest_edge_proj is not None and nearest_edge_dist < node_dist:
            # Set its parent
            self.Parent[tuple(nearest_edge_proj)] = tuple(nearest_edge.node_1)
            return nearest_edge_proj
        else:
            return nearest_node

    def nearest_edge_projection(self, edge_list, n):
        """
        TODO description.
        """
        min_distance = float("inf")
        proj = None
        nearest_edge = None

        for edge in edge_list:
            proj_node_coords = self.orthogonal_projection(edge, n)

            if proj_node_coords is not None:
                distance = getDist(proj_node_coords, n)

                if distance < min_distance:
                    min_distance = distance
                    proj = proj_node_coords
                    nearest_edge = edge

        return min_distance, proj, nearest_edge

    @staticmethod
    def orthogonal_projection(edge, new_node):
        P1 = np.array(edge.node_1)
        P2 = np.array(edge.node_2)

        A = np.array(new_node)

        B = P2 - P1
        B_norm_sq = np.dot(B, B)

        A_shifted = A - P1

        P_A = (np.dot(A_shifted, B) / B_norm_sq) * B

        # If the projection does not lie on the edge
        if np.dot(P_A, B) < 0 or np.dot(P_A, B) > B_norm_sq:
            return None

        proj_coords = P_A + P1

        return proj_coords
    

if __name__ == "__main__":
    p = RrtEdge()
    starttime = time.time()
    p.planning()
    print("Time used = " + str(time.time() - starttime))
