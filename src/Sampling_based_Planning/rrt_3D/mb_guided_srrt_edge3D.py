import math
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import psutil


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "../Sampling_based_Planning/"
)

from rrt_edge3D import Edge
from guided_srrt_edge3D import GuidedSrrtEdge
from rrt_3D.env3D import env
from rrt_3D.utils3D import (
    getDist,
    sampleFree,
    steer,
    isCollide,
    visualization,
    path_from_point,
)


class MbGuidedSrrtEdge(GuidedSrrtEdge):
    """
    TODO
    """

    def __init__(self, t=0.1, m=10000):
        super().__init__()
        self.t = t
        self.m = m

    def run(self):
        self.V.append(self.x0)
        best_path = None
        best_path_dist = float("inf")
        start_time = time.time()
        while True:
            # Check for termination criteria
            elapsed_time = time.time() - start_time

            process = psutil.Process(os.getpid())
            memory_usage = (process.memory_info().rss) / (1024 * 1024)

            if elapsed_time > self.t or memory_usage > self.m:
                break

            # Sample new node
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
                    current_path, D = path_from_point(self, tuple(xnew))

                    if D < best_path_dist:
                        self.Path = current_path
                        best_path = current_path
                        best_path_dist = D
                        self.update_ellipsoid(best_path)

                # Checks for direct path from points along the added edge to the goal
                k = self.calculate_k(new_edge)
                partition_points = self.get_k_partitions(k, new_edge)
                for partition_point in partition_points:
                    self.Parent[tuple(partition_point)] = tuple(new_edge.node_1)
                    goal_partition_collide, _ = isCollide(
                        self, partition_point, self.xt, goal_dist
                    )
                    if not goal_partition_collide:
                        self.wireup(tuple(self.xt), tuple(partition_point))
                        current_path, D = path_from_point(self, tuple(partition_point))

                        if D < best_path_dist:
                            self.Path = current_path
                            best_path = current_path
                            best_path_dist = D
                            self.update_ellipsoid(best_path)

                # visualization(self)
                self.i += 1

        self.done = True
        visualization(self)
        plt.show()

        print(best_path_dist)

        if self.Path:
            return True
        else:
            return False


if __name__ == "__main__":
    TIME = 2.53
    p = MbGuidedSrrtEdge(TIME)
    p.run()
