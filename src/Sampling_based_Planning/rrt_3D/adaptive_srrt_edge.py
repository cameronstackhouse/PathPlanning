import sys
import os
import time

import numpy as np
import psutil

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "../Sampling_based_Planning/"
)

from rrt_3D.rrt_edge3D import Edge
from rrt_3D.mb_guided_srrt_edge3D import MbGuidedSrrtEdge
from rrt_3D.utils3D import (
    getDist,
    sampleFree,
    steer,
    isCollide,
    visualization,
)


class AdaptiveSRRTEdge(MbGuidedSrrtEdge):
    def __init__(self, t=5, m=10000, threshold=25, x=10):
        super().__init__(t, m)
        self.reject_count = 0
        self.b_path = None
        self.best_path_dist = float("inf")
        self.T = threshold
        self.x = x

    def unlimited_step(self, x, y):
        if np.equal(x, y).all():
            return x, 0.0

        dist = getDist(x, y)
        step = dist

        increment = (
            (y[0] - x[0]) / dist * step,
            (y[1] - x[1]) / dist * step,
            (y[2] - x[2]) / dist * step,
        )
        xnew = (x[0] + increment[0], x[1] + increment[1], x[2] + increment[2])

        return xnew, dist

    def change_env(self, map_name, obs_name=None, size=None):
        super().change_env(map_name, obs_name, size)
        self.reject_count = 0
        self.b_path = None
        self.best_path_dist = float("inf")

    def planning(self):
        self.Parent = {}
        self.done = False
        self.E = []
        self.V = []
        self.flag = {}

        self.V.append(self.x0)
        self.flag[tuple(self.x0)] = "Valid"
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
            xnew, dist = self.unlimited_step(self, xnearest, xrand)
            collide, _ = isCollide(self, xnearest, xnew, dist=dist)

            if not collide:
                self.accept_sample(xnew, xnearest)
                self.reject_count = 0

                if self.stepsize + self.x > self.env.xmax:
                    self.stepsize = float("inf")
                else:
                    self.stepsize += self.x
            else:
                node, new_dist = steer(self, xnearest, xrand)
                collide, _ = isCollide(self, xnearest, xnew, dist=dist)

                if not collide:
                    self.accept_sample(node, xnearest)
                    self.reject_count = 0

                    if self.stepsize + self.x > self.env.xmax:
                        self.stepsize = float("inf")
                    else:
                        self.stepsize += self.x

                else:
                    self.reject_count += 1
                    if self.reject_count == self.T:
                        if self.stepsize == float("inf"):
                            self.stepsize = self.env.xmax - self.x
                        elif self.stepsize - self.x > 0:
                            self.stepsize -= self.x
                        else:
                            self.stepsize = 1

            self.i += 1

        self.done = True
        
        return self.Path

    def accept_sample(self, xnew, xnearest):
        new_edge = Edge(xnew, xnearest)
        self.E.append(new_edge)
        self.V.append(xnew)
        self.flag[tuple(xnew)] = "valid"

        self.wireup(tuple(xnew), tuple(xnearest))

        goal_dist = getDist(xnew, self.xt)
        goal_collide, _ = isCollide(self, xnew, self.xt, goal_dist)
        if not goal_collide:
            self.flag[tuple(self.xt)] = "Valid"
            self.wireup(tuple(self.xt), tuple(xnew))
            current_path, D = self.path_from_point(tuple(xnew))

            if D < self.best_path_dist:
                self.Path = current_path
                self.b_path = current_path
                self.best_path_dist = D
                self.update_ellipsoid(self.b_path)

        # Checks for direct path from points along the added edge to the goal
        k = self.calculate_k(new_edge)
        partition_points = self.get_k_partitions(k, new_edge)
        for partition_point in partition_points:
            self.wireup(tuple(partition_point), tuple(xnearest))
            goal_partition_collide, _ = isCollide(
                self, partition_point, self.xt, goal_dist
            )
            if not goal_partition_collide:
                self.wireup(tuple(self.xt), tuple(partition_point))
                current_path, D = self.path_from_point(tuple(partition_point))

                if D < self.best_path_dist:
                    self.Path = current_path
                    self.b_path = current_path
                    self.best_path_dist = D
                    self.update_ellipsoid(self.b_path)


if __name__ == "__main__":
    pass