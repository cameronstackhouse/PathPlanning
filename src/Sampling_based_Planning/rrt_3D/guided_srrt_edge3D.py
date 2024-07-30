import math
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "../Sampling_based_Planning/"
)

from rrt_edge3D import Edge
from srrt_edge3D import SRrtEdge
from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, steer, isCollide, visualization


class GuidedSrrtEdge(SRrtEdge):
    def __init__(self):
        super().__init__()
        self.ellipsoid = None
        self.maxiter = 200

    def change_env(self, map_name, obs_name=None, size=None):
        super().change_env(map_name, obs_name, size)
        self.ellipsoid = None
        self.Path = []
        self.E = []

    def sample_unit_ball(self):
        # TODO credit author
        r = np.random.uniform(0.0, 1.0)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def RotationToWorldFrame(self, xstart, xgoal):
        # TODO credit author
        d = getDist(xstart, xgoal)
        xstart, xgoal = np.array(xstart), np.array(xgoal)
        a1 = (xgoal - xstart) / d
        M = np.outer(a1, [1, 0, 0])
        U, S, V = np.linalg.svd(M)
        C = U @ np.diag([1, 1, np.linalg.det(U) * np.linalg.det(V)]) @ V.T
        return C

    def update_ellipsoid(self, path):
        self.E = []
        self.V = [self.x0]
        unique_points = set()
        for segment in path:
            for point in segment:
                unique_points.add(tuple(point))
        unique_points = list(unique_points)

        x1, y1, z1 = unique_points[0]
        x2, y2, z2 = unique_points[-1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_z = (z1 + z2) / 2

        semi_major_axis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) / 2

        max_distance = max(
            np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)
            for x, y, z in unique_points
        )

        semi_minor_axis = max_distance

        rotation_matrix = self.RotationToWorldFrame((x1, y1, z1), (x2, y2, z2))

        self.ellipsoid = (
            center_x,
            center_y,
            center_z,
            semi_major_axis,
            semi_minor_axis,
            rotation_matrix,
        )

    def sampleFree(self):
        if self.ellipsoid:
            (
                center_x,
                center_y,
                center_z,
                semi_major_axis,
                semi_minor_axis,
                rotation_matrix,
            ) = self.ellipsoid

            u, v, w = self.sample_unit_ball()

            x_scaled = semi_major_axis * u
            y_scaled = semi_minor_axis * v
            z_scaled = semi_minor_axis * w

            rotated_point = rotation_matrix @ np.array([x_scaled, y_scaled, z_scaled])

            x = center_x + rotated_point[0]
            y = center_y + rotated_point[1]
            z = center_z + rotated_point[2]

            return (x, y, z)

        else:
            return self.sampleFree()

    def planning(self):
        self.V.append(self.x0)
        
        best_path = None
        best_path_dist = float("inf")
        for _ in range(self.maxiter):
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
                    current_path, D = self.path_from_point(tuple(xnew))

                    if D < best_path_dist:
                        self.Path = current_path
                        best_path = current_path
                        best_path_dist = D

                        self.update_ellipsoid(best_path)

                # Checks for direct path from points along the added edge to the goal
                k = self.calculate_k(new_edge)
                partition_points = self.get_k_partitions(k, new_edge)
                for partition_point in partition_points:
                    self.wireup(tuple(new_edge.node_1), tuple(partition_point))
                    goal_partition_collide, _ = isCollide(
                        self, partition_point, self.xt, goal_dist
                    )
                    if not goal_partition_collide:
                        self.wireup(tuple(self.xt), tuple(partition_point))
                        current_path, D = self.path_from_point(tuple(partition_point))

                        if D < best_path_dist:
                            self.Path = current_path
                            best_path = current_path
                            best_path_dist = D
                            self.update_ellipsoid(best_path)

                # visualization(self)
                # self.i += 1

        self.done = True

        if self.Path:
            return True
        else:
            return False


if __name__ == "__main__":
    p = GuidedSrrtEdge()

    p.change_env(
        "Evaluation/Maps/3D/main/house_17_3d.json",
        size=28
    )

    # visualization(p)
    # plt.show()

    print("1")
    p.planning()

    p.change_env(
        "Evaluation/Maps/3D/main/house_19_3d.json",
        size=28
    )

    # visualization(p)
    # plt.show()

    print("2")
    p.planning()

    print(p.Path)
