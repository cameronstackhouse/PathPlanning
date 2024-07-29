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
)


class Edge:
    def __init__(self, p1, p2) -> None:
        self.node_1 = p1
        self.node_2 = p2


class RrtEdge(rrt):
    """
    Implements an anytime, time bounded, RRT-Edge in three dimensions.
    """

    def __init__(self, time=float("inf")):
        super().__init__()
        self.E = []
        self.stepsize = float("inf")
        self.flag = {}
        self.total_time = None

        self.initial_path = None
        self.time = time

    def path_from_point(self, point, dist=0):
        path = [np.array([point, self.xt])]
        dist += getDist(point, self.xt)
        x = point
        while x != self.x0:
            x2 = self.Parent[x]
            path.append(np.array([x2, x]))
            dist += getDist(x, x2)
            x = x2
        return path, dist

    def planning(self):
        self.V.append(self.x0)
        best_path = None
        best_path_dist = float("inf")

        start_time = time.time()
        while self.ind < self.maxiter:
            current_time = time.time()

            if current_time - start_time > self.time:
                break

            xrand = sampleFree(self)
            xnearest = self.nearest(xrand, self.E)
            xnew, dist = steer(self, xnearest, xrand)

            collide, _ = isCollide(self, xnearest, xnew, dist=dist)
            if not collide:
                new_edge = Edge(xnearest, xnew)
                self.E.append(new_edge)
                self.V.append(xnew)
                self.flag[xnew] = "Valid"
                self.wireup(tuple(xnew), tuple(xnearest))

                goal_dist = getDist(xnew, self.xt)
                goal_collide, _ = isCollide(self, xnew, self.xt, goal_dist)
                if not goal_collide:
                    self.wireup(tuple(self.xt), tuple(xnew))
                    new_path, D = self.path_from_point(xnew)

                    if D < best_path_dist:
                        best_path = new_path
                        best_path_dist = D

                self.i += 1
            self.ind += 1
        self.done = True

        self.Path = best_path

        return self.Path

    def nearest(self, point, edge_list):
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

    def move_dynamic_obs(self):
        for obj in self.dynamic_obs:
            old, new = self.env.move_block(
                a=obj.velocity, block_to_move=obj.index, mode="translation"
            )
            obj.current_pos = obj.update_pos()

    def plot_traversal(self, path):
        # TODO
        if len(path) > 0:
            pass
        else:
            print("Error, can't plot empty path")

    def move(self, path, mps=6):
        if self.current_index >= len(path) - 1:
            return self.xt

        current = self.agent_pos
        next = path[self.current_index + 1][1]

        seg_distance = getDist(current, next)

        direction = (
            (next[0] - current[0]) / seg_distance,
            (next[1] - current[1]) / seg_distance,
            (next[2] - current[2]) / seg_distance,
        )

        new_pos = (
            current[0] + direction[0] * mps,
            current[1] + direction[1] * mps,
            current[2] + direction[2] * mps,
        )

        if getDist(current, new_pos) >= seg_distance:
            self.agent_pos = next
            self.current_index += 1
            return next

        future_uav_positions = []
        PREDICTION_HORIZON = 4
        for t in range(1, PREDICTION_HORIZON):
            future_pos = (
                current[0] + direction[0] * mps * t,
                current[1] + direction[1] * mps * t,
                current[2] + direction[2] * mps * t,
            )

            if getDist(current, future_pos) >= seg_distance:
                break

            future_uav_positions.append(future_pos)

        for future_pos in future_uav_positions:
            for dynamic_object in self.dynamic_obs:
                dynamic_future_pos = dynamic_object.predict_future_positions(
                    PREDICTION_HORIZON
                )

                for pos in dynamic_future_pos:
                    original_pos = dynamic_object.current_pos
                    dynamic_object.current_pos = pos

                    if self.in_dynamic_obj(future_pos, dynamic_object):
                        dynamic_object.current_pos = original_pos
                        return [None, None, None]

                    dynamic_object.current_pos = original_pos

        return new_pos

    def run(self):
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)
        prev_coords = self.x0

        start_time = time.time()
        path = self.planning()
        start_time = time.time() - start_time

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        start_time = time.time()
        if path:
            path = path[::-1]
            self.compute_time = start_time
            start = self.env.start
            goal = self.env.goal

            current = start
            self.agent_pos = current

            self.agent_positions.append(tuple(self.agent_pos))

            while tuple(self.agent_pos) != tuple(goal):
                self.move_dynamic_obs()
                new_coords = self.move(path, self.speed)
                if new_coords[0] is None:
                    start_replan = time.time()
                    self.E = []
                    self.flag = {}
                    self.V = [self.agent_pos]
                    self.ind = 0

                    new_path = self.planning()
                    end_replan = time.time() - start_replan
                    self.replanning_time.append(end_replan)

                    if not new_path:
                        self.agent_positions.append(self.agent_pos)
                        return None
                    else:
                        path = new_path[::-1]
                        self.current_index = 0
                        self.agent_positions.append(self.agent_pos)
                else:
                    self.agent_positions.append(new_coords)
                    self.agent_pos = new_coords

                    self.distance_travelled += getDist(prev_coords, new_coords)
                    prev_coords = new_coords

            self.total_time = time.time() - start_time
            return self.agent_positions
        else:
            return None


if __name__ == "__main__":
    p = RrtEdge(5)
    p.change_env("Evaluation/Maps/3D/block_map_25_3d/block_18_3d.json")
    starttime = time.time()
    a = p.run()

    print(a)
    print("Time used = " + str(time.time() - starttime))
