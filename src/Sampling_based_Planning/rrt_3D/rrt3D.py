"""
This is rrt star code for 3D
@author: yue qi
"""

import json
import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt

import os
import sys

import psutil

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_3D.env3D import env, CustomEnv
from rrt_3D.utils3D import (
    getDist,
    sampleFree,
    nearest,
    steer,
    isCollide,
    near,
    visualization,
    cost,
    path,
)

from rrt_3D.DynamicObj import DynamicObj


class rrt:
    def __init__(self):
        self.env = env()
        self.Parent = {}
        self.V = []
        # self.E = edgeset()
        self.i = 0
        self.maxiter = 10000
        self.stepsize = 0.5
        self.Path = []
        self.done = False
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)

        self.ind = 0
        # self.fig = plt.figure(figsize=(10, 8))

        # Dynamic variables
        self.current_index = 0
        self.agent_positions = []
        self.agent_pos = None
        self.distance_travelled = 0
        self.time = 3
        self.compute_time = None
        self.speed = 6

        self.dynamic_obs = []

        self.replanning_time = []
    
    def corner_coords(self, x1, y1, z1, width, height, depth):
        x2 = x1 + width
        y2 = y1 + height
        z2 = z1 + depth
        return (x1, y1, z1, x2, y2, z2)

    def in_dynamic_obj(self, pos, obj):
        x, y, z = pos
        x0, y0, z0 = obj.current_pos
        width, height, depth = obj.size
        return (
            (x0 <= x <= x0 + width)
            and (y0 <= y <= y0 + height)
            and (z0 <= z <= z0 + depth)
        )

    def wireup(self, x, y):
        # self.E.add_edge([s, y])  # add edge
        self.Parent[x] = y

    def planning(self):
        self.V.append(self.x0)
        while self.ind < self.maxiter:
            xrand = sampleFree(self)
            xnearest = nearest(self, xrand)
            xnew, dist = steer(self, xnearest, xrand)
            collide, _ = isCollide(self, xnearest, xnew, dist=dist)
            if not collide:
                self.V.append(xnew)  # add point
                self.wireup(xnew, xnearest)

                if getDist(xnew, self.xt) <= self.stepsize:
                    self.wireup(self.xt, xnew)
                    self.Path, D = path(self)
                    print("Total distance = " + str(D))
                    break
                # visualization(self)
                self.i += 1
            self.ind += 1
            # if the goal is really reached

        self.done = True

        return self.Path

    def move(self, path, mps=6):
        # TODO look at
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.xt

        current_pos = self.agent_pos
        next_node = path[self.current_index + 1]

        seg_distance = getDist(current_pos, next_node)

        direction = (
            (next_node[0] - current_pos[0]) / seg_distance,
            (next_node[1] - current_pos[1]) / seg_distance,
            (next_node[2] - current_pos[2]) / seg_distance,
        )

        new_pos = (
            current_pos[0] + direction[0] * mps,
            current_pos[1] + direction[1] * mps,
            current_pos[2] + direction[2] * mps,
        )

        # Checks for overshoot
        if getDist(current_pos, new_pos) >= seg_distance:
            self.agent_pos = next_node
            self.current_index += 1
            return next_node

        future_uav_positions = []
        PREDICTION_HORIZON = 4
        for t in range(1, PREDICTION_HORIZON):
            future_pos = (
                current_pos[0] + direction[0] * mps * t,
                current_pos[1] + direction[1] * mps * t,
                current_pos[2] + direction[2] * mps * t,
            )

            if getDist(current_pos, future_pos) >= seg_distance:
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
                        return [None, None]

                    dynamic_object.current_pos = original_pos

        return new_pos

    def change_env(self, map_name, obs_name=None, size=None):
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.V = []
            self.i = 0
            self.done = False
            self.Path = []
            self.Parent = {}

            if size:
                self.env = CustomEnv(data, xmax=size, ymax=size, zmax=size)
            else:
                self.env = CustomEnv(data)

            self.x0 = tuple(self.env.start)
            self.xt = tuple(self.env.goal)

            self.dobs_dir = obs_name
        else:
            print("Error, failed to load custom environment.")

    def set_dynamic_obs(self, filename):
        obj_json = None
        with open(filename) as f:
            obj_json = json.load(f)

        if obj_json:
            for obj in obj_json["objects"]:
                new_obj = DynamicObj()
                new_obj.velocity = obj["velocity"]
                new_obj.current_pos = obj["position"]
                new_obj.old_pos = obj["position"]
                new_obj.size = obj["size"]
                new_obj.init_pos = new_obj.current_pos
                new_obj.corners = self.corner_coords(
                    new_obj.current_pos[0],
                    new_obj.current_pos[1],
                    new_obj.current_pos[2],
                    new_obj.size[0],
                    new_obj.size[1],
                    new_obj.size[2],
                )

                new_obj.index = len(self.env.blocks) - 1
                self.dynamic_obs.append(new_obj)

                self.env.new_block_corners(new_obj.corners)


if __name__ == "__main__":
    p = rrt()
    starttime = time.time()
    p.planning()
    print("time used = " + str(time.time() - starttime))
