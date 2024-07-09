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


class DynamicObj:
    def __init__(self) -> None:
        self.velocity = []
        self.size = []
        self.known = False
        self.current_pos = []
        self.index = 0
        self.init_pos = None
        self.old_pos = None

    def update_pos(self):
        velocity = self.velocity
        new_pos = [
            self.current_pos[0] + (velocity[0]),
            self.current_pos[1] + (velocity[1]),
            self.current_pos[2] + (velocity[2]),
        ]

        return new_pos


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

    def wireup(self, x, y):
        # self.E.add_edge([s, y])  # add edge
        self.Parent[x] = y

    def run(self):
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
        # visualization(self)
        # plt.show()

    def change_env(self, map_name, obs_name=None):
        """
        TODO
        """
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.V = []
            self.i = 0
            self.done = False
            self.Path = []
            self.Parent = {}

            self.env = CustomEnv(data)

            self.x0 = tuple(self.env.start)
            self.xt = tuple(self.env.goal)
        else:
            print("Error, failed to load custom environment.")


if __name__ == "__main__":
    p = rrt()
    starttime = time.time()
    p.run()
    print("time used = " + str(time.time() - starttime))
