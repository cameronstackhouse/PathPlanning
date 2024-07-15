import json
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from collections import defaultdict

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)
from Search_3D.env3D import CustomEnv, env
from Search_3D.utils3D import (
    getDist,
    heuristic_fun,
    getNearest,
    isinbound,
    cost,
    children,
    StateSpace,
)
from Search_3D.plot_util3D import visualization
from Search_3D import queue
import time


class DynamicObj:
    def __init__(self) -> None:
        self.velocity = []
        self.size = []
        self.known = False
        self.current_pos = []
        self.index = 0
        self.init_pos = None
        self.old_pos = None
        self.corners = []

    def update_pos(self):
        velocity = self.velocity
        new_pos = [
            self.current_pos[0] + (velocity[0]),
            self.current_pos[1] + (velocity[1]),
            self.current_pos[2] + (velocity[2]),
        ]

        return new_pos


class D_star_Lite(object):
    # Original version of the D*lite
    def __init__(self, resolution=1):
        self.Alldirec = {
            (1, 0, 0): 1,
            (0, 1, 0): 1,
            (0, 0, 1): 1,
            (-1, 0, 0): 1,
            (0, -1, 0): 1,
            (0, 0, -1): 1,
            (1, 1, 0): np.sqrt(2),
            (1, 0, 1): np.sqrt(2),
            (0, 1, 1): np.sqrt(2),
            (-1, -1, 0): np.sqrt(2),
            (-1, 0, -1): np.sqrt(2),
            (0, -1, -1): np.sqrt(2),
            (1, -1, 0): np.sqrt(2),
            (-1, 1, 0): np.sqrt(2),
            (1, 0, -1): np.sqrt(2),
            (-1, 0, 1): np.sqrt(2),
            (0, 1, -1): np.sqrt(2),
            (0, -1, 1): np.sqrt(2),
            # (1, 1, 1): np.sqrt(3),
            # (-1, -1, -1): np.sqrt(3),
            # (1, -1, -1): np.sqrt(3),
            # (-1, 1, -1): np.sqrt(3),
            # (-1, -1, 1): np.sqrt(3),
            # (1, 1, -1): np.sqrt(3),
            # (1, -1, 1): np.sqrt(3),
            # (-1, 1, 1): np.sqrt(3),
        }
        self.dynamic_obs = []
        self.env = env(resolution=resolution)
        self.settings = "CollisionChecking"  # for collision checking
        self.x0, self.xt = tuple(self.env.start), tuple(self.env.goal)
        self.OPEN = queue.MinheapPQ()
        self.km = 0
        self.g = {}  # all g initialized at inf
        self.rhs = {self.xt: 0}  # rhs(x0) = 0
        self.h = {}
        self.OPEN.put(self.xt, self.CalculateKey(self.xt))
        self.CLOSED = set()

        # init children set:
        self.CHILDREN = {}
        # init Cost set
        self.COST = defaultdict(lambda: defaultdict(dict))

        # for visualization
        self.V = set()  # vertice in closed
        self.ind = 0
        self.Path = []
        self.done = False

        self.name = "D* Lite"

        self.current_index = 0
        self.agent_pos = None
        self.agent_positions = []
        self.speed = 6

    def updatecost(self, range_changed=None, new=None, old=None, mode=False):
        # scan graph for changed Cost, if Cost is changed update it
        CHANGED = set()
        for xi in self.CLOSED:
            if isinbound(old, xi, mode) or isinbound(new, xi, mode):
                newchildren = set(children(self, xi))  # B
                self.CHILDREN[xi] = newchildren
                for xj in newchildren:
                    self.COST[xi][xj] = cost(self, xi, xj)
                CHANGED.add(xi)
        return CHANGED

    def getcost(self, xi, xj):
        # use a LUT for getting the costd
        if xi not in self.COST:
            for xj, xjcost in children(self, xi, settings=1):
                self.COST[xi][xj] = cost(self, xi, xj, xjcost)
        # this might happen when there is a node changed.
        if xj not in self.COST[xi]:
            self.COST[xi][xj] = cost(self, xi, xj)
        return self.COST[xi][xj]

    def getchildren(self, xi):
        if xi not in self.CHILDREN:
            allchild = children(self, xi)
            self.CHILDREN[xi] = set(allchild)
        return self.CHILDREN[xi]

    def geth(self, xi):
        # when the heurisitic is first calculated
        if xi not in self.h:
            self.h[xi] = heuristic_fun(self, xi, self.x0)
        return self.h[xi]

    def getg(self, xi):
        if xi not in self.g:
            self.g[xi] = np.inf
        return self.g[xi]

    def getrhs(self, xi):
        if xi not in self.rhs:
            self.rhs[xi] = np.inf
        return self.rhs[xi]

    # -------------main functions for D*Lite-------------

    def CalculateKey(self, s, epsilion=1):
        return [
            min(self.getg(s), self.getrhs(s)) + epsilion * self.geth(s) + self.km,
            min(self.getg(s), self.getrhs(s)),
        ]

    def UpdateVertex(self, u):
        # if still in the hunt
        if not getDist(self.xt, u) <= self.env.resolution:  # originally: u != x_goal
            if u in self.CHILDREN and len(self.CHILDREN[u]) == 0:
                self.rhs[u] = np.inf
            else:
                self.rhs[u] = min(
                    [self.getcost(s, u) + self.getg(s) for s in self.getchildren(u)]
                )
        # if u is in OPEN, remove it
        self.OPEN.check_remove(u)
        # if rhs(u) not equal to g(u)
        if self.getg(u) != self.getrhs(u):
            self.OPEN.put(u, self.CalculateKey(u))

    def ComputeShortestPath(self):
        while self.OPEN.top_key() < self.CalculateKey(self.x0) or self.getrhs(
            self.x0
        ) != self.getg(self.x0):
            kold = self.OPEN.top_key()
            u = self.OPEN.get()
            self.V.add(u)
            self.CLOSED.add(u)
            if not self.done:  # first time running, we need to stop on this condition
                if getDist(self.x0, u) < 1 * self.env.resolution:
                    self.x0 = u
                    break
            if kold < self.CalculateKey(u):
                self.OPEN.put(u, self.CalculateKey(u))
            if self.getg(u) > self.getrhs(u):
                self.g[u] = self.rhs[u]
            else:
                self.g[u] = np.inf
                self.UpdateVertex(u)
            for s in self.getchildren(u):
                self.UpdateVertex(s)
            # visualization(self)
            self.ind += 1

    def change_env(self, map_name, obs_name=None):
        """
        TODO
        """
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.current_index = 0
            self.agent_pos = None
            self.dynamic_obs = []
            self.V = set()
            self.i = 0
            self.done = False
            self.Path = []
            self.Parent = {}
            self.km = 0

            self.env = CustomEnv(data)

            self.x0 = tuple(self.env.start)
            self.xt = tuple(self.env.goal)

            self.rhs = {self.xt: 0}  # rhs(x0) = 0

            self.OPEN = queue.MinheapPQ()
            self.OPEN.put(self.xt, self.CalculateKey(self.xt))

            self.g = {}
            self.h = {}
            self.CLOSED = set()
            self.CHILDREN = {}

            self.COST = defaultdict(lambda: defaultdict(dict))

            if obs_name:
                self.set_dynamic_obs(obs_name)

            return self.env
        else:
            print("Error, failed to load custom environment.")

    def corner_coords(self, x1, y1, z1, width, height, depth):
        x2 = x1 + width
        y2 = y1 + height
        z2 = z1 + depth
        return (x1, y1, z1, x2, y2, z2)

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

    def path(self, s_start=None):
        """After ComputeShortestPath()
        returns, one can then follow a shortest path from x_init to
        x_goal by always moving from the current vertex s, starting
        at x_init. , to any successor s' that minimizes cBest(s,s') + g(s')
        until x_goal is reached (ties can be broken arbitrarily)."""
        path = []
        s_goal = self.xt
        if not s_start:
            s = self.x0
        else:
            s = s_start
        ind = 0
        while s != s_goal:
            if s == self.x0:
                children = [
                    i
                    for i in self.CLOSED
                    if getDist(s, i) <= self.env.resolution * np.sqrt(3)
                ]
            else:
                children = list(self.CHILDREN[s])
            snext = children[
                np.argmin([self.getcost(s, s_p) + self.getg(s_p) for s_p in children])
            ]
            path.append([s, snext])
            s = snext
            if ind > 100:
                break
            ind += 1
        return path

    def visualise(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the environment blocks
        for block in self.env.blocks:
            x, y, z = block[0], block[1], block[2]
            dx, dy, dz = block[3] - block[0], block[4] - block[1], block[5] - block[2]
            ax.bar3d(x, y, z, dx, dy, dz, color="b", alpha=0.5)

        # Plot the dynamic obstacles
        for obj in self.dynamic_obs:
            x, y, z = obj.current_pos
            dx, dy, dz = obj.size
            ax.bar3d(x, y, z, dx, dy, dz, color="r", alpha=0.5)

        # Plot the path
        path_points = np.array([point[0] for point in path])
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            color="g",
            marker="o",
        )

        ax.scatter(
            self.env.start[0],
            self.env.start[1],
            self.env.start[2],
            color="g",
            s=100,
            label="Start",
        )
        ax.scatter(
            self.env.goal[0],
            self.env.goal[1],
            self.env.goal[2],
            color="r",
            s=100,
            label="Goal",
        )

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("D* Lite Path visualisation")
        ax.legend()

        plt.show()

    def move_dynamic_obs(self):
        self.Path = []
        changed = None
        for obj in self.dynamic_obs:
            old, new = self.env.move_block(
                a=obj.velocity, block_to_move=obj.index, mode="translation"
            )
            n_changed = self.updatecost(True, new, old)
            if changed is None:
                changed = n_changed
            else:
                changed = changed.union(n_changed)

        self.V = set()
        if changed is not None:
            for u in changed:
                self.UpdateVertex(u)
            self.ComputeShortestPath()

        self.Path = self.path(self.x0)

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
            v1 = np.array(next) - np.array(current)
            v2 = np.array(new_pos) - np.array(next)
            dot_product = np.dot(v1, v2)

            mag_v1 = np.linalg.norm(v1)
            mag_v2 = np.linalg.norm(v2)

            same_dir = np.isclose(dot_product, mag_v1 * mag_v2)

            if same_dir:
                # Move the agent far forward without turning
                self.current_index += 1
                count = 0
                while self.current_index < len(path) - 1:
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
                    v1 = np.array(next) - np.array(current)
                    v2 = np.array(new_pos) - np.array(next)
                    dot_product = np.dot(v1, v2)
                    mag_v1 = np.linalg.norm(v1)
                    mag_v2 = np.linalg.norm(v2)
                    same_dir = np.isclose(dot_product, mag_v1 * mag_v2)
                    if not same_dir or count >= mps - 1:
                        break
                    current = next
                    self.agent_pos = current
                    self.current_index += 1
                    count += 1
            else:
                self.agent_pos = next
                self.current_index += 1
        else:
            self.agent_pos = new_pos

        return self.agent_pos

    def run(self):
        # TODO
        self.agent_pos = self.x0
        self.ComputeShortestPath()
        self.Path = self.path(self.x0)
        t = 0
        self.V = set()
        while self.agent_pos != self.xt:
            self.env.start = self.x0
            self.move_dynamic_obs()
            self.move(self.Path)
            self.agent_positions.append(self.agent_pos)

            t += 1


if __name__ == "__main__":

    D_lite = D_star_Lite(1)
    D_lite.change_env("Evaluation/Maps/3D/block_map_25_3d/4_3d.json")
    a = time.time()
    # D_lite.run()
    # print("used time (s) is " + str(time.time() - a))

    D_lite.ComputeShortestPath()
    path = D_lite.path()

    D_lite.visualise(path)
    # print(path)
