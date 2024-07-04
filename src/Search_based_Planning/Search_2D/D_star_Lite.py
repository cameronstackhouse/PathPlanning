"""
D_star_Lite 2D
@author: huiming zhou
"""

import json
import os
import sys
import math
import time
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_2D import plotting, env


class DStar:
    def __init__(self, s_start, s_goal, heuristic_type, time=float("inf")):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env
        self.Plot = plotting.Plotting(s_start, s_goal)

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0

        self.name = "D* Lite"
        self.first_success = None
        self.time = time

        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.CalculateKey(self.s_goal)
        self.visited = set()
        self.count = 0
        self.fig = plt.figure()

    def run(self):
        self.Plot.plot_grid("D* Lite")
        self.ComputePath()
        self.plot_path(self.path_to_end())
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        plt.show()

    def path_to_end(self):
        s_curr = self.s_start
        path = [self.s_start]

        while s_curr != self.s_goal:
            s_list = {}

            for s in self.get_neighbor(s_curr):
                s_list[s] = self.g[s] + self.cost(s_curr, s)
            s_curr = min(s_list, key=s_list.get)
            path.append(s_curr)

        return path

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Change position: s =", x, ",", "y =", y)

            s_curr = self.s_start
            s_last = self.s_start
            i = 0
            path = [self.s_start]

            while s_curr != self.s_goal:
                s_list = {}

                for s in self.get_neighbor(s_curr):
                    s_list[s] = self.g[s] + self.cost(s_curr, s)
                s_curr = min(s_list, key=s_list.get)
                path.append(s_curr)

                if i < 1:
                    self.km += self.h(s_last, s_curr)
                    s_last = s_curr
                    if (x, y) not in self.obs:
                        self.obs.add((x, y))
                        plt.plot(x, y, "sk")
                        self.g[(x, y)] = float("inf")
                        self.rhs[(x, y)] = float("inf")
                    else:
                        self.obs.remove((x, y))
                        plt.plot(x, y, marker="s", color="white")
                        self.UpdateVertex((x, y))
                    for s in self.get_neighbor((x, y)):
                        self.UpdateVertex(s)
                    i += 1

                    self.count += 1
                    self.visited = set()
                    self.ComputePath()

            self.plot_visited(self.visited)
            self.plot_path(path)
            self.fig.canvas.draw_idle()

    def ComputePath(self):
        start_time = time.time()
        while True:
            if (len(self.U)) == 0:
                return None

            if time.time() - start_time > self.time:
                if (
                    v >= self.CalculateKey(self.s_start)
                    and self.rhs[self.s_start] == self.g[self.s_start]
                ):
                    break
                else:
                    return None

            s, v = self.TopKey()
            if (
                v >= self.CalculateKey(self.s_start)
                and self.rhs[self.s_start] == self.g[self.s_start]
            ):
                break

            k_old = v
            self.U.pop(s)
            self.visited.add(s)

            if k_old < self.CalculateKey(s):
                self.U[s] = self.CalculateKey(s)
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)
            else:
                self.g[s] = float("inf")
                self.UpdateVertex(s)
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)

        return self.path_to_end()

    def UpdateVertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            for x in self.get_neighbor(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.cost(s, x))
        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.CalculateKey(s)

    def CalculateKey(self, s):
        return [
            min(self.g[s], self.rhs[s]) + self.h(self.s_start, s) + self.km,
            min(self.g[s], self.rhs[s]),
        ]

    def TopKey(self):
        """
        :return: return the min key and its value.
        """

        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def h(self, s_start, s_goal):
        heuristic_type = self.heuristic_type  # heuristic type

        if heuristic_type == "manhattan":
            return abs(s_goal[0] - s_start[0]) + abs(s_goal[1] - s_start[1])
        else:
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def get_neighbor(self, s):
        nei_list = set()
        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if s_next not in self.obs and s_next in self.rhs.keys():
                nei_list.add(s_next)

        return nei_list

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for k in range(100):
            g_list = {}
            for x in self.get_neighbor(s):
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == self.s_goal:
                break

        return list(path)

    def plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")

    def plot_visited(self, visited):
        color = [
            "gainsboro",
            "lightgray",
            "silver",
            "darkgray",
            "bisque",
            "navajowhite",
            "moccasin",
            "wheat",
            "powderblue",
            "skyblue",
            "lightskyblue",
            "cornflowerblue",
        ]

        if self.count >= len(color) - 1:
            self.count = 0

        for x in visited:
            plt.plot(x[0], x[1], marker="s", color=color[self.count])

    def change_env(self, map_name):
        """
        TODO
        """

        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.s_start = tuple(data["agent"])
            self.s_goal = tuple(data["goal"])
            self.Env = env.CustomEnv(data)

            self.rhs = {}
            self.g = {}

            for i in range(self.Env.x_range):
                for j in range(self.Env.y_range):
                    self.rhs[(i, j)] = float("inf")
                    self.g[(i, j)] = float("inf")

            self.obs = self.Env.obs

            self.Plot.env = self.Env
            self.Plot.xI = self.s_start
            self.Plot.xG = self.s_goal
            self.Plot.obs = self.Env.obs

            self.rhs[self.s_goal] = 0.0
            self.U = {}
            self.U[self.s_goal] = self.CalculateKey(self.s_goal)
        else:
            print("Error, map not found")


def main():
    s_start = (5, 5)
    s_goal = (989, 888)

    dstar = DStar(s_start, s_goal, "euclidean")
    dstar.change_env("Evaluation/Maps/2D/house_25/house_3.json")
    dstar.run()


if __name__ == "__main__":
    main()
