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
from matplotlib.animation import FuncAnimation
import numpy as np

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_2D import plotting, env


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
        ]

        return new_pos


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

        if time != float("inf"):
            self.name = f"D* Lite: {time}"
        else:
            self.name = "D* Lite"
        self.first_success = None
        self.time = time

        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.CalculateKey(self.s_goal)  # THIS IS OPEN
        self.visited = set()
        self.count = 0
        self.s_last = s_start

        # Data associated with traversal of the found path
        self.current_index = 0
        self.dynamic_objects = []
        self.speed = 6
        self.time_steps = 0
        self.agent_pos = self.s_start
        self.agent_positions = []
        self.traversed_path = []
        self.replan_time = []
        self.total_time = 0
        self.dobs_dir = None
        self.compute_time = None

    def euclidean_distance(self, point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)

        distance = np.linalg.norm(point1 - point2)

        return distance

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

    def ComputePath(self):
        start_time = time.time()

        start_pos = self.s_start

        while True:
            if (len(self.U)) == 0:
                return None
            if time.time() - start_time > self.time:
                if (
                    v >= self.CalculateKey(start_pos)
                    and self.rhs[start_pos] == self.g[start_pos]
                ):
                    break
                else:
                    return None

            s, v = self.TopKey()
            if (
                v >= self.CalculateKey(start_pos)
                and self.rhs[start_pos] == self.g[start_pos]
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

        # return self.path_to_end()
        return self.extract_path()

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
        dynamic_covered_cells = self.Env.dynamic_obs_cells
        if (
            s_start in self.obs
            or s_end in self.obs
            or s_start in dynamic_covered_cells
            or s_end in dynamic_covered_cells
        ):
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if (
                s1 in self.obs
                or s2 in self.obs
                or s1 in dynamic_covered_cells
                or s2 in dynamic_covered_cells
            ):
                return True

        return False

    def get_neighbor(self, s):
        dynamic_covered_cells = self.Env.dynamic_obs_cells
        nei_list = set()
        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if (
                s_next not in self.obs
                and s_next not in dynamic_covered_cells
                and s_next in self.rhs.keys()
            ):
                nei_list.add(s_next)

        return nei_list

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for k in range(1000):
            g_list = {}
            for x in self.get_neighbor(s):
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]

            if not g_list:
                return None
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
        plt.show()

    def change_env(self, map_name, obj_dir=None):
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.s_start = tuple(data["agent"])
            self.s_goal = tuple(data["goal"])
            self.Env = env.CustomEnv(data)
            self.dynamic_objects = []
            self.current_index = 0
            self.traversed_path = []
            self.replan_time = []

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

            self.agent_pos = self.s_start

            self.dobs_dir = obj_dir

            return self.Env
        else:
            print("Error, map not found")

    def move(self, path, mps=6):
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.s_goal

        current = self.agent_pos
        next = path[self.current_index + 1]

        seg_distance = self.euclidean_distance(current, next)

        direction = (
            (next[0] - current[0]) / seg_distance,
            (next[1] - current[1]) / seg_distance,
        )

        new_pos = (current[0] + direction[0] * mps, current[1] + direction[1] * mps)

        if self.euclidean_distance(current, new_pos) >= seg_distance:
            # Calculate vectors and check if in the same direction
            v1 = np.array(next) - np.array(current)
            v2 = np.array(new_pos) - np.array(next)
            dot_product = np.dot(v1, v2)

            mag_v1 = np.linalg.norm(v1)
            mag_v2 = np.linalg.norm(v2)

            same_dir = np.isclose(dot_product, mag_v1 * mag_v2)

            if same_dir:
                # Move the agent far forward without turning
                count = 0
                while self.current_index < len(path) - 1:
                    next = path[self.current_index + 1]
                    seg_distance = self.euclidean_distance(current, next)
                    direction = (
                        (next[0] - current[0]) / seg_distance,
                        (next[1] - current[1]) / seg_distance,
                    )
                    new_pos = (
                        current[0] + direction[0] * mps,
                        current[1] + direction[1] * mps,
                    )
                    v1 = np.array(next) - np.array(current)
                    v2 = np.array(new_pos) - np.array(next)
                    dot_product = np.dot(v1, v2)
                    mag_v1 = np.linalg.norm(v1)
                    mag_v2 = np.linalg.norm(v2)
                    same_dir = np.isclose(dot_product, mag_v1 * mag_v2)
                    if not same_dir or count >= self.speed - 1:
                        break
                    current = next
                    self.agent_pos = current
                    self.current_index += 1
                    count += 1
            else:
                # Move to the next node and update position
                self.agent_pos = next
                self.current_index += 1
        else:
            self.agent_pos = new_pos

        return self.agent_pos

    def update_object_positions(self):
        self.Env.dynamic_obs_cells = set()
        for i, object in enumerate(self.dynamic_objects):
            prev_pos = object.current_pos
            new_pos = object.update_pos()

            new_pos = prev_pos

            object.old_pos = object.current_pos
            object.current_pos = new_pos

            self.Env.update_dynamic_obj_pos(i, new_pos[0], new_pos[1])

    def get_affected_cells(self, position, width, height):
        x, y = position

        x_max = self.Env.x_range
        y_max = self.Env.y_range

        affected_cells = [
            (x + dx, y + dy)
            for dx in range(width)
            for dy in range(height)
            if 0 <= (x + dx) < x_max and 0 <= (y + dy) < y_max
        ]

        return affected_cells

    def update_costs(self):
        current_pos = self.agent_pos
        SIGHT = 3

        new_cells = set()
        old_cells = set()

        sight_range = range(-SIGHT, SIGHT + 1)
        dynamic_obj_in_sight = False

        for dx in sight_range:
            for dy in sight_range:
                if dx == 0 and dy == 0:
                    continue

                check_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if check_pos in self.Env.dynamic_obs_cells:
                    dynamic_obj_in_sight = True
                    new_cells.add(check_pos)

        if dynamic_obj_in_sight:
            replan_start_time = time.time()

            self.s_start = self.agent_pos

            path = [self.s_start]

            self.km += self.h(self.s_last, self.s_start)

            self.s_last = self.s_start

            for obj in self.dynamic_objects:
                old_pos = obj.old_pos
                new_pos = obj.current_pos
                width, height = obj.size

                # Determine affected cells by old and new positions of the object
                old_cells.update(self.get_affected_cells(old_pos, width, height))
                new_cells.update(self.get_affected_cells(new_pos, width, height))

            all_cells = new_cells.union(old_cells)

            for cell in old_cells:
                self.UpdateVertex(cell)

            for cell in new_cells:
                self.g[cell] = float("inf")
                self.rhs[cell] = float("inf")

            # Update neighbors of all affected cells
            for cell in all_cells:
                for neighbour in self.get_neighbor(cell):
                    self.UpdateVertex(neighbour)

            self.visited = set()
            self.ComputePath()

            replan_end_time = time.time() - replan_start_time
            self.replan_time.append(replan_end_time)

        path = self.extract_path()

        self.current_index = 0
        return path

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
                new_obj.init_pos = obj["position"]
                new_obj.size = obj["size"]
                new_obj.init_pos = new_obj.current_pos

                self.dynamic_objects.append(new_obj)

                # Add to the env
                self.Env.dynamic_obs.append(new_obj)
        else:
            print("Error, dynamic objects could not be loaded")

    def plot(self, path):
        self.Plot.plot_grid("D* Lite")
        if path:
            self.plot_path(path)
        else:
            self.plot_path(self.agent_positions)
        plt.show()

    def run(self):
        """
        Runs the simulation involving pathfinding, traversing the found path,
        and reacting to dynamic objects.
        """
        self.time_steps = 0

        start_time = time.time()
        path = self.ComputePath()

        end_time = time.time() - start_time
        self.compute_time = end_time

        self.initial_path = path

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        self.s_last = self.s_start
        start_time = time.time()

        if path:
            GOAL = path[-1]

            while self.agent_pos != GOAL:
                if (
                    self.g[self.agent_pos] == float("inf")
                    or self.agent_pos in self.Env.dynamic_obs_cells
                ):
                    return None

                self.update_object_positions()
                path = self.update_costs()

                if path is None:
                    return None

                self.move(path)
                self.agent_positions.append(self.agent_pos)

                self.time_steps += 1

                self.s_start = self.agent_pos

                print(self.agent_pos)

        end_time = time.time() - start_time
        self.total_time = end_time

        return self.agent_positions

    def plot_traversal(self):
        plotter = plotting.DynamicPlotting(
            self.s_start,
            self.s_goal,
            self.dynamic_objects,
            self.time_steps,
            self.agent_positions,
            self.initial_path,
        )

        plotter.env = self.Env
        plotter.obs = self.Env.obs
        plotter.xI = self.initial_path[0]
        plotter.xG = self.s_goal

        plotter.animation(self.agent_positions, "D* Lite Original Path vs Path Taken")


def main():
    s_start = (5, 5)
    s_goal = (989, 888)

    dstar = DStar(
        s_start,
        s_goal,
        "euclidian",
    )
    # Block 12 to debug!
    dstar.change_env(
        "Evaluation/Maps/2D/main/block_12.json",
        "Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json",
    )

    path = dstar.run()

    dstar.plot_traversal()


if __name__ == "__main__":
    main()
