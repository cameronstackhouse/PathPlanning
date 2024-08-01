"""
INFORMED_RRT_STAR 2D
@author: huiming zhou
"""

import copy
import json
import os
import sys
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import psutil
from scipy.spatial.transform import Rotation as Rot
import matplotlib.patches as patches

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import env, plotting, utils
from rrt_2D.rrt import DynamicObj


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.coords = n


class IRrtStar:
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        goal_sample_rate,
        search_radius,
        iter_max,
        time=float("inf"),
        obj_dir=None,
    ):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.delta = self.utils.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.start_rect = None

        self.V = [self.x_start]
        self.X_soln = set()
        self.path = None
        self.peak_cpu = 0
        self.name = "Informed RRT*"
        self.first_success = None
        self.time = time
        self.dobs_dir = obj_dir
        self.current_index = 0
        self.agent_positions = []
        self.time_steps = 0
        self.distance_travelled = 0
        self.speed = 60
        self.dynamic_objects = []
        self.initial_start = None
        self.initial_path = None
        self.total_time = 0
        self.compute_time = 0
        self.replan_time = []

    def init(self):
        cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array(
            [
                [(self.x_start.x + self.x_goal.x) / 2.0],
                [(self.x_start.y + self.x_goal.y) / 2.0],
                [0.0],
            ]
        )
        x_best = self.x_start

        return theta, cMin, xCenter, C, x_best

    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf

        start = time.time()

        for k in range(self.iter_max):
            if self.X_soln:
                cost = {node: self.Cost(node) for node in self.X_soln}
                x_best = min(cost, key=cost.get)
                c_best = cost[x_best]

            if time.time() - start > self.time:
                break

            x_rand = self.Sample(c_best, dist, x_center, C)
            x_nearest = self.Nearest(self.V, x_rand)
            x_new = self.Steer(x_nearest, x_rand)

            if x_new and not self.utils.is_collision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new)
                c_min = self.Cost(x_nearest) + self.Line(x_nearest, x_new)
                self.V.append(x_new)

                # choose parent
                for x_near in X_near:
                    c_new = self.Cost(x_near) + self.Line(x_near, x_new)
                    if c_new < c_min:
                        x_new.parent = x_near
                        c_min = c_new

                # rewire
                for x_near in X_near:
                    c_near = self.Cost(x_near)
                    c_new = self.Cost(x_new) + self.Line(x_new, x_near)
                    if c_new < c_near:
                        x_near.parent = x_new

                if self.InGoalRegion(x_new):
                    if not self.utils.is_collision(x_new, self.x_goal):
                        if len(self.X_soln) == 0:
                            self.first_success = time.time() - start
                        self.X_soln.add(x_new)

        self.path = self.ExtractPath(x_best)

        if self.path and self.utils.is_collision(
            Node(self.path[-1]), Node(self.path[-2])
        ):
            return None

        return self.path

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)
        node_new = Node(
            (x_start.x + dist * math.cos(theta), x_start.y + dist * math.sin(theta))
        )
        node_new.parent = x_start

        return node_new

    def Near(self, nodelist, node):
        n = len(nodelist) + 1
        r = 50 * math.sqrt((math.log(n) / n))

        dist_table = [(nd.x - node.x) ** 2 + (nd.y - node.y) ** 2 for nd in nodelist]
        X_near = [
            nodelist[ind]
            for ind in range(len(dist_table))
            if dist_table[ind] <= r**2
            and not self.utils.is_collision(nodelist[ind], node)
        ]

        return X_near

    def Sample(self, c_max, c_min, x_center, C):
        if c_max < np.inf:
            r = [
                c_max / 2.0,
                math.sqrt(c_max**2 - c_min**2) / 2.0,
                math.sqrt(c_max**2 - c_min**2) / 2.0,
            ]
            L = np.diag(r)

            while True:
                x_ball = self.SampleUnitBall()
                x_rand = np.dot(np.dot(C, L), x_ball) + x_center
                if (
                    self.x_range[0] + self.delta
                    <= x_rand[0]
                    <= self.x_range[1] - self.delta
                    and self.y_range[0] + self.delta
                    <= x_rand[1]
                    <= self.y_range[1] - self.delta
                ):
                    break
            x_rand = Node((x_rand[(0, 0)], x_rand[(1, 0)]))
        else:
            x_rand = self.SampleFreeSpace()

        return x_rand

    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x**2 + y**2 < 1:
                return np.array([[x], [y], [0.0]])

    def SampleFreeSpace(self):
        delta = self.delta

        if np.random.random() > self.goal_sample_rate:
            return Node(
                (
                    np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                    np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                )
            )

        return self.x_goal

    def ExtractPath(self, node):
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            path.append([node.x, node.y])
            node = node.parent

        path.append([self.x_start.x, self.x_start.y])

        return path

    def InGoalRegion(self, node):
        if self.Line(node, self.x_goal) < self.step_len:
            return True

        return False

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array(
            [[(x_goal.x - x_start.x) / L], [(x_goal.y - x_start.y) / L], [0.0]]
        )
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    @staticmethod
    def Nearest(nodelist, n):
        return nodelist[
            int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2 for nd in nodelist]))
        ]

    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    def Cost(self, node):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        cost = 0.0
        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def animation(self, x_center=None, c_best=None, dist=None, theta=None):
        plt.cla()
        self.plot_grid("Informed rrt*, N = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

        for node in self.V:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)

        plt.pause(0.01)

    def plot_grid(self, name):

        for ox, oy, w, h in self.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h, edgecolor="black", facecolor="black", fill=True
                )
            )

        for ox, oy, w, h in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h, edgecolor="black", facecolor="gray", fill=True
                )
            )

        for ox, oy, r in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r, edgecolor="black", facecolor="gray", fill=True
                )
            )

        plt.plot(self.x_start.x, self.x_start.y, "bs", linewidth=3)
        plt.plot(self.x_goal.x, self.x_goal.y, "rs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        a = math.sqrt(c_best**2 - dist**2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler("z", -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, ".b")
        plt.plot(px, py, linestyle="--", color="darkorange", linewidth=2)

    def change_env(self, map_name, obs_name=None):
        """
        Method which changes the env based on custom map input.
        """
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.x_start = Node(data["agent"])
            self.x_goal = Node(data["goal"])
            self.V = [self.x_start]
            self.X_soln = set()
            self.path = None

            self.dynamic_objects = []
            self.agent_positions = []
            self.time_steps = 0
            self.current_index = 0
            self.replan_time = []

            # Initialize the new custom environment
            self.env = env.CustomEnv(data)

            # Update plotting with new environment details
            self.plotting = plotting.Plotting(data["agent"], data["goal"])
            self.plotting.env = self.env
            self.plotting.xI = data["agent"]
            self.plotting.xG = data["goal"]
            self.plotting.obs_bound = self.env.obs_boundary
            self.plotting.obs_circle = self.env.obs_circle
            self.plotting.obs_rectangle = self.env.obs_rectangle

            # Update utilities with new environment details
            self.utils = utils.Utils()
            self.utils.env = self.env
            self.utils.obs_boundary = self.env.obs_boundary
            self.utils.obs_circle = self.env.obs_circle
            self.utils.obs_rectangle = self.env.obs_rectangle

            # Update environment properties
            self.x_range = self.env.x_range
            self.y_range = self.env.y_range
            self.obs_circle = self.env.obs_circle
            self.obs_rectangle = self.env.obs_rectangle
            self.obs_boundary = self.env.obs_boundary

            self.agent_pos = data["agent"]

            # Add dynamic obs if needed
            # if obs_name:
            #     self.set_dynamic_obs(obs_name)

            self.first_success = None

        else:
            print("Error, map not found")

    def set_dynamic_obs(self, filename):
        """
        Adds dynamic objects to the environment given a JSON filename
        containing the data of the dynamic objects.
        """
        # Loads the objects
        obj_json = None
        with open(filename) as f:
            obj_json = json.load(f)

        # Adds each object to the environment
        if obj_json:
            for obj in obj_json["objects"]:
                new_obj = DynamicObj()
                new_obj.velocity = obj["velocity"]
                new_obj.current_pos = obj["position"]
                new_obj.size = obj["size"]
                new_obj.init_pos = new_obj.current_pos

                self.env.add_rect(
                    new_obj.current_pos[0],
                    new_obj.current_pos[1],
                    new_obj.size[0],
                    new_obj.size[1],
                )

                new_obj.index = len(self.env.obs_rectangle) - 1
                self.dynamic_objects.append(new_obj)

        else:
            print("Error, dynamic objects could not be loaded")

    def in_dynamic_obj(self, node, obj):
        x, y = node.coords
        x0, y0 = obj.current_pos
        width, height = obj.size
        return (x0 <= x < x0 + width) and (y0 <= y < y0 + height)

    def move(self, path, mps=6):
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.s_goal.coords

        current_pos = self.agent_pos
        next_node = path[self.current_index + 1]

        seg_distance = self.utils.euclidian_distance(current_pos, next_node)

        direction = (
            (next_node[0] - current_pos[0]) / seg_distance,
            (next_node[1] - current_pos[1]) / seg_distance,
        )

        new_pos = (
            current_pos[0] + direction[0] * mps,
            current_pos[1] + direction[1] * mps,
        )

        # Checks for overshoot
        if self.utils.euclidian_distance(current_pos, new_pos) >= seg_distance:
            self.agent_pos = next_node
            self.current_index += 1
            return next_node

        future_uav_positions = []
        PREDICTION_HORIZON = 4
        for t in range(1, PREDICTION_HORIZON):
            future_pos = (
                current_pos[0] + direction[0] * mps * t,
                current_pos[1] + direction[1] * mps * t,
            )

            if self.utils.euclidian_distance(current_pos, future_pos) >= seg_distance:
                break

            future_uav_positions.append(future_pos)

        for future_pos in future_uav_positions:
            for dynamic_object in self.dynamic_objects:
                dynamic_future_pos = dynamic_object.predict_future_positions(
                    PREDICTION_HORIZON
                )

                for pos in dynamic_future_pos:
                    original_pos = dynamic_object.current_pos
                    dynamic_object.current_pos = pos

                    if self.in_dynamic_obj(Node(future_pos), dynamic_object):
                        dynamic_object.current_pos = original_pos
                        return [None, None]

                    dynamic_object.current_pos = original_pos

        return new_pos

    def update_object_positions(self, time_steps=1):
        """
        Updates the position of dynamic objects over one timestep.
        The object moves in a fixed direction and comes to an immediate stop
        if a fixed object is detected.

        :param: time_steps: the number of time steps in the future to predict the object positions.
        """
        for object in self.dynamic_objects:
            # Attempt to move in direction of travel
            prev_pos = object.current_pos
            new_pos = object.update_pos()
            if not (
                0 <= new_pos[0] < self.x_range[1] and 0 <= new_pos[1] < self.y_range[1]
            ):
                new_pos = prev_pos
            object.current_pos = new_pos

            self.env.update_obj_pos(object.index, new_pos[0], new_pos[1])
            self.utils.env.update_obj_pos(object.index, new_pos[0], new_pos[1])

    def run(self):
        """ """
        self.initial_start = self.x_start
        self.start_rect = copy.deepcopy(self.env.obs_rectangle)
        prev_coords = self.x_start.coords
        start_time = time.time()
        global_path = self.planning()
        end_time = time.time() - start_time
        self.compute_time = end_time
        self.initial_path = global_path

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        start_time = time.time()

        TIMEOUT = 60

        if global_path:
            global_path = global_path[::-1]
            current = global_path[self.current_index]
            GOAL = global_path[-1]

            current = np.array(current)
            GOAL = np.array(GOAL)

            while not np.array_equal(self.agent_pos, GOAL):
                if TIMEOUT - start_time <= 0:
                    return None

                self.update_object_positions()
                new_coords = self.move(global_path, self.speed)

                if new_coords[0] is None:
                    replan_time = time.time()
                    new_path = self.planning()
                    replan_time = time.time() - replan_time

                    self.replan_time.append(replan_time)
                    if not new_path:
                        self.agent_positions.append(self.agent_pos)
                        return None
                    else:
                        global_path = new_path[::-1]
                        self.current_index = 0
                        self.agent_positions.append(self.agent_pos)
                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords

                    self.distance_travelled += self.utils.euclidian_distance(
                        prev_coords, new_coords
                    )
                    prev_coords = new_coords

                self.time_steps += 1

            self.path = global_path
            self.total_time = time.time() - start_time
            return self.agent_positions
        else:
            self.total_time = time.time() - start_time
            return None

    def plot(self):
        dynamic_objects = self.dynamic_objects

        nodelist = self.V
        path = self.path

        plotter = plotting.DynamicPlotting(
            self.x_start.coords,
            self.x_goal.coords,
            dynamic_objects,
            self.time_steps,
            self.agent_positions,
            self.initial_path,
        )

        plotter.env = self.env
        plotter.obs_bound = self.env.obs_boundary
        plotter.obs_circle = self.env.obs_circle
        plotter.obs_rectangle = self.env.obs_rectangle

        plotter.animation(nodelist, path, "Test", animation=True)


def main():
    x_start = (18, 8)  # Starting node
    x_goal = (809, 909)  # Goal node

    rrt_star = IRrtStar(
        x_start,
        x_goal,
        10,
        0.10,
        12,
        5000,
        obj_dir="Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json",
    )

    # rrt_star.change_env("Evaluation/Maps/2D/main/house_11.json")
    # path = rrt_star.planning()

    rrt_star.change_env("Evaluation/Maps/2D/main/block_20.json")
    success = rrt_star.run()
    print(success)

    if success:
        rrt_star.plot()


if __name__ == "__main__":
    main()
