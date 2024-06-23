"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import env


class Plotting:
    def __init__(self, x_start, x_goal):
        self.xI, self.xG = x_start, x_goal
        self.env = env.Env()
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle

    def animation(self, nodelist, path, name, animation=False):
        fig, ax = self.plot_grid(name)
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)
        plt.close(fig)

    def animation_connect(self, V1, V2, path, name):
        fig, ax = self.plot_grid(name)
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)
        plt.close(fig)

    def plot_grid(self, name):
        fig, ax = plt.subplots()

        for ox, oy, w, h in self.obs_bound:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h, edgecolor="black", facecolor="black", fill=True
                )
            )

        for ox, oy, w, h in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h, edgecolor="black", facecolor="gray", fill=True
                )
            )

        for ox, oy, r in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r, edgecolor="black", facecolor="gray", fill=True
                )
            )

        plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        plt.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect(
                        "key_release_event",
                        lambda event: [exit(0) if event.key == "escape" else None],
                    )
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].parent:
                    plt.plot([V1[k].x, V1[k].parent.x], [V1[k].y, V1[k].parent.y], "-g")
            if k < len2:
                if V2[k].parent:
                    plt.plot([V2[k].x, V2[k].parent.x], [V2[k].y, V2[k].parent.y], "-g")

            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], "-r", linewidth=2)
            plt.pause(0.01)
        plt.show()


class DynamicPlotting(Plotting):
    def __init__(self, x_start, x_goal, dynamic_objects, t):
        super().__init__(x_start, x_goal)
        self.dynamic_objects = dynamic_objects
        self.t = t
        for obj in self.dynamic_objects:
            obj.current_pos = obj.init_pos

    def update_dynamic_objects(self):
        for obj in self.dynamic_objects:
            obj.update_pos()

    def plot_dynamic_objects(self):
        for obj in self.dynamic_objects:
            rect = patches.Rectangle(
                (obj.current_pos[0], obj.current_pos[1]),
                obj.size[0],
                obj.size[1],
                edgecolor="red",
                facecolor="red",
                fill=True,
            )
            plt.gca().add_patch(rect)

    def animation(self, nodelist, path, name, animation=False):
        plt.ion()

        # Plot initial path
        fig, ax = plt.subplots()
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)
        plt.pause(1)
        plt.close(fig)

        for _ in range(self.t):
            plt.cla()
            self.plot_grid(name)
            self.plot_visited(nodelist, animation)
            self.plot_dynamic_objects()
            self.plot_path(path)
            self.update_dynamic_objects()
            plt.pause(0.1)
            plt.close(fig)
        plt.ioff()
        plt.show()
