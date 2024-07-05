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
        fig = self.plot_grid(name)
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)

    def animation_connect(self, V1, V2, path, name):
        fig, ax = self.plot_grid(name)
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)
        plt.close(fig)

    def plot_grid(self, name, ax=None):
        if ax is None:
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
        return ax
        # plt.show()

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

    @staticmethod
    def plot_original_path(original_path):
        if len(original_path) != 0:
            plt.plot(
                [x[0] for x in original_path],
                [x[1] for x in original_path],
                "-p",
                linewidth=2,
            )
            plt.pause(0.01)
        plt.show()


class DynamicPlotting(Plotting):
    def __init__(self, x_start, x_goal, dynamic_objects, t, agent_pos, initial_path):
        super().__init__(x_start, x_goal)
        self.dynamic_objects = dynamic_objects
        self.t = t
        self.agent_pos = agent_pos
        self.initial_path = initial_path
        for obj in self.dynamic_objects:
            obj.current_pos = obj.init_pos

    def animation_connect(self, V1, V2, path, name):
        fig, ax = self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)
        self.plot_original_path(self.initial_path)
        plt.close()

    def update_dynamic_objects(self):
        for obj in self.dynamic_objects:
            old_pos = obj.current_pos
            obj.current_pos = [
                obj.current_pos[0] + obj.velocity[0],
                obj.current_pos[1] + obj.velocity[1],
            ]

            if not (0 <= obj.current_pos[0] < 1000 and 0 <= obj.current_pos[1] < 1000):
                obj.current_pos = old_pos

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

    def plot_agent(self, agent_pos):
        plt.plot(agent_pos[0], agent_pos[1], "bo")

    def animation(self, nodelist, path, name, animation=False):
        plt.ion()

        # Plot initial path
        fig, ax = plt.subplots()
        self.plot_grid(name, ax)
        self.plot_path(path)
        self.plot_dynamic_objects()
        self.plot_original_path(self.initial_path)
        plt.pause(1)

        for i in range(self.t):
            ax.clear()
            self.plot_grid(name, ax)
            self.plot_dynamic_objects()
            self.plot_path(path)
            self.plot_original_path(self.initial_path)
            self.plot_agent(self.agent_pos[i])
            self.update_dynamic_objects()
            fig.canvas.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show()
