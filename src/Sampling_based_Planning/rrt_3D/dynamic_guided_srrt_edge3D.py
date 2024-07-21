"""
TODO
"""

import json
import os
import sys

import numpy as np


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_3D.env3D import CustomEnv
from rrt_3D.utils3D import getDist, sampleFree, nearest, isCollide
from rrt_3D.plot_util3D import (
    set_axes_equal,
    draw_block_list,
    draw_Spheres,
    draw_obb,
    draw_line,
    make_transparent,
)
from rrt_3D.mb_guided_srrt_edge3D import MbGuidedSrrtEdge
from rrt_3D.rrt3D import DynamicObj


class DynamicGuidedSrrtEdge(MbGuidedSrrtEdge):
    def __init__(self, t=0.1, m=10000):
        super().__init__(t, m)
        self.current_index = 0
        self.agent_positions = []
        self.agent_pos = None
        self.distance_travelled = 0
        self.time = 3

        self.dynamic_obs = []

        self.replanning_time = []

    def regrow(self):
        self.V.clear()
        self.E.clear()
        self.ellipsoid = None
        self.Path = None

        current_pos = self.agent_pos

        self.x0 = current_pos
        self.V = [current_pos]
        self.E = []

        result = self.run()

        if result:
            self.current_index = 0
            return self.Path
        else:
            return None

    def reconnect(self, path):
        current_pos = self.agent_pos
        goal_pos = path[self.current_index + 1]

        time_steps = int(self.time)

        for t in range(1, time_steps + 1):
            collision_detected = False
            for obj in self.dynamic_obs:
                future_pos = [
                    obj.current_pos[0] + obj.velocity[0] * t,
                    obj.current_pos[1] + obj.velocity[1] * t,
                    obj.current_pos[2] + obj.velocity[2] * t,
                ]

                original_pos = obj.current_pos
                obj.current_pos = future_pos

                # Check for collisions
                if self.in_dynamic_obj(current_pos, obj) or self.in_dynamic_obj(
                    goal_pos, obj
                ):
                    collision_detected = True

                obj.current_pos = original_pos

                if collision_detected:
                    break

            if not collision_detected:
                return True
        return False

    def move_dynamic_obs(self):
        """
        TODO
        """
        for obj in self.dynamic_obs:
            old, new = self.env.move_block(
                a=obj.velocity, block_to_move=obj.index, mode="translation"
            )

    def move(self, path, mps=6):
        """
        TODO change
        """
        if self.current_index >= len(path) - 1:
            return self.env.goal

        current_pos = self.agent_pos
        next_node = path[self.current_index + 1][0]

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

        if getDist(current_pos, new_pos) >= seg_distance:
            self.agent_pos = next_node
            self.current_index += 1
            return next_node

        return new_pos

    def main(self):
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)
        prev_coords = self.x0

        self.agent_pos = self.x0
        self.agent_positions.append(self.agent_pos)

        # Find initial path
        path = self.run()
        self.done = True
        t = 0

        if path:
            path = self.Path
            start = self.env.start
            goal = self.env.goal

            current = start

            current = np.array(current)
            GOAL = np.array(goal)
            while not np.array_equal(current, GOAL):
                self.move_dynamic_obs()

                new_coords = self.move(path)
                if new_coords is None:
                    if not self.reconnect():
                        new_path = self.regrow()
                        if not new_path:
                            self.agent_positions.append(self.agent_pos)
                            return False
                        else:
                            path = new_path

                    self.agent_positions.append(self.agent_pos)

                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords

                    self.distance_travelled += getDist(prev_coords, new_coords)
                    prev_coords = new_coords

                t += 1
            self.Path = path
            return True

        else:
            return False

    def corner_coords(self, x1, y1, z1, width, height, depth):
        x2 = x1 + width
        y2 = y1 + height
        z2 = z1 + depth
        return (x1, y1, z1, x2, y2, z2)

    def change_env(self, map_name, obs_name=None):
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.current_index = 0
            self.agent_positions = []
            self.agent_pos = None
            self.distance_travelled = 0
            self.dynamic_obs = []

            self.env = CustomEnv(data)

            if obs_name:
                self.set_dynamic_obs(obs_name)

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

                # TODO Add to env
                self.env.new_block_corners(new_obj.corners)


if __name__ == "__main__":
    rrt = DynamicGuidedSrrtEdge(1)
    rrt.change_env(
        "Evaluation/Maps/3D/block_map_25_3d/6_3d.json", "Evaluation/Maps/3D/obs.json"
    )
    res = rrt.main()
