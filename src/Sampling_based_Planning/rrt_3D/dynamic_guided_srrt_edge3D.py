import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in double_scalars",
)

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_3D.env3D import CustomEnv
from rrt_3D.utils3D import getDist, sampleFree, nearest, isCollide, visualization
from rrt_3D.mb_guided_srrt_edge3D import MbGuidedSrrtEdge


class DynamicGuidedSrrtEdge(MbGuidedSrrtEdge):
    def __init__(self, t=1.0, m=10000):
        super().__init__(t, m)
        self.maxiter = float("inf")
        self.replan_time = []
        self.flag = {}
        self.E = []
        self.V = []
        self.name = f"SRRT-Edge: {t}"

    def change_env(self, map_name, obs_name=None, size=None):
        super().change_env(map_name, obs_name, size)
        self.agent_positions = []
        self.current_index = 0
        self.Path = None
        self.ellipsoid = None
        self.replan_time = []
        self.flag = {}
        self.E = []
        self.V = []

    def corner_coords(self, x1, y1, z1, width, height, depth):
        x2 = x1 + width
        y2 = y1 + height
        z2 = z1 + depth
        return (x1, y1, z1, x2, y2, z2)

    def move(self, path, mps=6):
        if self.current_index >= len(path):
            return self.agent_pos

        current = self.agent_pos

        current_segment = path[self.current_index]

        next_pos = current_segment[1]

        seg_distance = getDist(current, next_pos)

        direction = (
            (next_pos[0] - current[0]) / seg_distance,
            (next_pos[1] - current[1]) / seg_distance,
            (next_pos[2] - current[2]) / seg_distance,
        )

        new_pos = (
            current[0] + direction[0] * mps,
            current[1] + direction[1] * mps,
            current[2] + direction[2] * mps,
        )

        if getDist(current, new_pos) >= seg_distance:
            self.agent_pos = next_pos
            self.current_index += 1
            return next_pos

        future_uav_positions = []
        PREDICTION_HORIZON = 3
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
        self.agent_pos = new_pos
        return new_pos

    def regrow(self):
        self.V.clear()
        self.E.clear()
        self.ellipsoid = None
        self.Path = None

        current_pos = self.agent_pos

        self.x0 = current_pos
        self.V = [current_pos]
        self.E = []

        result = self.planning()

        if result:
            self.current_index = 0
            return result[::-1]
        else:
            return None

    def reconnect(self, path):
        current_pos = self.agent_pos

        if self.current_index + 1 >= len(path):
            return False

        goal_pos = path[self.current_index][1]

        time_steps = int(round(self.t))

        for t in range(1, time_steps + 1):
            collision_detected = False
            for obj in self.dynamic_obs:
                future_pos = [
                    obj.current_pos[0] + obj.velocity[0] * t,
                    obj.current_pos[1] + obj.velocity[1] * t,
                    obj.current_pos[2] + obj.velocity[2] * t,
                ]

                future_agent_pos = [
                    current_pos[0] + (goal_pos[0] - current_pos[0]) * (t / time_steps),
                    current_pos[1] + (goal_pos[1] - current_pos[1]) * (t / time_steps),
                    current_pos[2] + (goal_pos[2] - current_pos[2]) * (t / time_steps),
                ]

                seg_distance = getDist(current_pos, future_agent_pos)

                if getDist(current_pos, future_pos) >= seg_distance:
                    break

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

    def run(self):
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)

        self.dynamic_obs = []

        start_time = time.time()
        path = self.planning()
        end_time = time.time() - start_time
        self.compute_time = end_time

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        self.agent_positions.append(tuple(self.x0))

        if path:
            path = path[::-1]
            start = self.env.start
            goal = self.env.goal

            current = start
            self.agent_pos = current

            prev = self.agent_pos
            same_counter = 0

            start_time = time.time()
            while tuple(self.agent_pos) != tuple(goal):
                if same_counter == 20:
                    return None

                for dynam_obj in self.dynamic_obs:
                    if self.in_dynamic_obj(tuple(self.agent_pos), dynam_obj):
                        return None

                self.move_dynamic_obs()
                new_coords = self.move(path)
                if new_coords[0] is None:
                    replan_time = time.time()
                    if not self.reconnect(path):
                        new_path = self.regrow()
                        if not new_path:
                            self.agent_positions.append(tuple(self.agent_pos))
                            return None
                        else:
                            self.current_index = 0
                            same_counter += 1
                            path = new_path

                    self.replan_time.append(time.time() - replan_time)
                    same_counter += 1
                    self.agent_positions.append(tuple(self.agent_pos))

                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords

                    if tuple(prev) == tuple(current):
                        same_counter += 1
                    else:
                        same_counter = 0

                    prev = current

            self.total_time = time.time() - start_time
            return self.agent_positions

        else:
            return None


if __name__ == "__main__":
    rrt = DynamicGuidedSrrtEdge(t=5)

    rrt.change_env(
        map_name="Evaluation/Maps/3D/main/block_20_3d.json",
        obs_name="Evaluation/Maps/3D/block_obs.json",
    )

    path = rrt.run()

    print(path)

    rrt.change_env(
        map_name="Evaluation/Maps/3D/main/block_2_3d.json",
        obs_name="Evaluation/Maps/3D/block_obs.json",
    )

    path = rrt.run()

    print(path)
