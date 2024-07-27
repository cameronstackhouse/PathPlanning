import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_3D.env3D import CustomEnv
from rrt_3D.utils3D import getDist, sampleFree, nearest, isCollide, visualization
from rrt_3D.mb_guided_srrt_edge3D import MbGuidedSrrtEdge


class DynamicGuidedSrrtEdge(MbGuidedSrrtEdge):
    def __init__(self, t=1.0, m=10000):
        super().__init__(t, m)

    def corner_coords(self, x1, y1, z1, width, height, depth):
        x2 = x1 + width
        y2 = y1 + height
        z2 = z1 + depth
        return (x1, y1, z1, x2, y2, z2)

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

        start_time = time.time()
        path = self.planning()
        start_time = time.time() - start_time

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        start_time = time.time()

        if path:
            path = path[::-1]
            self.compute_time = start_time
            start = self.env.start
            goal = self.env.goal

            current = start
            self.agent_pos = current

            while tuple(self.agent_pos) != tuple(goal):

                self.move_dynamic_obs()
                new_coords = self.move(path)
                if new_coords[0] is None:
                    if not self.reconnect():
                        new_path = self.regrow()
                        if not new_path:
                            self.agent_positions.append(tuple(self.agent_pos))
                            return False
                        else:
                            self.current_index = 0
                            path = new_path

                    self.agent_positions.append(tuple(self.agent_pos))

                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords

            return self.agent_positions

        else:
            return None


if __name__ == "__main__":
    rrt = DynamicGuidedSrrtEdge(t=5)
    rrt.change_env("Evaluation/Maps/3D/block_map_25_3d/12_3d.json")

    # "Evaluation/Maps/3D/obs.json"

    res = rrt.planning()

    print(res)
