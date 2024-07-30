import json
import time

import numpy as np
from utils3D import getDist, isCollide, near, nearest, steer
from informed_rrt_star3D import IRRT
from env3D import CustomEnv
from DynamicObj import DynamicObj


class AnytimeIRRTTStar(IRRT):
    def __init__(self, speed=6, time=float("inf")):
        super().__init__(False)

        self.N = 50000
        self.stepsize = 10

        self.agent_pos = None
        self.agent_positions = []
        self.current_index = 0
        self.speed = speed
        self.distance_travelled = 0
        self.dynamic_obs = []
        self.time = time
        self.compute_time = None
        self.replan_time = []
        self.total_time = None

    def in_dynamic_obj(self, pos, obj):
        x, y, z = pos
        x0, y0, z0 = obj.current_pos
        width, height, depth = obj.size
        return (
            (x0 <= x <= x0 + width)
            and (y0 <= y <= y0 + height)
            and (z0 <= z <= z0 + depth)
        )

    def corner_coords(self, x1, y1, z1, width, height, depth):
        x2 = x1 + width
        y2 = y1 + height
        z2 = z1 + depth
        return (x1, y1, z1, x2, y2, z2)

    def change_env(self, map_name, obs_name=None, size=100):
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.V = []
            self.i = 0
            self.Path = []
            self.Parent = {}
            self.ind = 0
            self.current_index = 0

            self.env = CustomEnv(data, zmax=size, xmax=size, ymax=size)

            self.xstart = tuple(self.env.start)
            self.xgoal = tuple(self.env.goal)

            self.x0 = self.xstart
            self.xt = self.xgoal

            self.dobs_dir = obs_name

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

    def path_from_point(self, point, dist=0):
        path = [np.array([point, self.xt])]
        dist += getDist(point, self.xt)
        x = point
        while tuple(x) != tuple(self.x0):
            x2 = self.Parent[x]
            path.append(np.array([x2, x]))
            dist += getDist(x, x2)
            x = x2
        return path, dist

    def move_dynamic_obs(self):
        for obj in self.dynamic_obs:
            old, new = self.env.move_block(
                a=obj.velocity, block_to_move=obj.index, mode="translation"
            )
            obj.current_pos = obj.update_pos()

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

    def planning(self):
        """
        Adapted from yue qi's code for IRRT* in the original codebase.

        The planning algorithm is now an anytime algorithm, bounded
        by either time or max iteraions.
        """
        self.V = [self.xstart]
        self.E = set()
        self.Xsoln = set()
        self.T = (self.V, self.E)
        self.ind = 0

        best_path = None
        best_path_dist = float("inf")

        c = 1
        start_time = time.time()
        while self.ind <= self.N:
            current_time = time.time()

            if current_time - start_time > self.time:
                break

            if len(self.Xsoln) == 0:
                cbest = np.inf
            else:
                cbest = min({self.cost(xsln) for xsln in self.Xsoln})
            xrand = self.Sample(self.xstart, self.xgoal, cbest)
            xnearest = nearest(self, xrand)
            xnew, dist = steer(self, xnearest, xrand)

            collide, _ = isCollide(self, xnearest, xnew, dist=dist)
            if not collide:
                self.V.append(xnew)
                Xnear = near(self, xnew)
                xmin = xnearest
                cmin = self.cost(xmin) + c * self.line(xnearest, xnew)
                for xnear in Xnear:
                    xnear = tuple(xnear)
                    cnew = self.cost(xnear) + c * self.line(xnear, xnew)
                    if cnew < cmin:
                        collide, _ = isCollide(self, xnear, xnew)
                        if not collide:
                            xmin = xnear
                            cmin = cnew
                self.E.add((xmin, xnew))
                self.Parent[xnew] = xmin

                for xnear in Xnear:
                    xnear = tuple(xnear)
                    cnear = self.cost(xnear)
                    cnew = self.cost(xnew) + c * self.line(xnew, xnear)
                    # rewire
                    if cnew < cnear:
                        collide, _ = isCollide(self, xnew, xnear)
                        if not collide:
                            xparent = self.Parent[xnear]
                            self.E.difference_update((xparent, xnear))
                            self.E.add((xnew, xnear))
                            self.Parent[xnear] = xnew
                self.i += 1
                if self.InGoalRegion(xnew):
                    self.done = True
                    self.Parent[self.xgoal] = xnew
                    new_path, D = self.path_from_point(xnew)

                    if D < best_path_dist:
                        best_path_dist = D
                        best_path = new_path
                        self.Path = new_path

                    self.Xsoln.add(xnew)
            self.ind += 1

        return best_path

    def run(self):
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)
        prev_coords = self.x0

        planning_time = time.time()
        path = self.planning()
        planning_time = time.time() - planning_time
        self.compute_time = planning_time
        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        if path:
            path = path[::-1]

            start = self.env.start
            end = self.env.goal

            self.agent_pos = start

            self.agent_positions.append(tuple(self.agent_pos))

            # Traverse the found path
            start_time = time.time()
            while tuple(self.agent_pos) != tuple(end):
                self.move_dynamic_obs()
                new_coords = self.move(path, self.speed)
                if new_coords[0] is None:
                    start_replan = time.time()
                    # Replan from current pos
                    self.Parent = {}

                    self.xstart = self.agent_pos
                    self.x0 = self.agent_pos
                    self.ind = 0
                    self.i = 0

                    new_path = self.planning()
                    end_replan = time.time() - start_replan

                    self.replan_time.append(end_replan)

                    if path:
                        path = new_path[::-1]
                        self.current_index = 0
                        self.agent_positions.append(tuple(self.agent_pos))
                    else:
                        self.agent_positions.append(tuple(self.agent_pos))
                        return None
                else:
                    self.agent_positions.append(tuple(new_coords))
                    self.agent_pos = new_coords

                    self.distance_travelled += getDist(prev_coords, new_coords)
                    prev_coords = new_coords

            self.total_time = time.time() - start_time
            return self.agent_positions

        else:
            return None


if __name__ == "__main__":
    rrt = AnytimeIRRTTStar(time=1)
    rrt.change_env(
        "Evaluation/Maps/3D/main/block_19_3d.json",
        obs_name="Evaluation/Maps/3D/block_obs.json",
    )
    res = rrt.run()

    print(res)

    rrt.change_env(
        "Evaluation/Maps/3D/main/block_8_3d.json",
        obs_name="Evaluation/Maps/3D/block_obs.json",
    )
    
    res = rrt.run()
    
    print(res)
