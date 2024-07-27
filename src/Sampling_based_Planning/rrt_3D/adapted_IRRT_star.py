import json

import numpy as np
from utils3D import getDist, isCollide, near, nearest, steer
from informed_rrt_star3D import IRRT
from env3D import CustomEnv
from DynamicObj import DynamicObj


class AnytimeIRRTTStar(IRRT):
    def __init__(self, speed=6, time=float("inf")):
        super().__init__(False)

        self.N = 500

        self.agent_pos = None
        self.agent_positions = []
        self.current_index = 0
        self.speed = speed
        self.distance_travelled = 0
        self.dynamic_obs = []

    def change_env(self, map_name, obs_name=None):
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.V = []
            self.i = 0
            self.Path = []
            self.Parent = {}

            self.env = CustomEnv(data)

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
        while x != self.x0:
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
        # TODO add in future check for future object positions!
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

    def planning(self):
        """
        Adapted from yue qi's code for IRRT* in the original codebase.
        """
        self.V = [self.xstart]
        self.E = set()
        self.Xsoln = set()
        self.T = (self.V, self.E)

        c = 1
        while self.ind <= self.N:
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
                    self.Path, _ = self.path_from_point(xnew)
                    self.Xsoln.add(xnew)
            # update path
            self.ind += 1

        return self.Path

    def run(self):
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)
        prev_coords = self.x0

        path = self.planning()

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        if len(path) > 0:
            path = path[::-1]

            start = self.env.start
            end = self.env.goal

            self.agent_pos = start

            self.agent_positions.append(tuple(self.agent_pos))

            # Traverse the found path
            while tuple(self.agent_pos) != tuple(end):
                self.move_dynamic_obs()  # TODO
                new_coords = self.move(path)  # TODO
                if new_coords[0] is None:
                    # Replan from current pos
                    self.Parent = {}

                    self.xstart = self.agent_pos
                    self.x0 = self.agent_pos
                    self.ind = 0
                    self.i = 0

                    new_path = self.planning()

                    if len(path) > 0:
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

            return self.agent_positions

        else:
            return None


if __name__ == "__main__":
    rrt = AnytimeIRRTTStar()
    rrt.change_env("Evaluation/Maps/3D/block_map_25_3d/13_3d.json")

    path = rrt.run()
    # NOTE, path is in same format as it is for SRRT-Edge

    print(path)
