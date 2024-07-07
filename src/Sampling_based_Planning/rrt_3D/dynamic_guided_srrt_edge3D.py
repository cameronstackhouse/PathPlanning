"""
TODO
"""

import os
import sys

import numpy as np


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_3D.env3D import env
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


class DynamicGuidedSrrtEdge(MbGuidedSrrtEdge):
    def __init__(self, t=0.1, m=10000):
        super().__init__(t, m)
        self.current_index = 0
        self.agent_positions = []
        self.agent_pos = None
        self.distance_travelled = 0

    def regrow(self):
        pass

    def reconnect(self):
        pass

    def update_objs(self):
        pass

    def FindAffectedEdges(self, obstacle):
        Affectededges = []
        for e in self.E:
            child, parent = e.node_1, e.node_2
            collide, _ = isCollide(self, child, parent)
            if collide:
                Affectededges.append(e)
        return Affectededges

    def PathisInvalid(self, path):
        for edge in path:
            if (
                self.flag[tuple(edge[0])] == "Invalid"
                or self.flag[tuple(edge[1])] == "Invalid"
            ):
                return True

    def InvalidateNodes(self, obstacle):
        Edges = self.FindAffectedEdges(obstacle)
        for edge in Edges:
            qe = self.ChildEndpointNode(edge)
            self.flag[qe] = "Invalid"

    def move(self, path, mps=6):
        # if self.PathisInvalid(path):
        #     return None

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

    def Main(self):
        self.x0 = tuple(self.env.goal)
        self.xt = tuple(self.env.start)
        prev_coords = self.x0
        self.flag[tuple(self.x0)] = "Valid"

        self.agent_pos = self.x0

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
                self.update_objs()

                new_coords = self.move(path)
                print(new_coords)
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


if __name__ == "__main__":
    rrt = DynamicGuidedSrrtEdge(1)
    res = rrt.Main()
    print(res)
