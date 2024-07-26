import queue
import time
from matplotlib import pyplot as plt
import numpy as np
from DstarLite3D import D_star_Lite

from Search_3D.Octree import Octree
from Search_3D.utils3D import children_non_uniform, cost, isinbound


class ADStarLite(D_star_Lite):
    def __init__(self, resolution=1):
        self.name = "AD* Lite"
        super().__init__(resolution)
        self.octree = None
        self.speed = 6

    def change_env(self, map_name, obs_name=None):
        new_env = super().change_env(map_name, obs_name)
        self.octree = Octree(new_env)

        self.rhs = {}
        self.g = {}
        self.leaf_nodes = {}
        for leaf in self.octree.leafs:
            center = (
                leaf.x + leaf.width // 2,
                leaf.y + leaf.height // 2,
                leaf.z + leaf.depth // 2,
            )

            self.rhs[center] = float("inf")
            self.g[center] = float("inf")
            self.leaf_nodes[center] = leaf

        self.rhs = {self.xt: 0}
        self.h = {}
        self.OPEN = queue.MinheapPQ()
        self.V = set()
        self.OPEN.put(self.xt, self.CalculateKey(self.xt))

    def updatecost(self, range_changed=None, new=None, old=None, mode=False):
        CHANGED = set()
        for xi in self.CLOSED:
            if isinbound(old, xi, mode) or isinbound(new, xi, mode):
                newchildren = set(children_non_uniform(self, xi))  # B
                self.CHILDREN[xi] = newchildren
                for xj in newchildren:
                    self.COST[xi][xj] = cost(self, xi, xj)
                CHANGED.add(xi)
        return CHANGED

    def getcost(self, xi, xj):
        # use a LUT for getting the costd
        if xi not in self.COST:
            for xj, xjcost in children_non_uniform(self, xi, settings=1):
                self.COST[xi][xj] = cost(self, xi, xj, xjcost)
        # this might happen when there is a node changed.
        if xj not in self.COST[xi]:
            self.COST[xi][xj] = cost(self, xi, xj)
        return self.COST[xi][xj]

    def getchildren(self, xi):
        if xi not in self.CHILDREN:
            allchild = children_non_uniform(self, xi)
            self.CHILDREN[xi] = set(allchild)
        return self.CHILDREN[xi]

    def in_dobs(self, point):
        for obj in self.dynamic_obs:
            if obj.contains_point(point):
                return True

        return False

    def update(self, path):
        current_pos = self.agent_pos
        SIGHT = 3

        sight_range = range(-SIGHT, SIGHT + 1)

        repartition = False
        affected_leafs = set()
        for dx in sight_range:
            for dy in sight_range:
                for dz in sight_range:
                    check_pos = (
                        current_pos[0] + dx,
                        current_pos[1] + dy,
                        current_pos[2] + dz,
                    )

                    if self.in_dobs(check_pos):
                        repartition = True
                        leaf_containing_point = self.octree.find_leaf_containing_point(
                            check_pos
                        )

                        if leaf_containing_point:
                            affected_leafs.add(leaf_containing_point)

        if repartition:
            replan_time = time.time()
            self.update_costs_and_queue(affected_leafs)
            new_path = self.plan_new_path()
            replan_time = time.time() - replan_time
            self.replan_time.append(replan_time)
            return new_path
        else:
            return path

    def update_costs_and_queue(self, affected_leafs):
        old_cells = set()
        new_cells = {}

        for leaf in affected_leafs:
            for x in range(leaf.x, leaf.x + leaf.width):
                for y in range(leaf.y, leaf.y + leaf.height):
                    for z in range(leaf.z, leaf.z + leaf.depth):
                        node_center = (
                            x + leaf.width // 2,
                            y + leaf.height // 2,
                            z + leaf.width // 2,
                        )

                        old_cells.add(node_center)

        for leaf in affected_leafs:
            self.octree.partition(leaf)

        if len(affected_leafs) > 0:
            self.octree.update_leafs()

            for leaf in self.octree.leafs:
                for x in range(leaf.x, leaf.x + leaf.width):
                    for y in range(leaf.y, leaf.y + leaf.height):
                        for z in range(leaf.z, leaf.z + leaf.depth):
                            midpoint = (
                                x + leaf.width // 2,
                                y + leaf.height // 2,
                                z + leaf.width // 2,
                            )
                            new_cells[midpoint] = leaf

            for cell in old_cells:
                if cell in self.rhs:
                    del self.rhs[cell]
                if cell in self.g:
                    del self.g[cell]
                if cell in self.OPEN:
                    del self.OPEN[cell]

            for cell, leaf in new_cells.items():
                self.rhs[cell] = float("inf")
                self.g[cell] = float("inf")
                self.leaf_nodes[cell] = leaf
                self.UpdateVertex(cell)

    def ComputeShortestPath(self):
        while self.OPEN.top_key() < self.CalculateKey(self.x0) or self.getrhs(
            self.x0
        ) != self.getg(self.x0):
            kold = self.OPEN.top_key()
            u = self.OPEN.get()
            self.V.add(u)
            self.CLOSED.add(u)
            if not self.done:
                if self.x0 == u:
                    self.x0 = u
                    break
            if kold < self.CalculateKey(u):
                self.OPEN.put(u, self.CalculateKey(u))
            if self.getg(u) > self.getrhs(u):
                self.g[u] = self.rhs[u]
            else:
                self.g[u] = np.inf
                self.UpdateVertex(u)
            for s in self.getchildren(u):
                self.UpdateVertex(s)

            self.ind += 1

    def path(self, s_start=None):
        """
        Extracts the path to the goal
        """
        path = []

        s_goal = self.xt

        if not s_start:
            s = self.x0
        else:
            s = s_start
        ind = 0
        while s != s_goal:
            children = list(self.CHILDREN[s])

            snext = children[
                np.argmin([self.getcost(s, s_p) + self.getg(s_p) for s_p in children])
            ]
            path.append([s, snext])
            s = snext

            if ind > 100:
                break
            ind += 1

        return path

    def move_dynamic_obs(self):
        for obj in self.dynamic_obs:
            old, new = self.env.move_block(
                a=obj.velocity, block_to_move=obj.index, mode="translation"
            )
            n_changed = self.updatecost(True, new, old)

    def run(self):
        # TODO make sure works
        self.agent_pos = self.x0
        self.ComputeShortestPath()
        self.Path = self.path(self.x0)

        self.V = set()

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        while self.agent_pos != self.xt:

            self.move_dynamic_obs()
            path = self.update()

            if path is None:
                return None

            self.agent_pos = self.move(path)
            self.traversed_path.append(self.agent_pos)

        return self.traversed_path


if __name__ == "__main__":
    ADStarlite = ADStarLite(1)
    ADStarlite.change_env("Evaluation/Maps/3D/block_map_25_3d/3_3d.json")

    t = time.time()
    ADStarlite.ComputeShortestPath()
    path = ADStarlite.path()
    print(path)
    print(time.time() - t)

    # print(path)
    ADStarlite.visualise(path)

    # print(a)
