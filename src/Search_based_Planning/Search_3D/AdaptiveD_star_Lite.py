import queue
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from DstarLite3D import D_star_Lite

from Search_3D.utils3D import children_non_uniform, cost, getDist, isinbound


class TreeNode:
    def __init__(self, x, y, z, width, height, depth, env) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.one = None
        self.two = None
        self.three = None
        self.four = None
        self.five = None
        self.six = None
        self.seven = None
        self.eight = None
        self.parent = None
        self.env = env
        self.coords = [x, y, z]

    def is_leaf(self):
        return (
            self.one is None
            and self.two is None
            and self.three is None
            and self.four is None
            and self.five is None
            and self.six is None
            and self.seven is None
            and self.eight is None
        )

    def is_uniform(self):
        def is_within_block(block, x, y, z):
            return (
                block[0] <= x <= block[3]
                and block[1] <= y <= block[4]
                and block[2] <= z <= block[5]
            )

        init_val = any(
            is_within_block(block, self.x, self.y, self.z) for block in self.env.blocks
        )

        for i in range(self.x, self.x + self.width):
            for j in range(self.y, self.y + self.height):
                for k in range(self.z, self.z + self.depth):
                    current_val = any(
                        is_within_block(block, i, j, k) for block in self.env.blocks
                    )
                    if current_val != init_val:
                        return False

        return True

    def contains_point(self, point):
        return (
            self.x <= point[0] < self.x + self.width
            and self.y <= point[1] < self.y + self.height
            and self.z <= point[2] < self.z + self.depth
        )

    def partition(self, leafs):
        if self.width == 1 and self.height == 1 and self.depth == 1:
            return

        if self.is_leaf() and (
            not self.is_uniform()
            or self.contains_point(self.env.start)
            or self.contains_point(self.env.goal)
        ):
            mid_width = (self.width + 1) // 2 if self.width > 1 else 1
            mid_height = (self.height + 1) // 2 if self.height > 1 else 1
            mid_depth = (self.depth + 1) // 2 if self.width > 1 else 1

            self.one = TreeNode(
                self.x, self.y, self.z, mid_width, mid_height, mid_depth, self.env
            )
            self.two = TreeNode(
                self.x + mid_width,
                self.y,
                self.z,
                self.width - mid_width,
                mid_height,
                mid_depth,
                self.env,
            )
            self.three = TreeNode(
                self.x,
                self.y + mid_height,
                self.z,
                mid_width,
                self.height - mid_height,
                mid_depth,
                self.env,
            )
            self.four = TreeNode(
                self.x + mid_width,
                self.y + mid_height,
                self.z,
                self.width - mid_width,
                self.height - mid_height,
                mid_depth,
                self.env,
            )
            self.five = TreeNode(
                self.x,
                self.y,
                self.z + mid_depth,
                mid_width,
                mid_height,
                self.depth - mid_depth,
                self.env,
            )
            self.six = TreeNode(
                self.x + mid_width,
                self.y,
                self.z + mid_depth,
                self.width - mid_width,
                mid_height,
                self.depth - mid_depth,
                self.env,
            )
            self.seven = TreeNode(
                self.x,
                self.y + mid_height,
                self.z + mid_depth,
                mid_width,
                self.height - mid_height,
                self.depth - mid_depth,
                self.env,
            )
            self.eight = TreeNode(
                self.x + mid_width,
                self.y + mid_height,
                self.z + mid_depth,
                self.width - mid_width,
                self.height - mid_height,
                self.depth - mid_depth,
                self.env,
            )

            self.one.parent = self
            self.two.parent = self
            self.three.parent = self
            self.four.parent = self
            self.five.parent = self
            self.six.parent = self
            self.seven.parent = self
            self.eight.parent = self

            leafs.remove(self)
            leafs.extend(
                [
                    self.one,
                    self.two,
                    self.three,
                    self.four,
                    self.five,
                    self.six,
                    self.seven,
                    self.eight,
                ]
            )

            self.one.partition(leafs)
            self.two.partition(leafs)
            self.three.partition(leafs)
            self.four.partition(leafs)
            self.five.partition(leafs)
            self.six.partition(leafs)
            self.seven.partition(leafs)
            self.eight.partition(leafs)

    def first_inconsistent(self):
        current = self
        while current is not None:
            if not current.is_uniform():
                return current

            current = current.parent

        return None


class Octree:
    def __init__(self, env) -> None:
        self.env = env
        self.root = TreeNode(0, 0, 0, env.x_range, env.y_range, env.z_range, env)
        self.leafs = [self.root]
        self.partition(self.root)

    def partition(self, node):
        if node.is_leaf() and not node.is_uniform():
            node.partition(self.leafs)

    def update_leafs(self):
        self.leafs = []
        self._collect_leafs(self.root)

    def _collect_leafs(self, node):
        if node.is_leaf():
            self.leafs.append(node)
        else:
            for child in {
                node.one,
                node.two,
                node.three,
                node.four,
                node.five,
                node.six,
                node.seven,
                node.eight,
            }:
                if child:
                    self._collect_leafs(child)

    def visualize(self, path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([0, self.root.width])
        ax.set_ylim([0, self.root.height])
        ax.set_zlim([0, self.root.depth])

        for leaf in self.leafs:
            self._draw_cube(ax, leaf)

        plt.show()

    def _draw_cube(self, ax, node):
        x, y, z = node.x, node.y, node.z
        dx, dy, dz = node.width, node.height, node.depth
        vertices = [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                faces, facecolors="white", linewidths=1, edgecolors="r", alpha=0.25
            )
        )


class ADStarLite(D_star_Lite):
    def __init__(self, resolution=1):
        self.name = "AD* Lite"
        new_env = super().__init__(resolution)
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

        self.rhs[self.xt] = 0.0
        self.OPEN = queue.MinheapPQ()
        self.OPEN.put(self.xt, self.CalculateKey(self.xt))

    # TODO look at methods to override to make function

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

    # TODO maybe update vertex
    # TODO maybe compute shortest path

    def path(self, s_start=None):
        """After ComputeShortestPath() returns, one can then follow a shortest path from x_init to
        x_goal by always moving from the current vertex s, starting at x_init,
        to any successor s' that minimizes cBest(s,s') + g(s') until x_goal is reached (ties can be broken arbitrarily).
        """
        path = []
        s_goal = self.xt
        if not s_start:
            s = self.x0
        else:
            s = s_start
        ind = 0
        while s != s_goal:
            if s == self.x0:
                children = [
                    i
                    for i in self.CLOSED
                    if getDist(s, i)
                    <= max(
                        self.leaf_nodes[s].width,
                        self.leaf_nodes[s].height,
                        self.leaf_nodes[s].depth,
                    )
                    * np.sqrt(3)
                ]
            else:
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

    def path(self, s_start=None):
        pass

    def run(self):
        self.agent_pos = self.x0
        self.ComputeShortestPath()
        self.Path = self.path(self.x0)

        self.V = set()
        while self.agent_pos != self.xt:
            pass


if __name__ == "__main__":
    plan = ADStarLite(1)
    plan.change_env("Evaluation/Maps/3D/block_map_25_3d/17_3d.json")
    tree = Octree(plan.env)

    print(len(tree.leafs))
    tree.visualize()
