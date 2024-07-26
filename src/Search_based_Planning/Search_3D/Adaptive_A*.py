import json
import queue

from matplotlib import pyplot as plt
import numpy as np
from Astar3D import Weighted_A_star
from Search_3D.Octree import Octree
from Search_3D.env3D import CustomEnv
from Search_3D.utils3D import heuristic_fun, getDist, cost, isinobb, isinball, isinbound


class AdaptiveAStar(Weighted_A_star):
    def __init__(self):
        super().__init__(1)
        self.Alldirec = {
            (1, 0, 0): 1,
            (0, 1, 0): 1,
            (0, 0, 1): 1,
            (-1, 0, 0): 1,
            (0, -1, 0): 1,
            (0, 0, -1): 1,
            (1, 1, 0): np.sqrt(2),
            (1, 0, 1): np.sqrt(2),
            (0, 1, 1): np.sqrt(2),
            (-1, -1, 0): np.sqrt(2),
            (-1, 0, -1): np.sqrt(2),
            (0, -1, -1): np.sqrt(2),
            (1, -1, 0): np.sqrt(2),
            (-1, 1, 0): np.sqrt(2),
            (1, 0, -1): np.sqrt(2),
            (-1, 0, 1): np.sqrt(2),
            (0, 1, -1): np.sqrt(2),
            (0, -1, 1): np.sqrt(2),
        }

        self.octree = None
        self.leaf_nodes = {}

        self.speed = 6
        self.name = "Adaptive A*"
        self.dynamic_obs = []
        self.agent_positions = []
        self.agent_pos = None
        self.dobs_dir = None
        self.current_index = 0

    def visualise(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the environment blocks
        for block in self.env.blocks:
            x, y, z = block[0], block[1], block[2]
            dx, dy, dz = block[3] - block[0], block[4] - block[1], block[5] - block[2]
            ax.bar3d(x, y, z, dx, dy, dz, color="b", alpha=0.5)

        # Plot the dynamic obstacles
        for obj in self.dynamic_obs:
            x, y, z = obj.current_pos
            dx, dy, dz = obj.size
            ax.bar3d(x, y, z, dx, dy, dz, color="r", alpha=0.5)

        # Plot the path
        path_points = np.array([point[0] for point in path])
        # print(f"path: {path}")
        path_points = np.vstack([path_points, self.env.goal])

        # print(f"path points: {path_points}")
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            color="g",
            marker="o",
        )

        ax.scatter(
            self.env.start[0],
            self.env.start[1],
            self.env.start[2],
            color="g",
            s=100,
            label="Start",
        )
        ax.scatter(
            self.env.goal[0],
            self.env.goal[1],
            self.env.goal[2],
            color="r",
            s=100,
            label="Goal",
        )

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Adatptive A* Path visualisation")
        ax.legend()

        plt.show()

    def change_env(self, map_name, obs_name=None):
        self.dobs_dir = obs_name
        data = None
        with open(map_name) as f:
            data = json.load(f)

        if data:
            self.current_index = 0
            self.agent_pos = None
            self.dynamic_obs = []

            self.env = CustomEnv(data)
            self.octree = Octree(self.env)

            self.leaf_nodes = {}
            for leaf in self.octree.leafs:
                center = (
                    leaf.x + leaf.width // 2,
                    leaf.y + leaf.height // 2,
                    leaf.z + leaf.depth // 2,
                )
                self.leaf_nodes[center] = leaf

            self.start, self.goal = tuple(self.env.start), tuple(self.env.goal)

            self.g = {self.start: 0, self.goal: np.inf}
            self.Parent = {}
            self.CLOSED = set()
            self.V = []
            self.done = False
            self.Path = []
            self.ind = 0
            self.x0, self.xt = self.start, self.goal
            self.OPEN = queue.MinheapPQ()
            self.OPEN.put(self.x0, self.g[self.x0] + heuristic_fun(self, self.x0))
            self.lastpoint = self.x0

    def children_non_uniform(self, x, settings=0):
        allchild = []
        allcost = []

        current_leaf = self.leaf_nodes[x]
        current_center = (
            current_leaf.x + current_leaf.width // 2,
            current_leaf.y + current_leaf.height // 2,
            current_leaf.z + current_leaf.depth // 2,
        )

        for leaf in self.octree.leafs:
            if leaf.coords != current_leaf.coords:
                if (
                    (
                        current_leaf.x + current_leaf.width == leaf.x
                        or current_leaf.x == leaf.x + leaf.width
                    )
                    and (
                        current_leaf.y < leaf.y + leaf.height
                        and current_leaf.y + current_leaf.height > leaf.y
                    )
                    and (
                        current_leaf.z < leaf.z + leaf.depth
                        and current_leaf.z + current_leaf.depth > leaf.z
                    )
                ):
                    neighbor_center = (
                        leaf.x + leaf.width // 2,
                        leaf.y + leaf.height // 2,
                        leaf.z + leaf.depth // 2,
                    )

                    in_obj = (
                        any([isinobb(i, neighbor_center) for i in self.env.OBB])
                        or any([isinball(i, neighbor_center) for i in self.env.balls])
                        or any([isinbound(i, neighbor_center) for i in self.env.blocks])
                    )

                    if not in_obj and isinbound(self.env.boundary, neighbor_center):
                        allchild.append(neighbor_center)
                        distance = np.linalg.norm(
                            np.array(current_center) - np.array(neighbor_center)
                        )
                        allcost.append((neighbor_center, distance))

                # Check for y-aligned neighbors
                if (
                    (
                        current_leaf.y + current_leaf.height == leaf.y
                        or current_leaf.y == leaf.y + leaf.height
                    )
                    and (
                        current_leaf.x < leaf.x + leaf.width
                        and current_leaf.x + current_leaf.width > leaf.x
                    )
                    and (
                        current_leaf.z < leaf.z + leaf.depth
                        and current_leaf.z + current_leaf.depth > leaf.z
                    )
                ):
                    neighbor_center = (
                        leaf.x + leaf.width // 2,
                        leaf.y + leaf.height // 2,
                        leaf.z + leaf.depth // 2,
                    )

                    in_obj = (
                        any([isinobb(i, neighbor_center) for i in self.env.OBB])
                        or any([isinball(i, neighbor_center) for i in self.env.balls])
                        or any([isinbound(i, neighbor_center) for i in self.env.blocks])
                    )

                    if not in_obj and isinbound(self.env.boundary, neighbor_center):
                        allchild.append(neighbor_center)
                        distance = np.linalg.norm(
                            np.array(current_center) - np.array(neighbor_center)
                        )
                        allcost.append((neighbor_center, distance))

                # Check for z-aligned neighbors
                if (
                    (
                        current_leaf.z + current_leaf.depth == leaf.z
                        or current_leaf.z == leaf.z + leaf.depth
                    )
                    and (
                        current_leaf.x < leaf.x + leaf.width
                        and current_leaf.x + current_leaf.width > leaf.x
                    )
                    and (
                        current_leaf.y < leaf.y + leaf.height
                        and current_leaf.y + current_leaf.height > leaf.y
                    )
                ):
                    neighbor_center = (
                        leaf.x + leaf.width // 2,
                        leaf.y + leaf.height // 2,
                        leaf.z + leaf.depth // 2,
                    )

                    in_obj = (
                        any([isinobb(i, neighbor_center) for i in self.env.OBB])
                        or any([isinball(i, neighbor_center) for i in self.env.balls])
                        or any([isinbound(i, neighbor_center) for i in self.env.blocks])
                    )

                    if not in_obj and isinbound(self.env.boundary, neighbor_center):
                        allchild.append(neighbor_center)
                        distance = np.linalg.norm(
                            np.array(current_center) - np.array(neighbor_center)
                        )
                        allcost.append((neighbor_center, distance))
        if settings == 0:
            return allchild
        if settings == 1:
            return allcost

    def compute_path(self):
        xt = self.xt
        xi = self.x0
        while self.OPEN:
            xi = self.OPEN.get()
            if xi not in self.CLOSED:
                self.V.append(np.array(xi))
            self.CLOSED.add(xi)
            if getDist(xi, xt) <= 1:
                break
            for xj in self.children_non_uniform(xi):
                if xj not in self.g:
                    self.g[xj] = np.inf
                else:
                    pass
                a = self.g[xi] + cost(self, xi, xj)
                if a < self.g[xj]:
                    self.g[xj] = a
                    self.Parent[xj] = xi
                    self.OPEN.put(xj, a + heuristic_fun(self, xj))

        self.lastpoint = xi
        if self.lastpoint in self.CLOSED:
            self.done = True
            self.Path = self.path()
            return self.Path
        else:
            return None


if __name__ == "__main__":
    astar = AdaptiveAStar()
    # Check with this, going through objects
    astar.change_env("Evaluation/Maps/3D/block_map_25_3d/10_3d.json")

    path = astar.compute_path()

    print(path)

    if path:
        astar.visualise(path[::-1])
