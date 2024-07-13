from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from DstarLite3D import D_star_Lite


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
        super().__init__(resolution)


if __name__ == "__main__":
    plan = ADStarLite(1)
    plan.change_env("Evaluation/Maps/3D/block_map_25_3d/2_3d.json")
    tree = Octree(plan.env)

    tree.visualize()
