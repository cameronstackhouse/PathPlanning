from matplotlib import patches, pyplot as plt


class TreeNode:
    def __init__(self, x, y, width, height, env) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.left_top = None
        self.right_top = None
        self.left_bottom = None
        self.right_bottom = None
        self.parent = None
        self.env = env
        self.coords = [x, y]

    def clear(self):
        self.left_top = None
        self.right_top = None
        self.left_bottom = None
        self.right_bottom = None

    def is_leaf(self):
        return (
            self.left_top is None
            and self.right_top is None
            and self.left_bottom is None
            and self.right_bottom is None
        )

    def is_uniform(self):
        init_val = (self.x, self.y) in self.env.obs or (
            self.x,
            self.y,
        ) in self.env.dynamic_obs_cells
        for i in range(self.x, self.x + self.width):
            for j in range(self.y, self.y + self.height):
                if (
                    (i, j) in self.env.obs or (i, j) in self.env.dynamic_obs_cells
                ) != init_val:
                    return False
        return True

    def has_object_and_uniform(self):
        init_val = (self.x, self.y) in self.env.obs or (
            self.x,
            self.y,
        ) in self.env.dynamic_obs_cells

        if not init_val:
            return False

        consistent = True
        for i in range(self.x, self.x + self.width):
            for j in range(self.y, self.y + self.height):
                if (
                    (i, j) in self.env.obs or (i, j) in self.env.dynamic_obs_cells
                ) != init_val:
                    consistent = False
                    break

            if not consistent:
                break

        return consistent and self.is_uniform()

    def contains_point(self, point):
        return (
            self.x <= point[0] < self.x + self.width
            and self.y <= point[1] < self.y + self.height
        )

    def partition(self, leafs, created_nodes):
        if self.width == 1 and self.height == 1:
            return

        if self.is_leaf() and (
            not self.is_uniform()
            or self.contains_point(self.env.s_start)
            or self.contains_point(self.env.s_goal)
        ):
            mid_width = (self.width + 1) // 2 if self.width > 1 else 1
            mid_height = (self.height + 1) // 2 if self.height > 1 else 1

            self.left_top = TreeNode(self.x, self.y, mid_width, mid_height, self.env)
            self.right_top = TreeNode(
                self.x + mid_width, self.y, self.width - mid_width, mid_height, self.env
            )
            self.left_bottom = TreeNode(
                self.x,
                self.y + mid_height,
                mid_width,
                self.height - mid_height,
                self.env,
            )
            self.right_bottom = TreeNode(
                self.x + mid_width,
                self.y + mid_height,
                self.width - mid_width,
                self.height - mid_height,
                self.env,
            )

            for child in [
                self.left_top,
                self.right_top,
                self.left_bottom,
                self.right_bottom,
            ]:
                child.parent = self

            leafs.remove(self)
            leafs.extend(
                [self.left_top, self.right_top, self.left_bottom, self.right_bottom]
            )

            if self in created_nodes:
                created_nodes.remove(self)
            created_nodes.extend(
                [self.left_top, self.right_top, self.left_bottom, self.right_bottom]
            )

            self.left_top.partition(leafs, created_nodes)
            self.right_top.partition(leafs, created_nodes)
            self.left_bottom.partition(leafs, created_nodes)
            self.right_bottom.partition(leafs, created_nodes)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TreeNode):
            return False

        return (
            self.x == value.x
            and self.y == value.y
            and self.width == value.width
            and self.height == value.height
        )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.width, self.height))


class QuadTree:
    def __init__(self, env) -> None:
        self.env = env
        self.root = TreeNode(0, 0, env.x_range, env.y_range, env)
        self.leafs = [self.root]
        self.created_nodes = []
        self.partition(self.root)

    def partition(self, node):
        self.created_nodes = []
        if node.is_leaf() and not node.is_uniform():
            node.partition(self.leafs, self.created_nodes)

    def update_leafs(self):
        self.leafs = []
        self._collect_leafs(self.root)

    def _collect_leafs(self, node):
        if node.is_leaf():
            self.leafs.append(node)
        else:
            for child in [
                node.left_top,
                node.right_top,
                node.left_bottom,
                node.right_bottom,
            ]:
                if child:
                    self._collect_leafs(child)

    def visualize(self, path=None):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.root.width)
        ax.set_ylim(0, self.root.height)
        for node in self.leafs:
            if node.contains_point(self.env.s_start):
                color = "blue"
            elif node.contains_point(self.env.s_goal):
                color = "green"
            elif node.is_uniform():
                color = "black" if (node.x, node.y) in node.env.obs else "white"
            rect = patches.Rectangle(
                (node.x, node.y),
                node.width,
                node.height,
                linewidth=1,
                edgecolor="r",
                facecolor=color,
                fill=True,
            )
            ax.add_patch(rect)

        for cell in self.env.dynamic_obs_cells:
            rect = patches.Rectangle(
                (cell[0], cell[1]),
                1,
                1,
                linewidth=1,
                edgecolor="r",
                facecolor="grey",
                fill=True,
            )
            ax.add_patch(rect)

        if path:
            px = [x[0] for x in path]
            py = [x[1] for x in path]
            ax.plot(px, py, color="green", linewidth=2)

        plt.gca()
        plt.show()

    def find_leaf_contaning_point(self, point):
        current_node = self.root

        while not current_node.is_leaf():
            if current_node.left_top and current_node.left_top.contains_point(point):
                current_node = current_node.left_top
            elif current_node.right_top and current_node.right_top.contains_point(
                point
            ):
                current_node = current_node.right_top
            elif current_node.left_bottom and current_node.left_bottom.contains_point(
                point
            ):
                current_node = current_node.left_bottom
            elif current_node.right_bottom and current_node.right_bottom.contains_point(
                point
            ):
                current_node = current_node.right_bottom
            else:
                return None

        return current_node
