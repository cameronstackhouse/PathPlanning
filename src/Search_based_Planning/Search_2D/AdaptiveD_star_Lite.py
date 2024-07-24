import time
from D_star_Lite import DStar

import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

    def is_leaf(self):
        return (
            self.left_top is None
            and self.right_top is None
            and self.left_bottom is None
            and self.right_bottom is None
        )

    def is_uniform(self):
        init_val = (self.x, self.y) in self.env.obs
        for i in range(self.x, self.x + self.width):
            for j in range(self.y, self.y + self.height):
                if (
                    (i, j) in self.env.obs or (i, j) in self.env.dynamic_obs_cells
                ) != init_val:
                    return False
        return True

    def contains_point(self, point):
        return (
            self.x <= point[0] < self.x + self.width
            and self.y <= point[1] < self.y + self.height
        )

    def partition(self, leafs):
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

            self.left_top.partition(leafs)
            self.right_top.partition(leafs)
            self.left_bottom.partition(leafs)
            self.right_bottom.partition(leafs)

    def first_inconsistent(self):
        current = self
        while current is not None:
            if not current.is_uniform():
                return current

            current = current.parent

        return None


class QuadTree:
    def __init__(self, env) -> None:
        self.env = env
        self.root = TreeNode(0, 0, env.x_range, env.y_range, env)
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

        plt.gca().invert_yaxis()
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


class ADStarLite(DStar):
    def __init__(self, s_start, s_goal, heuristic_type, time=...):
        super().__init__(s_start, s_goal, heuristic_type, time)
        self.heuristic_type = heuristic_type
        self.quadtree = None
        self.speed = 6

    def change_env(self, map_name, obj_dir=None):
        new_env = super().change_env(map_name, obj_dir)
        self.quadtree = QuadTree(new_env)

        self.rhs = {}
        self.g = {}
        self.leaf_nodes = {}
        for leaf in self.quadtree.leafs:
            center = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
            self.rhs[center] = float("inf")
            self.g[center] = float("inf")
            self.leaf_nodes[center] = leaf

        self.rhs[self.s_goal] = 0.0
        self.count = 0
        self.U = {}
        self.U[self.s_goal] = self.CalculateKey(self.s_goal)

    def get_neighbor(self, s):
        # TODO LOOK AT!
        neighbours = set()
        current_leaf = self.leaf_nodes[s]

        for leaf in self.quadtree.leafs:
            if leaf.coords != current_leaf.coords:
                if (
                    current_leaf.x + current_leaf.width == leaf.x
                    or current_leaf.x == leaf.x + leaf.width
                ) and (
                    current_leaf.y < leaf.y + leaf.height
                    and current_leaf.y + current_leaf.height > leaf.y
                ):
                    neighbor_center = (
                        leaf.x + leaf.width // 2,
                        leaf.y + leaf.height // 2,
                    )
                    neighbours.add(neighbor_center)

                if (
                    current_leaf.y + current_leaf.height == leaf.y
                    or current_leaf.y == leaf.y + leaf.height
                ) and (
                    current_leaf.x < leaf.x + leaf.width
                    and current_leaf.x + current_leaf.width > leaf.x
                ):
                    neighbor_center = (
                        leaf.x + leaf.width // 2,
                        leaf.y + leaf.height // 2,
                    )
                    neighbours.add(neighbor_center)

        return neighbours

    def find_leaf_with_point(self, point):
        def traverse(node):
            if node.is_leaf():
                return node
            for child in [
                node.left_top,
                node.right_top,
                node.left_bottom,
                node.right_bottom,
            ]:
                if child and child.contains_point(point):
                    return traverse(child)
            return None

        return traverse(self.quadtree.root)

    def update_costs(self, path):
        current_pos = self.agent_pos
        SIGHT = 3

        sight_range = range(-SIGHT, SIGHT + 1)

        new_cells = set()
        old_cells = set()

        repartition = False
        affected_leafs = set()
        for dx in sight_range:
            for dy in sight_range:
                check_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if check_pos in self.Env.dynamic_obs_cells:
                    repartition = True
                    leaf_containing_point = self.quadtree.find_leaf_contaning_point(
                        check_pos
                    )
                    if leaf_containing_point:
                        affected_leafs.add(leaf_containing_point)

        if repartition:
            replan_time = time.time()

            self.update_costs_and_queue(affected_leafs)

            self.s_start = self.agent_pos
            path = [self.s_start]

            self.km += self.h(self.s_last, self.s_start)

            self.s_last = self.s_start

            for obj in self.dynamic_objects:
                old_pos = obj.old_pos
                new_pos = obj.current_pos
                width, height = obj.size

                old_cells.update(self.get_affected_cells(old_pos, width, height))
                new_cells.update(self.get_affected_cells(new_pos, width, height))

            # Find the leafs the new cells belong to, they will be entirely objects
            # and update
            all_cells = new_cells.union(old_cells)

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
                    node_center = (x + leaf.width // 2, y + leaf.height // 2)
                    old_cells.add(node_center)

        for leaf in affected_leafs:
            self.quadtree.partition(leaf)

        if len(affected_leafs) > 0:
            self.quadtree.update_leafs()

            for leaf in self.quadtree.leafs:
                for x in range(leaf.x, leaf.x + leaf.width):
                    for y in range(leaf.y, leaf.y + leaf.height):
                        midpoint = (x + leaf.width // 2, y + leaf.height // 2)
                        new_cells[midpoint] = leaf

            for cell in old_cells:
                if cell in self.U:
                    del self.U[cell]
                if cell in self.rhs:
                    del self.rhs[cell]
                if cell in self.g:
                    del self.g[cell]

            for cell, leaf in new_cells.items():
                self.rhs[cell] = float("inf")
                self.g[cell] = float("inf")
                self.leaf_nodes[cell] = leaf
                print(f"cell: {cell}")
                self.UpdateVertex(cell)

    def plan_new_path(self):
        self.s_start = self.agent_pos
        path = self.ComputePath()
        return path

    def run(self):
        self.agent_positions.append(self.agent_pos)
        start_time = time.time()
        path = self.ComputePath()
        end_time = time.time() - start_time

        self.compute_time = end_time
        self.initial_path = path

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        start_time = time.time()

        if path:
            self.agent_pos = path[0]
            GOAL = path[-1]

            while self.agent_pos != GOAL:
                print(self.agent_pos)

                self.update_object_positions()
                path = self.update_costs(path)

                if path is None:
                    return None

                self.agent_pos = self.move(path)
                self.agent_positions.append(self.agent_pos)
                self.time_steps += 1

        end_time = time.time() - start_time
        self.total_time = end_time
        return self.agent_positions


if __name__ == "__main__":
    s = ADStarLite((0, 0), (1, 0), "euclidian", time=float("inf"))
    s.change_env("Evaluation/Maps/2D/main/block_15.json")
    s.dobs_dir = "Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json"
    #path = s.run()
    
    path = s.ComputePath()

    if path:
        # s.plot_traversal()
        s.quadtree.visualize(path)

    # s.quadtree.visualize(path)
