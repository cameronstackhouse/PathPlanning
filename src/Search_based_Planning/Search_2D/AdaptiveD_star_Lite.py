import time
from D_star_Lite import DStar
from Quadtree import QuadTree

import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
        neighbours = set()

        current_leaf = self.leaf_nodes[s]

        # Current cell bounds
        current_left = current_leaf.x
        current_right = current_leaf.x + current_leaf.width
        current_top = current_leaf.y
        current_bottom = current_leaf.y + current_leaf.height

        for leaf in self.quadtree.leafs:
            if leaf.coords != current_leaf.coords:
                # Neighbor cell bounds
                neighbor_left = leaf.x
                neighbor_right = leaf.x + leaf.width
                neighbor_top = leaf.y
                neighbor_bottom = leaf.y + leaf.height

                # Check if the neighbor is to the North, South, East, West, or in the corners
                if (
                    neighbor_right == current_left or neighbor_left == current_right
                ) and (neighbor_bottom > current_top and neighbor_top < current_bottom):
                    # East or West neighbor
                    neighbor_center = (
                        neighbor_left + leaf.width // 2,
                        neighbor_top + leaf.height // 2,
                    )
                    neighbours.add(neighbor_center)

                elif (
                    neighbor_bottom == current_top or neighbor_top == current_bottom
                ) and (neighbor_right > current_left and neighbor_left < current_right):
                    # North or South neighbor
                    neighbor_center = (
                        neighbor_left + leaf.width // 2,
                        neighbor_top + leaf.height // 2,
                    )
                    neighbours.add(neighbor_center)

                elif (
                    neighbor_right == current_left or neighbor_left == current_right
                ) and (
                    neighbor_bottom == current_top or neighbor_top == current_bottom
                ):
                    # Corner neighbors (NE, NW, SE, SW)
                    neighbor_center = (
                        neighbor_left + leaf.width // 2,
                        neighbor_top + leaf.height // 2,
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
        dynamic_obj_in_sight = False

        affected_leafs = set()
        for dx in sight_range:
            for dy in sight_range:
                if dx == 0 and dy == 0:
                    continue

                check_pos = (round(current_pos[0]) + dx, round(current_pos[1] + dy))

                if check_pos in self.Env.dynamic_obs_cells:
                    dynamic_obj_in_sight = True
                    leaf_containing_point = self.quadtree.find_leaf_contaning_point(
                        check_pos
                    )
                    if leaf_containing_point:
                        affected_leafs.add(leaf_containing_point)

        if dynamic_obj_in_sight:
            print("REPARTITION")
            replan_time = time.time()

            self.s_start = self.agent_pos
            self.km += self.h(self.s_last, self.s_start)
            self.s_last = self.s_start

            print("UPDATING COSTS AND QUEUE")
            self.update_costs_and_queue(affected_leafs)
            print("UPDATED")

            new_path = self.plan_new_path()
            replan_time = time.time() - replan_time
            self.replan_time.append(replan_time)
            return new_path
        else:
            return path

    def update_costs_and_queue(self, affected_leafs):
        new_cells = {}

        # Repartition affected leaves
        for leaf in affected_leafs:
            leaf.left_top = None
            leaf.right_top = None
            leaf.left_bottom = None
            leaf.right_bottom = None
            self.quadtree.partition(leaf)

        if len(affected_leafs) > 0:
            self.quadtree.update_leafs()

            for leaf in self.quadtree.leafs:
                center = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
                parent_center = (
                    leaf.parent.x + leaf.parent.width // 2,
                    leaf.parent.y + leaf.parent.height // 2,
                )
                # TODO Look at
                new_cells[center] = leaf
                parent = leaf.parent

                if parent:
                    # Initilise rhs and g values based on parents values
                    self.rhs[center] = self.rhs.get(parent_center, float("inf"))
                    self.g[center] = self.g.get(parent_center, float("inf"))
                else:
                    self.rhs[center] = float("inf")
                    self.g[center] = float("inf")

                self.U[center] = self.CalculateKey(center)

                new_cells[center] = leaf

            # Remove the old leafs from U
            for leaf in affected_leafs:
                cell = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
                if cell in self.U:
                    del self.U[cell]
                if cell in self.rhs:
                    del self.rhs[cell]
                if cell in self.g:
                    del self.g[cell]

            for center in new_cells:
                self.UpdateVertex(center)

    def plan_new_path(self):
        self.s_start = self.agent_pos
        self.visited = set()
        path = self.ComputePath()
        self.current_index = 0
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
            self.agent_pos = self.s_start
            GOAL = self.s_goal

            while self.agent_pos != GOAL:
                print(self.agent_pos)

                self.update_object_positions()  # FROM D* LITE
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
    # Test block 12
    s = ADStarLite((0, 0), (1, 0), "euclidian", time=float("inf"))
    s.change_env("Evaluation/Maps/2D/main/block_12.json")
    s.dobs_dir = "Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json"
    path = s.run()

    # path = s.ComputePath()

    if path:
        s.plot_traversal()
        # s.quadtree.visualize(path)

    # s.quadtree.visualize(path)
