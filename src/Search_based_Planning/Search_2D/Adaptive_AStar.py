import time
import numpy as np
from Search_2D.Astar import AStar
from Search_2D.Quadtree import QuadTree
from Search_2D.plotting import DynamicPlotting


class AdaptiveAStar(AStar):
    def __init__(self, s_start, s_goal, heuristic_type, time=float("inf")):
        super().__init__(s_start, s_goal, heuristic_type, time)
        if time != float("inf"):
            self.name = f"Adaptive A*: {time}"
        else:
            self.name = "Adaptive A*"
        self.quadtree = None
        self.speed = 6
        self.current_index = 0
        self.time_steps = 0
        self.dynamic_objects = []
        self.initial_path = None
        self.compute_time = None
        self.total_time = None
        self.replan_time = []

    def euclidean_distance(self, point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)

        distance = np.linalg.norm(point1 - point2)

        return distance

    def plot_traversal(self):
        plotter = DynamicPlotting(
            self.s_start,
            self.s_goal,
            self.dynamic_objects,
            self.time_steps,
            self.agent_positions,
            self.initial_path,
        )

        plotter.env = self.Env
        plotter.obs = self.Env.obs
        plotter.xI = self.initial_path[0]
        plotter.xG = self.s_goal

        plotter.animation(self.agent_positions, "D* Lite Original Path vs Path Taken")

    def change_env(self, map_name, dobs=None):
        new_env = super().change_env(map_name)

        self.current_index = 0
        self.initial_path = None
        self.compute_time = None
        self.total_time = None
        self.replan_time = []
        self.agent_positions = []

        self.quadtree = QuadTree(new_env)

        self.leaf_nodes = {}
        for leaf in self.quadtree.leafs:
            center = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
            self.leaf_nodes[center] = leaf

        self.dobs_dir = dobs

    def update_object_positions(self):
        self.Env.dynamic_obs_cells = set()
        for i, object in enumerate(self.dynamic_objects):
            prev_pos = object.current_pos
            new_pos = object.update_pos()

            # new_pos = prev_pos

            object.old_pos = object.current_pos
            object.current_pos = new_pos

            self.Env.update_dynamic_obj_pos(i, new_pos[0], new_pos[1])

    def move(self, path, mps=6):
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.s_goal

        current = self.agent_pos
        next = path[self.current_index + 1]

        seg_distance = self.euclidean_distance(current, next)

        direction = (
            (next[0] - current[0]) / seg_distance,
            (next[1] - current[1]) / seg_distance,
        )

        new_pos = (current[0] + direction[0] * mps, current[1] + direction[1] * mps)

        if self.euclidean_distance(current, new_pos) >= seg_distance:
            # Calculate vectors and check if in the same direction
            v1 = np.array(next) - np.array(current)
            v2 = np.array(new_pos) - np.array(next)
            dot_product = np.dot(v1, v2)

            mag_v1 = np.linalg.norm(v1)
            mag_v2 = np.linalg.norm(v2)

            same_dir = np.isclose(dot_product, mag_v1 * mag_v2)

            if same_dir:
                # Move the agent far forward without turning
                count = 0
                while self.current_index < len(path) - 1:
                    next = path[self.current_index + 1]
                    seg_distance = self.euclidean_distance(current, next)
                    direction = (
                        (next[0] - current[0]) / seg_distance,
                        (next[1] - current[1]) / seg_distance,
                    )
                    new_pos = (
                        current[0] + direction[0] * mps,
                        current[1] + direction[1] * mps,
                    )
                    v1 = np.array(next) - np.array(current)
                    v2 = np.array(new_pos) - np.array(next)
                    dot_product = np.dot(v1, v2)
                    mag_v1 = np.linalg.norm(v1)
                    mag_v2 = np.linalg.norm(v2)
                    same_dir = np.isclose(dot_product, mag_v1 * mag_v2)
                    if not same_dir or count >= self.speed - 1:
                        break
                    current = next
                    self.agent_pos = current
                    self.current_index += 1
                    count += 1
            else:
                # Move to the next node and update position
                self.agent_pos = next
                self.current_index += 1
        else:
            self.agent_pos = new_pos

        return self.agent_pos

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
                    if not self.Env.in_dynamic_object(
                        neighbor_center[0], neighbor_center[1], flag=True
                    ):
                        neighbours.add(neighbor_center)

                elif (
                    neighbor_bottom == current_top or neighbor_top == current_bottom
                ) and (neighbor_right > current_left and neighbor_left < current_right):
                    # North or South neighbor
                    neighbor_center = (
                        neighbor_left + leaf.width // 2,
                        neighbor_top + leaf.height // 2,
                    )
                    if not self.Env.in_dynamic_object(
                        neighbor_center[0], neighbor_center[1], True
                    ):
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

                    if not self.Env.in_dynamic_object(
                        neighbor_center[0], neighbor_center[1], True
                    ):
                        neighbours.add(neighbor_center)

        return neighbours

    def replan(self, path):
        # 1. Check for objects
        current_pos = self.agent_pos
        SIGHT = 3
        sight_range = range(-SIGHT, SIGHT + 1)
        dynamic_obj_in_sight = False

        affected_leafs = set()
        for dx in sight_range:
            for dy in sight_range:
                if dx == 0 and dy == 0:
                    continue

                check_pos = (round(current_pos[0] + dx), round(current_pos[1] + dy))
                if self.Env.in_dynamic_object(check_pos[0], check_pos[1]):
                    dynamic_obj_in_sight = True
                    leaf_containing_point = self.quadtree.find_leaf_contaning_point(
                        check_pos
                    )
                    if leaf_containing_point:
                        affected_leafs.add(leaf_containing_point)

        # 2. If changed, repartition affected nodes and replan
        if dynamic_obj_in_sight:
            replan_time = time.time()
            # Repartition
            for leaf in affected_leafs:
                leaf.clear()
                self.quadtree.partition(leaf)

            self.quadtree.update_leafs()

            for leaf in self.quadtree.leafs:
                center = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
                self.leaf_nodes[center] = leaf

            self.leaf_nodes[self.agent_pos] = self.quadtree.find_leaf_contaning_point(
                self.agent_pos
            )

            # Replan
            self.s_start = self.agent_pos
            self.PARENT = dict()
            self.OPEN = []
            self.CLOSED = []
            self.g = dict()
            path = self.planning()

            if path:
                path = path[::-1]
            else:
                replan_time = time.time() - replan_time
                self.replan_time.append(replan_time)
                return None

            replan_time = time.time() - replan_time
            self.replan_time.append(replan_time)
            self.current_index = 0

        return path

    def run(self):
        initial_compute_start = time.time()

        path = self.planning()

        initial_compute_time = time.time() - initial_compute_start

        self.compute_time = initial_compute_time

        if self.dobs_dir:
            self.set_dynamic_obs(self.dobs_dir)

        if path:
            path = path[::-1]
            self.agent_pos = path[0]
            self.agent_positions.append(self.agent_pos)

            self.initial_path = path

            traversal_time = time.time()
            while self.agent_pos != self.s_goal:
                self.update_object_positions()
                path = self.replan(path)

                if path is None:
                    return None

                self.agent_pos = self.move(path)
                self.agent_positions.append(self.agent_pos)
                self.time_steps += 1

            traversal_time = time.time - traversal_time()
            self.total_time = traversal_time
            return self.agent_positions
        else:
            return None


if __name__ == "__main__":
    s = AdaptiveAStar((0, 0), (0, 0), "euclidian", time=10)
    # 23 best example to show
    s.change_env(
        "Evaluation/Maps/2D/main/block_21.json",
        "Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json",
    )

    path = s.run()

    if path:
        s.plot_traversal()
