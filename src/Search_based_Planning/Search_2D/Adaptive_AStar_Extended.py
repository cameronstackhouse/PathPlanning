from Adaptive_AStar import AdaptiveAStar
from Quadtree import QuadTree, TreeNode


class ExtendedQuadtree(QuadTree):
    def __init__(self, env) -> None:
        super().__init__(env)
    
    def visualize(self, path=None):
        # TODO
        raise NotImplementedError

    def merge_leafs(self):
        self.update_leafs()

        nodes_to_remove = set()
        nodes_to_add = set()

        for node in self.leafs:
            if node.is_leaf() and node.is_uniform():
                adjacent_nodes = self._get_adjacent_leafs(node)
                for adj_node in adjacent_nodes:
                    if (
                        adj_node.is_leaf()
                        and adj_node.is_uniform()
                        and adj_node not in nodes_to_remove
                        and node not in nodes_to_remove
                        and adj_node not in nodes_to_add
                        and node not in nodes_to_add
                        and self._can_merge(node, adj_node)
                    ):
                        merged = self._merge_nodes(node, adj_node)
                        if self._is_rectangular(merged):
                            nodes_to_remove.add(node)
                            nodes_to_remove.add(adj_node)
                            nodes_to_add.add(merged)
                        break

        self.leafs = [node for node in self.leafs if node not in nodes_to_remove]
        print(len(nodes_to_add))
        self.leafs.extend(nodes_to_add)

    def _get_adjacent_leafs(self, node):
        adjacent = []

        current_leaf = node

        current_left = current_leaf.x
        current_right = current_leaf.x + current_leaf.width
        current_top = current_leaf.y
        current_bottom = current_leaf.y + current_leaf.height

        for leaf in self.leafs:
            if leaf != current_leaf:
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
                    if not self.env.in_dynamic_object(
                        neighbor_center[0], neighbor_center[1], flag=True
                    ):
                        adjacent.append(leaf)

                elif (
                    neighbor_bottom == current_top or neighbor_top == current_bottom
                ) and (neighbor_right > current_left and neighbor_left < current_right):
                    # North or South neighbor
                    neighbor_center = (
                        neighbor_left + leaf.width // 2,
                        neighbor_top + leaf.height // 2,
                    )
                    if not self.env.in_dynamic_object(
                        neighbor_center[0], neighbor_center[1], True
                    ):
                        adjacent.append(leaf)

        return adjacent

    def _are_adjacent(self, node1, node2):
        # Check if nodes are horizontally adjacent
        horizontal_adj = (
            node1.y <= node2.y + node2.height and node2.y <= node1.y + node1.height
        ) and (node1.x + node1.width == node2.x or node2.x + node2.width == node1.x)

        # Check if nodes are vertically adjacent
        vertical_adj = (
            node1.x <= node2.x + node2.width and node2.x <= node1.x + node1.width
        ) and (node1.y + node1.height == node2.y or node2.y + node2.height == node1.y)

        return horizontal_adj or vertical_adj

    def _can_merge(self, node1, node2):
        return (
            node1.is_uniform()
            and node2.is_uniform()
            and not node1.contains_point(self.env.s_start)
            and not node2.contains_point(self.env.s_start)
            and not node1.contains_point(self.env.s_goal)
            and not node2.contains_point(self.env.s_goal)
            and self._get_uniform_value(node1) == self._get_uniform_value(node2)
        )

    def _get_uniform_value(self, node):
        return (node.x, node.y) in self.env.obs or (
            node.x,
            node.y,
        ) in self.env.dynamic_obs_cells

    def _merge_nodes(self, node1, node2):
        x = min(node1.x, node2.x)
        y = min(node1.y, node2.y)
        width = max(node1.x + node1.width, node2.x + node2.width) - x
        height = max(node1.y + node1.height, node2.y + node2.height) - y
        merged_node = TreeNode(x, y, width, height, node1.env)
        return merged_node

    def _is_rectangular(self, node):
        return (
            node.width > 0
            and node.height > 0
            and (node.width % 1 == 0)
            and (node.height % 1 == 0)
        )

    def partition(self, node):
        super().partition(node)
        self.merge_leafs()


class AdaptiveAStarExtended(AdaptiveAStar):
    def __init__(self, s_start, s_goal, heuristic_type, time=float("inf")):
        super().__init__(s_start, s_goal, heuristic_type, time)

        if time != float("inf"):
            self.name = f"Adaptive A* Extended: {time}"
        else:
            self.name = "Adaptive A* Extended"

    def change_env(self, map_name, dobs=None):
        new_env = super().super_change_env(map_name)
        self.dynamic_objects = []

        self.current_index = 0
        self.initial_path = None
        self.compute_time = None
        self.total_time = None
        self.replan_time = []
        self.agent_positions = []

        self.quadtree = ExtendedQuadtree(new_env)

        self.leaf_nodes = {}
        for leaf in self.quadtree.leafs:
            center = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
            self.leaf_nodes[center] = leaf

        self.dobs_dir = dobs


if __name__ == "__main__":
    a = AdaptiveAStar((0, 0), (0, 0), "euclidian")
    a.change_env("Evaluation/Maps/2D/block_map_250/169.json")
    # b.change_env("Evaluation/Maps/2D/main/block_14.json")

    a.quadtree.visualize()
    # b.quadtree.visualize()
    