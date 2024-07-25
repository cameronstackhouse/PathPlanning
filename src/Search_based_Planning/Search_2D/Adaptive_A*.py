from Astar import AStar
from Quadtree import QuadTree


class AdaptiveAStar(AStar):
    def __init__(self, s_start, s_goal, heuristic_type):
        super().__init__(s_start, s_goal, heuristic_type)
        self.quadtree = None
        self.speed = 6

    def change_env(self, map_name):
        new_env = super().change_env(map_name)
        self.quadtree = QuadTree(new_env)
        
        self.leaf_nodes = {}
        for leaf in self.quadtree.leafs:
            center = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
            self.leaf_nodes[center] = leaf

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

    def run(self):
        pass

if __name__ == "__main__":
    s = AdaptiveAStar((0, 0), (0, 0), "euclidian")
    s.change_env("Evaluation/Maps/2D/main/house_11.json")
    
    path = s.planning()
    
    if path:
        s.quadtree.visualize(path)