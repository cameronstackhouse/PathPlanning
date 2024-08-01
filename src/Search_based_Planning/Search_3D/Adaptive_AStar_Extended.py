from Adaptive_AStar import AdaptiveAStar
from Search_3D.Octree import Octree


class ExtendedOctree(Octree):
    def __init__(self, env) -> None:
        super().__init__(env)
    
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
                    ):
                        pass

class AdaptiveAStarExtended(AdaptiveAStar):
    def __init__(self, time=float("inf")):
        super().__init__(time)