from D_star_Lite import DStar

class TreeNode:
    def __init__(self, x, y, width, height) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.left = None
        self.right = None
        self.parent = None

    def is_leaf(self):
        return self.left is None and self.right is None

class QuadTree:
    def __init__(self, root) -> None:
        self.root = root


class ABFStarLite(DStar):
    def __init__(self, s_start, s_goal, heuristic_type, time=...):
        super().__init__(s_start, s_goal, heuristic_type, time)
        self.heuristic_type = "manhattan"
        self.partitions = {}
    


