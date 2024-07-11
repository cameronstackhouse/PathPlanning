from D_star_Lite import DStar

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

    def is_leaf(self):
        return self.left_top is None and self.right_top is None and self.left_bottom is None and self.right_bottom is None
    
    def is_uniform(self):
        """
        TODO Work out.
        """
        init_val = (self.x, self.y) in self.env.obs
        for i in range(self.x, self.x + self.width):
            for j in range(self.y, self.y + self.height):
                if ((i, j) in self.env.obs) != init_val:
                    return False
        return True

    def partition(self):
        if self.is_leaf() and not self.is_uniform():
            mid_width = self.width // 2
            mid_height = self.height // 2

            self.left_top = TreeNode(self.x, self.y, mid_width, mid_height, self.env)
            self.right_top = TreeNode(self.x + mid_width, self.y, mid_width, mid_height, self.env)
            self.left_bottom = TreeNode(self.x, self.y + mid_height, mid_width, mid_height, self.env)
            self.right_bottom = TreeNode(self.x + mid_width, self.y + mid_height, mid_width, mid_height, self.env)

            self.left = [self.left_top, self.right_top]
            self.right = [self.left_bottom, self.right_bottom]

            for child in [self.left_top, self.right_top, self.left_bottom, self.right_bottom]:
                child.parent = self
            
            self.left_top.partition()
            self.right_top.partition()
            self.left_bottom.partition()
            self.right_bottom.partition()

class QuadTree:
    def __init__(self, env) -> None:
        self.root = TreeNode(0, 0, env.x_range, env.y_range, env)
        self.leafs = [self.root]
        self.partition(self.root)
    
    def partition(self, node):
        if node.is_leaf() and not node.is_uniform():
            node.partition()
            self.partition(node.left_top)
            self.partition(node.right_top)
            self.partition(node.left_bottom)
            self.partition(node.right_bottom)

class ABFStarLite(DStar):
    def __init__(self, s_start, s_goal, heuristic_type, time=...):
        super().__init__(s_start, s_goal, heuristic_type, time)
        self.heuristic_type = "manhattan"
        self.partitions = {}
        self.quadtree = None

    def change_env(self, map_name, obj_dir=None):
        new_env = super().change_env(map_name, obj_dir)
        self.quadtree = QuadTree(new_env)

    def get_neighbor(self, s):
        pass


if __name__ == "__main__":
    s = ABFStarLite((0,0), (1, 0), "manhattan", time=2)
    s.change_env("Evaluation/Maps/2D/main/block_10.json")

    curr = s.quadtree.root
    count = 0

    while curr.left_top is not None:
        curr = curr.right_top
        count += 1

    print(count)