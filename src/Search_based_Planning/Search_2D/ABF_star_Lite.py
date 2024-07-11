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

    def is_leaf(self):
        return self.left_top is None and self.right_top is None and self.left_bottom is None and self.right_bottom is None
    
    def is_uniform(self):
        init_val = (self.x, self.y) in self.env.obs
        for i in range(self.x, self.x + self.width):
            for j in range(self.y, self.y + self.height):
                if ((i, j) in self.env.obs) != init_val:
                    return False
        return True
    
    def contains_point(self, point):
        return self.x <= point[0] < self.x + self.width and self.y <= point[1] < self.y + self.height

    def partition(self, leafs):
        if self.width == 1 and self.height == 1:
            return

        if self.is_leaf() and (not self.is_uniform() or self.contains_point(self.env.s_start) or self.contains_point(self.env.s_goal)):
            mid_width = (self.width + 1) // 2 if self.width > 1 else 1
            mid_height = (self.height + 1) // 2 if self.height > 1 else 1

            self.left_top = TreeNode(self.x, self.y, mid_width, mid_height, self.env)
            self.right_top = TreeNode(self.x + mid_width, self.y, self.width - mid_width, mid_height, self.env)
            self.left_bottom = TreeNode(self.x, self.y + mid_height, mid_width, self.height - mid_height, self.env)
            self.right_bottom = TreeNode(self.x + mid_width, self.y + mid_height, self.width - mid_width, self.height - mid_height, self.env)

            self.left = [self.left_top, self.right_top]
            self.right = [self.left_bottom, self.right_bottom]

            for child in [self.left_top, self.right_top, self.left_bottom, self.right_bottom]:
                child.parent = self

            leafs.remove(self)
            leafs.extend([self.left_top, self.right_top, self.left_bottom, self.right_bottom])
            
            self.left_top.partition(leafs)
            self.right_top.partition(leafs)
            self.left_bottom.partition(leafs)
            self.right_bottom.partition(leafs)

class QuadTree:
    def __init__(self, env) -> None:
        self.env = env
        self.root = TreeNode(0, 0, env.x_range, env.y_range, env)
        self.leafs = [self.root]
        self.partition(self.root)
    
    def partition(self, node):
        if node.is_leaf() and not node.is_uniform():
            node.partition(self.leafs)
    
    def visualize(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.root.width)
        ax.set_ylim(0, self.root.height)
        for node in self.leafs:
            if node.contains_point(self.env.s_start):
                color = 'blue'
            elif node.contains_point(self.env.s_goal):
                color = 'green'
            elif node.is_uniform():
                color = 'black' if (node.x, node.y) in node.env.obs else 'white'
            else:
                color = 'gray'
            rect = patches.Rectangle((node.x, node.y), node.width, node.height, linewidth=1, edgecolor='r', facecolor=color, fill=True)
            ax.add_patch(rect)
        plt.gca().invert_yaxis()
        plt.show()

class ABFStarLite(DStar):
    def __init__(self, s_start, s_goal, heuristic_type, time=...):
        super().__init__(s_start, s_goal, heuristic_type, time)
        self.heuristic_type = "manhattan"
        self.quadtree = None

    def change_env(self, map_name, obj_dir=None):
        new_env = super().change_env(map_name, obj_dir)
        self.quadtree = QuadTree(new_env)

        self.rhs = {}
        self.g = {}
        self.leaf_nodes = {}
        for leaf in self.quadtree.leafs:
            center = None
            if [leaf.x, leaf.y] == self.s_start or [leaf.x, leaf.y] == self.s_goal:
                center = (leaf.x, leaf.y)
            else:
                center = (leaf.x + leaf.width // 2, leaf.y + leaf.height // 2)
            self.rhs[center] = float("inf")
            self.g[center] = float("inf")
            self.leaf_nodes[center] = leaf
        
        self.rhs[self.s_start] = float("inf")
        self.g[self.s_start] = float("inf")
        # TODO look at
        self.g[self.s_goal] = float("inf")
        
        self.rhs[self.s_goal] = 0.0
        self.U = {}
        self.U[self.s_goal] = self.CalculateKey(self.s_goal)

    def get_neighbor(self, s):
        neighbours = set()
        leaf = self.leaf_nodes[s]
        adjacent_directions = [(0, leaf.height), (0, -leaf.height), (leaf.width, 0), (-leaf.width, 0),
                               (leaf.width, leaf.height), (-leaf.width, leaf.height), 
                               (leaf.width, -leaf.height), (-leaf.width, -leaf.height)]
        
        for direction in adjacent_directions:
            neighbor_center = (s[0] + direction[0], s[1] + direction[1])
            if neighbor_center in self.leaf_nodes:
                neighbours.add(neighbor_center)
        
        return neighbours

    def find_leaf(self, point):
        for leaf in self.quadtree.leafs:
            if leaf.contains_point(point):
                return leaf
        return None

if __name__ == "__main__":
    s = ABFStarLite((0,0), (1, 0), "manhattan", time=2)
    s.change_env("Evaluation/Maps/2D/main/block_21.json")
    s.quadtree.visualize()
    # res = s.ComputePath()
    # print(res)