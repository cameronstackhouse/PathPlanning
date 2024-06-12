"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (0, 1000)
        self.y_range = (0, 1000)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    def obs_boundary(self):
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        obs_boundary = [
            [x_min, y_min, x_max - x_min, 1], 
            [x_min, y_max - 1, x_max - x_min, 1], 
            [x_min, y_min, 1, y_max - y_min], 
            [x_max - 1, y_min, 1, y_max - y_min] 
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # NOTE: 
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        # NOTE: x,y,r
        obs_cir = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3],
            [50, 50, 10],
            [15, 57, 1],
            [69, 10, 7],
            [30, 78, 10],
            [12, 45, 10],
            [31, 44, 5],
            [6.2, 24.1, 5],
            [86, 34, 15],
            [63, 42.4, 6],
            [14, 69.3, 5],
            [500, 500, 200],
            [737, 183, 50],
            [796, 375, 50],
            [382, 146, 100],
            [125, 532, 50]
        ]

        return obs_cir


class CustomEnv(Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.obs_rectangle = self.gen_rectangles()
        self.obs_circle = []

    def gen_rectangles(self):
        # TODO. save this stuff so it doesnt have to be recalculated each time, reduces collision checking
        """
        Method which creates larger rectangles by grouping adjacent 1.0 points
        """
        grid = self.data["grid"]
        rows = len(grid)
        cols = len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        rectangles = []

        def dfs(x, y):
            stack = [(x, y)]
            min_x, max_x = x, x
            min_y, max_y = y, y
            while stack:
                cx, cy = stack.pop()
                if 0 <= cx < rows and 0 <= cy < cols and grid[cx][cy] == 1.0 and not visited[cx][cy]:
                    visited[cx][cy] = True
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    neighbors = [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
                    for nx, ny in neighbors:
                        stack.append((nx, ny))
            return min_x, min_y, max_x - min_x + 1, max_y - min_y + 1

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1.0 and not visited[i][j]:
                    rect = dfs(i, j)
                    rectangles.append([rect[1], rect[0], rect[3], rect[2]])  # x, y, width, height

        return rectangles