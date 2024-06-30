"""
Environment for rrt_2D
@author: huiming zhou
"""


import numpy as np


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
            [x_max - 1, y_min, 1, y_max - y_min],
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # NOTE:
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2],
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        # NOTE: x,y,r
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
            [125, 532, 50],
        ]

        return obs_cir

    def add_rect(self, x, y, height, width):
        self.obs_rectangle.append([x, y, width, height])

    def update_obj_pos(self, index, new_x, new_y):
        if 0 <= index < len(self.obs_rectangle):
            rect = self.obs_rectangle[index]
            self.obs_rectangle[index] = [new_x, new_y, rect[2], rect[3]]
        else:
            print("Error")


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
        visited = np.zeros((rows, cols), dtype=bool)
        rectangles = []

        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == 1 and not visited[y, x]:
                    # Find the extent of the rectangle
                    rect_x, rect_y = x, y
                    while rect_x < cols and grid[y][rect_x] == 1:
                        rect_x += 1
                    while rect_y < rows and all(
                        grid[rect_y][i] == 1 for i in range(x, rect_x)
                    ):
                        rect_y += 1
                    width = rect_x - x
                    height = rect_y - y
                    rectangles.append([x, y, width, height])
                    visited[y:rect_y, x:rect_x] = True

        return rectangles
