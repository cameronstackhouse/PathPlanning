"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    # TODO Add params
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
