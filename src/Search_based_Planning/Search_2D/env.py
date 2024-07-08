"""
Env 2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = 1000  # size of background
        self.y_range = 1000
        self.motions = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]
        self.obs = self.obs_map()
        self.dynamic_obs = []
        self.dynamic_obs_cells = set()

    def get_covered_cells(self, dynamic_obj):
        x, y = dynamic_obj.size
        center = dynamic_obj.current_pos

        covered_cells = []

        half_x = x / 2
        half_y = y / 2

        min_x = center[0] - half_x
        max_x = center[0] + half_x
        min_y = center[1] - half_y
        max_y = center[1] + half_y

        min_cell_x = int(min_x)
        max_cell_x = int(max_x)
        min_cell_y = int(min_y)
        max_cell_y = int(max_y)

        for i in range(min_cell_x, max_cell_x + 1):
            for j in range(min_cell_y, max_cell_y + 1):
                covered_cells.append((i, j))

        for cell in covered_cells:
            self.dynamic_obs_cells.add(cell)

    def update_dynamic_obj_pos(self, index, new_x, new_y):
        obj = self.dynamic_obs[index]
        obj.current_pos = [new_x, new_y]
        self.get_covered_cells(obj)

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))

        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs


class CustomEnv(Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.obs = []
        self.obs = self.gen_obs()
        self.dynamic_obs = []

    def gen_obs(self):
        obs = set()
        grid = self.data["grid"]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    obs.add((j, i))

        return obs
