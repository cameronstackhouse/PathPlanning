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

    def in_dynamic_object(self, x, y, flag=False):
        for rectangle in self.dynamic_obs:
            if not rectangle.known and flag:
                next

            left = rectangle.current_pos[0]
            right = left + rectangle.size[0]
            bottom = rectangle.current_pos[1]
            top = bottom + rectangle.size[1]

            if left <= x <= right and bottom <= y <= top:
                rectangle.known = True
                return True
        return False

    def add_dynamic_object(self, dynamic_obj):
        self.dynamic_obs.append(dynamic_obj)
        cells_covered = self.get_covered_cells(dynamic_obj)
        self.dynamic_obs_cells.update(cells_covered)

    def get_covered_cells(self, dynamic_obj):
        width, height = dynamic_obj.size
        bottom_left_corner = dynamic_obj.current_pos

        covered_cells = []

        # Determine the min and max cell indices based on the bottom-left corner
        min_cell_x = int(bottom_left_corner[0])
        max_cell_x = int(bottom_left_corner[0] + width)
        min_cell_y = int(bottom_left_corner[1])
        max_cell_y = int(bottom_left_corner[1] + height)

        # Iterate over the range to add all covered cells
        for i in range(min_cell_x, max_cell_x):
            for j in range(min_cell_y, max_cell_y):
                covered_cells.append((i, j))

        # Add all covered cells to the dynamic_obs_cells set
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
        self.s_start = None
        self.s_end = None
        self.obs = self.gen_obs()
        self.dynamic_obs = []

    def gen_obs(self):
        obs = set()
        grid = self.data["grid"]
        self.s_start = self.data["agent"]
        self.s_goal = self.data["goal"]

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    obs.add((j, i))

        return obs
