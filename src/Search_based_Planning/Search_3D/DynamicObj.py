class DynamicObj:
    def __init__(self) -> None:
        self.velocity = []
        self.size = []
        self.known = False
        self.current_pos = []
        self.index = 0
        self.init_pos = None
        self.old_pos = None
        self.corners = []

    def update_pos(self):
        velocity = self.velocity
        new_pos = [
            self.current_pos[0] + (velocity[0]),
            self.current_pos[1] + (velocity[1]),
            self.current_pos[2] + (velocity[2]),
        ]

        return new_pos

    def contains_point(self, point):
        px, py, pz = point

        x_min, y_min, z_min, x_max, y_max, z_max = self.corners

        if x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max:
            return True

        return False
