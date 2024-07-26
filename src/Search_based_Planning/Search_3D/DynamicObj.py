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
        px, py = point
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        x3, y3 = self.corners[2]
        x4, y4 = self.corners[3]

        if min(x1, x2, x3, x4) <= px <= max(x1, x2, x3, x4) and min(
            y1, y2, y3, y4
        ) <= py <= max(y1, y2, y3, y4):
            return True
        return False