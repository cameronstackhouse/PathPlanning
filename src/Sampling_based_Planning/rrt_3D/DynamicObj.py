class DynamicObj:
    def __init__(self) -> None:
        self.velocity = []
        self.size = []
        self.known = False
        self.current_pos = []
        self.index = 0
        self.init_pos = None
        self.old_pos = None

    def update_pos(self):
        velocity = self.velocity
        new_pos = [
            self.current_pos[0] + (velocity[0]),
            self.current_pos[1] + (velocity[1]),
            self.current_pos[2] + (velocity[2]),
        ]

        return new_pos

    def predict_future_positions(self, prediction_horizon):
        future_positions = []

        for i in range(1, prediction_horizon + 1):
            future_position = (
                self.current_pos[0] + (self.velocity[0] * i),
                self.current_pos[1] + (self.velocity[1] * i),
                self.current_pos[2] + (self.velocity[2] * i),
            )
            future_positions.append(future_position)

        return future_positions
