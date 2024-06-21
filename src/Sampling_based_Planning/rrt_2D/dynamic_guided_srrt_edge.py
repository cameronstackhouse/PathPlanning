import math
import os
import sys


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge


class DynamicObj:
    def __init__(self) -> None:
        self.velocity = []
        self.size = []
        self.known = False
        self.current_pos = []


class DynamicGuidedSRrtEdge(MBGuidedSRrtEdge):
    def __init__(
        self,
        start,
        end,
        goal_sample_rate,
        global_time=1.0,
        local_time=0.05,
        mem=100000,
        min_edge_length=4,
    ):
        super().__init__(
            start, end, goal_sample_rate, global_time, mem, min_edge_length
        )
        self.agent_pos = self.s_start
        self.dynamic_objects = []
        self.invalidated_nodes = set()
        self.invalidated_edges = set()
        self.speed = 6
        self.current_index = 0

    def run(self):
        # Find initial global path
        global_path = self.planning()

        if global_path:
            # Traverse the path
            time = 0
            current = global_path[self.current_index]
            GOAL = global_path[-1]
            # While the final node has not been reached
            while current != GOAL:
                self.update_object_positions()
                # Update agents view
                new_coords = self.move()
                if new_coords == [None, None]:
                    # TODO reroute and move
                    pass
                else:
                    current = new_coords
        else:
            print("No path found")

    def move(self, path, mps=6) -> bool:
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.s_goal.coords

        current_pos = self.agent_pos
        next_node = path[self.current_index + 1]

        # Checks for collision between current point and the waypoint node
        # TODO need to make a one-step-ahead check
        if self.utils.is_collision(current_pos, next_node):
            return [None, None]

        seg_distance = self.utils.euclidian_distance(current_pos, next_node)

        direction = (
            (next_node[0] - current_pos[0]) / seg_distance,
            (next_node[1] - current_pos[1]) / seg_distance,
        )

        new_pos = (
            current_pos[0] + direction[0] * mps,
            current_pos[1] + direction[1] * mps,
        )

        # Checks for overshoot
        if self.utils.euclidian_distance(current_pos, new_pos) >= seg_distance:
            self.agent_pos = next_node
            self.current_index += 1
            return path[self.current_index].coords

        return new_pos

    def update_object_positions(self, time_steps=1):
        """
        Updates the position of dynamic objects over one timestep.
        The object moves in a fixed direction and comes to an immediate stop
        if a fixed object is detected.

        :param: time_steps: the number of time steps in the future to predict the object positions.
        """
        for object in self.dynamic_objects:
            # Attempt to move in direction of travel
            velocity = object.velocity
            new_pos = object.current_pos + (velocity * time_steps)
            # TODO check for fixed object (or just allow to pass through)

            object.current_pos = new_pos

    def update_world_view(self):
        """
        Discovers dynamic objects within the vicinity of the UAV.
        """
        VISION = 30
        pos = self.agent_pos

        for obj in self.dynamic_objects:
            obj_pos, obj_size = obj
            obj_center_x = obj_pos[0]
            obj_center_y = obj_pos[1]
            obj_half_height = obj_size[0] / 2
            obj_half_width = obj_size[1] / 2

            # Calculate the distance from the UAV to the edges of the object's bounding box
            nearest_x = max(
                obj_center_x - obj_half_width,
                min(pos[0], obj_center_x + obj_half_width),
            )
            nearest_y = max(
                obj_center_y - obj_half_height,
                min(pos[1], obj_center_y + obj_half_height),
            )

            distance = math.sqrt((nearest_x - pos[0]) ** 2 + (nearest_y - pos[1]) ** 2)

            # Check if the object is within the UAV's vision radius
            if distance <= VISION:
                obj.known = True
            else:
                obj.known = False

    def in_dynamic_obj(self, node, obj):
        """
        TODO
        """
        x, y = node.coords
        x0, y0 = obj.current_pos
        width, height = obj.size
        return (x0 <= x < x0 + width) and (y0 <= y < y0 + height)

    def invalidate(self, obj):
        """
        Invalidates nodes and edges which have been blocked by a dynamic object.
        """
        # Check if nodes exist lie within the current object
        for node in self.vertex:
            if self.in_dynamic_obj(node, obj):
                self.invalidated_nodes.add(tuple(node))

        # Check if any part of an edge lies within the object
        for edge in self.edges:
            n1 = edge.node_1
            n2 = edge.node_2

            if self.utils.is_collision(n1, n2):
                self.invalidated_edges.add(tuple(edge))

    def revalidate(self):
        """
        Assesses invalidated nodes and edges to see if they are no longer invalidated by
        dynamic objects.
        """
        # Check nodes
        for node in self.invalidated_nodes:
            # Check if a node is still inside one of the dynamic objects
            blocked = False
            for object in self.dynamic_objects:
                if self.in_dynamic_obj(node, object):
                    blocked = True
                    break

            if not blocked:
                self.invalidated_nodes.remove(tuple(node))

        # Check edges
        for edge in self.invalidated_edges:
            n1 = edge.node_1
            n2 = edge.node_2

            if not self.utils.is_collision(n1, n2):
                self.invalidated_edges.remove(tuple(edge))

    def regrow(self):
        # TODO
        pass

    def reconnect(self):
        # TODO
        pass


if __name__ == "__main__":
    start = (900, 900)
    end = (901, 900)
    goal_sample_rate = 5
    rrt = DynamicGuidedSRrtEdge(start, end, goal_sample_rate)
    rrt.run()
