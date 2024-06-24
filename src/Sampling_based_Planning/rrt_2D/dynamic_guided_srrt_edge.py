import math
import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.plotting import DynamicPlotting
from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge
from rrt_2D.rrt import Node, DynamicObj


class DynamicGuidedSRrtEdge(MBGuidedSRrtEdge):
    def __init__(
        self,
        start,
        end,
        goal_sample_rate,
        global_time=2.0,
        local_time=0.05,
        mem=100000,
        min_edge_length=4,
    ):
        super().__init__(
            start, end, goal_sample_rate, global_time, mem, min_edge_length
        )
        self.path = []

    def run(self):
        """
        Attempts to run the algorithm to initially find a global path
        and then traverse the environment while avoiding dynamic objects
        """
        taken_path = []
        # Find initial global path without knowledge of dynamic objects
        global_path = self.planning()[::-1]
        self.initial_path = global_path
        self.init_dynamic_obs(1)

        if global_path:
            current = global_path[self.current_index]
            GOAL = global_path[-1]
            # While the final node has not been reached
            while current != GOAL:
                # print(
                #     f"Timestep: {self.time_steps}\ninvlaidated edges: {len(self.invalidated_edges)}\nInvalidated nodes: {len(self.invalidated_nodes)}"
                # )
                current = global_path[self.current_index]
                self.update_object_positions()
                self.update_world_view()
                new_coords = self.move(global_path, self.speed)
                # If the UAV can't move to the next position
                if new_coords == [None, None]:
                    # Attempt to reconnect
                    if not self.reconnect():
                        new_path = self.regrow()
                        if not new_path:
                            return False
                        else:
                            global_path = new_path[::-1]
                else:
                    self.agent_positions.append(new_coords)
                    current = new_coords
                    self.agent_pos = new_coords
                self.time_steps += 1
            # TODO update
            self.path = global_path
            return True
        else:
            return False

    def move(self, path, mps=6) -> bool:
        """
        Attempts to move the agent forward by a fixed amount of meters per second.
        """
        if self.current_index >= len(path) - 1:
            return self.s_goal.coords

        current_pos = self.agent_pos
        next_node = path[self.current_index + 1]

        # Checks for collision between current point and the waypoint node
        # TODO need to make a one-step-ahead check Might need to change based on implementation
        if self.utils.is_collision(Node(current_pos), Node(next_node)):
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
            return next_node

        return new_pos

    def init_dynamic_obs(self, n_obs):
        """
        TODO, tidy
        """
        for _ in range(n_obs):
            new_obj = DynamicObj()
            new_obj.velocity = [
                0,
                0,
            ]
            new_obj.size = [50, 50]
            new_obj.current_pos = [707, 610]
            new_obj.init_pos = new_obj.current_pos

            self.env.add_rect(
                new_obj.current_pos[0],
                new_obj.current_pos[1],
                new_obj.size[0],
                new_obj.size[1],
            )

            # TODO THIS IS THE KEY!
            self.utils.env.add_rect(
                new_obj.current_pos[0],
                new_obj.current_pos[1],
                new_obj.size[0],
                new_obj.size[1],
            )

            new_obj.index = len(self.env.obs_rectangle) - 1
            self.dynamic_objects.append(new_obj)

        new_obj = DynamicObj()
        new_obj.velocity = [
            0,
            0,
        ]
        new_obj.size = [100, 100]
        new_obj.current_pos = [787, 635]
        new_obj.init_pos = new_obj.current_pos

        self.env.add_rect(
            new_obj.current_pos[0],
            new_obj.current_pos[1],
            new_obj.size[0],
            new_obj.size[1],
        )

        self.utils.env.add_rect(
            new_obj.current_pos[0],
            new_obj.current_pos[1],
            new_obj.size[0],
            new_obj.size[1],
        )

        new_obj.index = len(self.env.obs_rectangle) - 1
        self.dynamic_objects.append(new_obj)

    def update_object_positions(self, time_steps=1):
        """
        Updates the position of dynamic objects over one timestep.
        The object moves in a fixed direction and comes to an immediate stop
        if a fixed object is detected.

        :param: time_steps: the number of time steps in the future to predict the object positions.
        """
        for object in self.dynamic_objects:
            # Attempt to move in direction of travel
            new_pos = object.update_pos()

            self.env.update_obj_pos(object.index, new_pos[0], new_pos[1])
            self.utils.env.update_obj_pos(object.index, new_pos[0], new_pos[1])

    def update_world_view(self):
        """
        Discovers dynamic objects within the vicinity of the UAV and updates
        validated and invalidated edges.
        """
        self.revalidate()
        VISION = 30
        pos = self.agent_pos

        for obj in self.dynamic_objects:
            self.invalidate(obj)
            obj_pos, obj_size = obj.current_pos, obj.size
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
                self.invalidated_nodes.add(node)

        # Check if any part of an edge lies within the object
        for edge in self.edges:
            n1 = edge.node_1
            n2 = edge.node_2

            if self.utils.is_collision(n1, n2):
                self.invalidated_edges.add(edge)

    def invalidated_graph_after_n_steps(self, n=1):
        pass

    def revalidate(self):
        """
        Assesses invalidated nodes and edges to see if they are no longer invalidated by
        dynamic objects.
        TODO, NOT CALLED ANYWHERE
        """
        # Check nodes
        nodes_to_remove = []
        for node in self.invalidated_nodes:
            # Check if a node is still inside one of the dynamic objects
            blocked = False
            for object in self.dynamic_objects:
                if self.in_dynamic_obj(node, object):
                    blocked = True
                    break

            if not blocked:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.invalidated_nodes.remove(node)

        # Check edges
        edges_to_remove = []
        for edge in self.invalidated_edges:
            n1 = edge.node_1
            n2 = edge.node_2

            if not self.utils.is_collision(n1, n2):
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            self.invalidated_edges.remove((edge))

    def regrow(self):
        """
        TODO:
        Regrows the tree to try and find path from current to end node.
        """
        self.vertex.clear()
        self.edges.clear()
        current_pos_node = Node(self.agent_pos)

        for node in self.invalidated_nodes:
            if node in self.vertex:
                self.vertex.remove(node)

        for edge in self.invalidated_edges:
            if edge in self.edges:
                self.edges.remove(edge)

        self.invalidated_nodes.clear()
        self.invalidated_edges.clear()

        self.s_start = current_pos_node
        self.vertex = [current_pos_node]
        self.edges = []

        new_path = self.planning()

        if new_path:
            self.current_index = 0
            return new_path
        else:
            return None

    def reconnect(self):
        """ """
        # TODO see if wating for one time period would clear it, AKA the edge is valid
        current_pos = self.agent_pos
        goal_pos = self.initial_path[self.current_index + 1]
        # TODO go one timestep ahead
        return not self.utils.is_collision(Node(current_pos), Node(goal_pos))


if __name__ == "__main__":
    start = (582, 230)
    end = (901, 900)
    goal_sample_rate = 0.05
    rrt = DynamicGuidedSRrtEdge(start, end, goal_sample_rate)

    success = rrt.run()

    dynamic_objects = rrt.dynamic_objects
    nodelist = rrt.vertex
    path = rrt.path

    plotter = DynamicPlotting(
        start,
        end,
        dynamic_objects,
        rrt.time_steps,
        rrt.agent_positions,
        rrt.initial_path,
    )

    plotter.env = rrt.env

    plotter.animation(nodelist, path, "Test", animation=False)
