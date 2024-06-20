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
        self.dynamic_objects = []
        self.invalidated_nodes = set()
        self.invalidated_edges = set()

    def run(self):
        # Find initial global path
        global_path = self.planning()

        if global_path:
            # Traverse the path
            current_pos = global_path[0]
            current_index = 0
            while current_pos != global_path[-1]:
                next_pos = global_path[current_index + 1]

                # Check for obstacles between now and next waypoint
                if self.utils.is_collision(Node(current_pos), Node(next_pos)):
                    pass

                    # If some exist and collide, split the path into two partitions and invalidate tree edges/nodes
                    # see if waiting for one time step would clear it
                    # Otherwise attempt to regrow tree to connect to disconnected tree with a time bound and reroute
                else:
                    current_index += 1
                    current_pos = next_pos
        else:
            print("No path found")
    
    def update_object_positions(self):
        # TODO
        pass

    def update_world_view(self):
        # TODO
        pass

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
            x, y = node.coords

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
        # Reassess the validity of nodes
        for node in self.invalidated_nodes:
            pass

        # Reassess the validity of edges
        for edge in self.invalidated_edges:
            pass


if __name__ == "__main__":
    start = (900, 900)
    end = (901, 900)
    goal_sample_rate = 5
    rrt = DynamicGuidedSRrtEdge(start, end, goal_sample_rate)
    rrt.run()
