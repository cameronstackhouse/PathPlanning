import os
import sys


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.flag = "VALID"
        self.partition = 1


class Edge:
    def __init__(self, n_p, n_c):
        self.parent = n_p
        self.child = n_c
        self.flag = "VALID"


class DynamicObj:
    def __init__(self) -> None:
        self.velocity = []
        self.shape = []
        self.known = False


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

    def invalidate(self):
        pass

    def regrow(self):
        pass

    def reconnect(self):
        pass


if __name__ == "__main__":
    start = (900, 900)
    end = (901, 900)
    goal_sample_rate = 5
    rrt = DynamicGuidedSRrtEdge(start, end, goal_sample_rate)
    rrt.run()
