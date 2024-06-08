import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D import utils
from rrt_2D.rrt import Node
from rrt_2D.rrt_edge import Edge
from rrt_2D.guided_srrt_edge import GuidedSRrtEdge

class GuidedSRrtEdgeDynamic(GuidedSRrtEdge):
    def __init__(self, start, end, goal_sample_rate, iter_max, min_edge_length=4):
        super().__init__(start, end, goal_sample_rate, iter_max, min_edge_length)

    
    