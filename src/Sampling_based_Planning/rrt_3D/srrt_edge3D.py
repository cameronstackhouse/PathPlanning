import math
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np


sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "../Sampling_based_Planning/"
)

from rrt_edge3D import RrtEdge
from rrt_3D.env3D import env
from rrt_3D.utils3D import (
    getDist,
    sampleFree,
    steer,
    isCollide,
    near,
    nearest,
    visualization,
    cost,
    path,
)


class SRrtEdge(RrtEdge):
    def __init__(self):
        super().__init__()

    def calculate_k(self, edge):
        """
        TODO
        """
        pass

    def get_k_partitions(self, k, edge):
        """
        TODO
        """
        pass
