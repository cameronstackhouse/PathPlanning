import os
import sys
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from rrt_2D import env, plotting, utils
from rrt import Node, Rrt

class Edge:
    def __init__(self, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2

class RrtEdge(Rrt):
    def __init__(self, start, end, goal_sample_rate, iter_max):
        self.s_start = Node(start)
        self.s_goal = Node(end)

        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.nodes = [self.s_start]
        self.edges = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(start, end)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        for _ in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor()
    
    def nearest_neighbour(self, node_list, edge_list, n):
        nearest_node = super.nearest_neigbour(node_list, n)
        nearest_edge = self.nearest_edge_projection(edge_list, n)
        

    @staticmethod
    def nearest_edge_projection(edge_list, n):
        pass