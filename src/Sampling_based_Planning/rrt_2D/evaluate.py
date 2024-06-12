"""
contains the functions related to the evaluation and comparison 
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/"
)
from load_map import create_custom_env

from glob import glob
from pathlib import Path

def comapare(algo_1, algo_2):
    pass

def gen_report(algo):
    MAP_DIR = "src/Evaluation/Maps/2D"
    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    # TODO metrics

    # Load and evaluate each map
    for map in map_name_list:
        with open(map, 'r') as f:
            map_data = json.load(f)
            alg = MBGuidedSRrtEdge(map_data["agent"], map_data["goal"], 0.05)

            alg.env = create_custom_env(map)
            alg.plotting.env = alg.env
            alg.plotting.obs_bound = alg.env.obs_boundary
            alg.plotting.obs_circle = alg.env.obs_circle
            alg.plotting.obs_rectangle = alg.env.obs_rectangle

            alg.utils.env = alg.env
            alg.utils.obs_boundary = alg.env.obs_boundary
            alg.utils.obs_circle = alg.env.obs_circle
            alg.utils.obs_rectangle = alg.env.obs_rectangle

            path = alg.planning()
            
            alg.plotting.animation(alg.vertex, path, "Bounded Guided SRRT-Edge", False)


def main():
    return 0

if __name__ == "__main__":
    gen_report(1)
