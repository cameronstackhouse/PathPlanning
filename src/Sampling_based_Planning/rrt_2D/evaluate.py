"""
contains the functions related to the evaluation and comparison 
"""

import json
import os
import sys
import time
import psutil

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge
from rrt_2D.rrt_edge import RrtEdge

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")
from load_map import create_custom_env

from glob import glob
from pathlib import Path


def comapare(algo_1, algo_2):
    pass


def gen_report():
    MAP_DIR = "src/Evaluation/Maps/2D/block_map_25"
    map_name_list = list(Path(MAP_DIR).glob("*.json"))
    NUM_MAPS = len(map_name_list)

    # TODO algorithms

    # TODO metrics
    avg_path_len = 0
    avg_time = 0
    success = 0

    # Load and evaluate each map
    for map in map_name_list:
        # TODO run multiple algorithms
        alg = MBGuidedSRrtEdge((0, 0), (0, 0), 0.05, 0.1)

        alg.change_env(map)

        start_time = time.time()
        path = alg.planning()
        total_time = time.time() - start_time

        print(map)

        if path:
            success += 1
            avg_path_len += alg.utils.path_cost(path)
            avg_time += total_time

           # alg.plotting.animation(alg.vertex, path, f"Bounded Guided SRRT-Edge {map}", False)

    # TODO Gen report here
    # for algorithm in algorithms:
    #     pass

    avg_path_len /= success
    avg_time /= success
    success /= NUM_MAPS

    print(f"Success percentage: {success * 100}%")
    print(f"Cost: {avg_path_len}. Time: {avg_time}")


def main():
    return 0


if __name__ == "__main__":
    gen_report()
