"""
contains the functions related to the evaluation and comparison of algorithms
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
from rrt_2D.rrt import Rrt
from rrt_2D.rrt_star import RrtStar

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")
from load_map import create_custom_env
from stats import Stats

from glob import glob
from pathlib import Path


def comapare(algo_1, algo_2):
    """
    TODO
    """
    pass


def gen_report(MAP_DIR):
    """
    TODO
    """
    START = (0, 0)
    END = (0, 0)
    map_name_list = list(Path(MAP_DIR).glob("*.json"))
    NUM_MAPS = len(map_name_list)

    # TODO algorithms
    algorithms = [
        MBGuidedSRrtEdge(START, END, 0.05, 0.1),
        RrtEdge(START, END, 0.05, 2000),
        Rrt(START, END, 4, 0.05, 2000),
        RrtStar(START, END, 4, 0.05, 5, 2000),
    ]

    # Measured metrics
    avg_path_len = 0
    avg_time = 0
    avg_energy = 0
    success = 0

    # Load and evaluate each map
    for map in map_name_list:
        # TODO run multiple algorithms
        alg = MBGuidedSRrtEdge((0, 0), (0, 0), 0.05, 0.3)

        alg.change_env(map)

        start_time = time.time()
        path = alg.planning()
        total_time = time.time() - start_time

        if path:
            success += 1
            avg_path_len += alg.utils.path_cost(path)
            avg_energy += alg.utils.path_energy(path)
            avg_time += total_time
            print(map)

    # TODO Gen report here
    # for algorithm in algorithms:
    #     pass

    avg_path_len /= success
    avg_time /= success
    avg_energy /= success
    success /= NUM_MAPS

    print(f"Success percentage: {success * 100}%")
    print(f"Cost: {avg_path_len}.\nTime: {avg_time}.\nEnergy: {avg_energy}.")


def main():
    return 0


if __name__ == "__main__":
    gen_report("src/Evaluation/Maps/2D/block_map_25")
