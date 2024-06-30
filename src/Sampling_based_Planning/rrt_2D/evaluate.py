"""
contains the functions related to the evaluation and comparison of algorithms
"""

import json
import os
import sys
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge
from rrt_2D.rrt_edge import RrtEdge
from rrt_2D.rrt import Rrt
from rrt_2D.rrt_star import RrtStar

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_2D.D_star_Lite import DStar

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")
from load_map import load_dynamic_obs
from stats import Stats

from glob import glob
from pathlib import Path


def evaluate(MAP_DIR: str, OBJ_DIR: str = None) -> dict:
    """
    TODO
    """
    START = (0, 0)
    END = (0, 0)
    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    if OBJ_DIR:
        obj_name_list = list(Path(OBJ_DIR).glob("*.json"))
        obj_names = [obj.stem for obj in obj_name_list]

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    algorithms = [
        MBGuidedSRrtEdge(START, END, 0.05, 1.5),
        # DStar(START, END, "euclidean"),
        RrtEdge(START, END, 0.05, 2000),
        # RrtStar(START, END, 6, 0.05, 5, 2000),
    ]
    results = []

    for algorithm in algorithms:
        print(algorithm)
        # Measured metrics
        path_len = []
        times = []
        energy = []
        nodes = []
        success = 0

        # Load and evaluate each map
        for i, map in enumerate(map_name_list):
            print(map)
            # Checks if the algorithm is to be evaluated dynamically
            if OBJ_DIR:
                algorithm.change_env(map, obj_name_list[i])
            else:
                algorithm.change_env(map)

            start_time = time.time()
            path = algorithm.planning()
            total_time = time.time() - start_time

            if path:
                success += 1
                path_len.append(algorithms[1].utils.path_cost(path))
                energy.append(algorithms[1].utils.path_energy(path))
                times.append(total_time)
            else:
                path_len.append(None)
                energy.append(None)
                times.append(None)

            # nodes.append(len(algorithm.vertex))

        success /= NUM_MAPS
        # TODO record CPU load + memory usage too
        result = {
            "Algorithm": algorithm.name,
            "Map Names": map_names,
            "Success Rate": success,
            "Path Length": path_len,
            "Time Taken To Calculate": times,
            "Energy To Traverse": energy,
            "CPU Usgae": cpu_usage,
            "Number of Nodes": nodes,
        }

        results.append(result)

    return results


def save_results(results, name):
    with open(name, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to {name}")


def meaure_cpu_usage(func, *args, **kwargs):
    """
    Measures the user, system, and idle cpu usage of a given function.
    """
    start_cpu_times = psutil.cpu_times_percent(interval=None)

    result = func(*args, **kwargs)
    end_cpu_times = psutil.cpu_times_percent(interval=None)

    cpu_usage = {
        "user": end_cpu_times.user - start_cpu_times.user,
        "system": end_cpu_times.system - start_cpu_times.system,
        "idle": end_cpu_times.idle - start_cpu_times.idle,
    }

    return result, cpu_usage


def main():
    results = evaluate("src/Evaluation/Maps/2D/block_map_21")
    save_results(results, "evaluation_results.json")


if __name__ == "__main__":
    main()
