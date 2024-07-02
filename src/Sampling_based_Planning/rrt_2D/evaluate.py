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
import tracemalloc

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge
from rrt_2D.rrt_edge import RrtEdge
from rrt_2D.rrt import Rrt
from rrt_2D.informed_rrt_star import IRrtStar

# from rrt_3D.mb_guided_srrt_edge3D import MbGuidedSrrtEdge
# from rrt_3D.rrt3D import rrt

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_2D.D_star_Lite import DStar

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")
from load_map import load_dynamic_obs
from stats import Stats

from glob import glob
from pathlib import Path


def evaluate(MAP_DIR: str, OBJ_DIR: str = None, TYPE: str = "2D") -> dict:
    """
    TODO
    """
    START = (0, 0)
    END = (0, 0)
    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    # TODO add dynamic capability to the evaluation suite
    if OBJ_DIR:
        obj_name_list = list(Path(OBJ_DIR).glob("*.json"))
        obj_names = [obj.stem for obj in obj_name_list]

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    algorithms = []

    if TYPE == "2D":
        algorithms = [
            MBGuidedSRrtEdge(START, END, 0.05, 2),
            IRrtStar(START, END, 5, 0.05, 5, 4000),
            #DStar(START, END, "euclidean"),
            RrtEdge(START, END, 0.05, 4000),
        ]

    else:
        # TODO 3D Algorithms
        pass

    results = []

    for algorithm in algorithms:
        print(algorithm)
        # Measured metrics
        path_len = []
        times = []
        energy = []
        nodes = []
        success = 0

        # TODO
        traversal_time = []
        cpu_usage = []
        memory_used = []  # psutil.virtual_memory()

        # Load and evaluate each map
        for i, map in enumerate(map_name_list):
            print(map)
            # Checks if the algorithm is to be evaluated dynamically
            if OBJ_DIR:
                algorithm.change_env(map, obj_name_list[i])
            else:
                algorithm.change_env(map)

            path = None
            tracemalloc.start()

            start_time = time.time()
            if OBJ_DIR:
                path = algorithm.run()
            else:
                path = algorithm.planning()

            cpu_usage.append(algorithm.peak_cpu)

            total_time = time.time() - start_time
            peak = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()

            memory_used.append(peak)

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
            "CPU Usage": cpu_usage,
            "Number of Nodes": nodes,
            "Memory Used": memory_used,
        }

        results.append(result)

    return results


def save_results(results, name):
    with open(name, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to {name}")


def main():
    results = evaluate("src/Evaluation/Maps/2D/main", TYPE="2D")
    save_results(results, "evaluation_results.json")


if __name__ == "__main__":
    main()
