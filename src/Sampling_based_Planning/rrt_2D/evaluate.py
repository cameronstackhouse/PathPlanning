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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")
from load_map import create_custom_env
from stats import Stats

from glob import glob
from pathlib import Path

def evaluate(MAP_DIR: str) -> dict:
    START = (0, 0)
    END = (0, 0)
    map_name_list = list(Path(MAP_DIR).glob("*.json"))
    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    # TODO algorithms and A*
    algorithms = [
        MBGuidedSRrtEdge(START, END, 0.05, 1.0),
        #RrtEdge(START, END, 0.05, 2000),
        Rrt(START, END, 4, 0.05, 2000),
        # RrtStar(START, END, 4, 0.05, 5, 2000),
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
        for map in map_name_list:
            print(map)
            algorithm.change_env(map)

            start_time = time.time()
            path = algorithm.planning()
            total_time = time.time() - start_time

            if path:
                success += 1
                path_len.append(algorithm.utils.path_cost(path))
                energy.append(algorithm.utils.path_energy(path))
                times.append(total_time)
            else:
                path_len.append(None)
                energy.append(None)
                times.append(None)
            
            nodes.append(len(algorithm.vertex))

        success /= NUM_MAPS
        # TODO record CPU load + memory usage too
        result = {
            "Map Names": map_names,
            "Success Rate": success,
            "Path Length": path_len,
            "Time Taken To Calculate": times,
            "Energy To Traverse": energy,
            "Number of Nodes": nodes
        }

        results.append(result)

    return results

def bar_chart_compare(res1, res2, key, y_label, title):
    """
    
    """
    sns.set_style("dark")

    data1 = res1[key]
    data2 = res2[key]
    map_names = res1["Map Names"]

    data1 = [0 if v is None else v for v in data1]
    data2 = [0 if v is None else v for v in data2]

    map_names_int = list(map(int, map_names))
    sorted_indices = sorted(range(len(map_names)), key=lambda i: map_names_int[i])
    data1 = [data1[i] for i in sorted_indices]
    data2 = [data2[i] for i in sorted_indices]
    map_names = [map_names[i] for i in sorted_indices]

    bar_width = 0.35
    index = range(len(data1))
    plt.figure(figsize=(10, 5))
    plt.bar(index, data1, bar_width, label="Data Set 1")
    plt.bar([i + bar_width for i in index], data2, bar_width, label="Data Set 2")
    plt.title(title)
    plt.xticks([i + bar_width / 2 for i in index], map_names, rotation=30, ha="center")
    plt.xlabel("Map Number")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def main():
    results = evaluate("src/Evaluation/Maps/2D/block_map_25")
    data_mb = results[0]
    print(data_mb["Success Rate"])
    data_rrt = results[1]
    KEY = "Energy To Traverse"
    UNIT = "(W)"

    bar_chart_compare(data_mb, data_rrt, KEY, KEY + UNIT, f"Comparison of {KEY}")


if __name__ == "__main__":
    main()
