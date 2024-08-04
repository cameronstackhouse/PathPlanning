"""
contains the functions related to the evaluation and comparison of algorithms
"""

import json
import os
import sys
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread

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
from Search_2D.Adaptive_AStar import AdaptiveAStar

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Evaluation/")

from glob import glob
from pathlib import Path


def evaluate_algorithm_on_map(algorithm, map, dummy_algo):
    algorithm.change_env(map)

    tracemalloc.start()

    if algorithm.name == "D* Lite":
        path, avg_cpu_load = measure_cpu_usage(algorithm.ComputePath)
    else:
        path, avg_cpu_load = measure_cpu_usage(algorithm.planning)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    traversal_time = algorithm.total_time
    compute_time = algorithm.compute_time

    if path is not None and len(path) != 0:
        path_len = dummy_algo.utils.path_cost(path)
        energy = dummy_algo.utils.path_energy(path)
    else:
        path_len = None
        energy = None

    return {
        "path": path,
        "path_len": path_len,
        "energy": energy,
        "compute_time": compute_time,
        "traversal_time": traversal_time,
        "cpu_usage": avg_cpu_load,
        "memory_used": peak,
    }


def evaluate(algorithms, dummy_algo, MAP_DIR: str, OBJ_DIR: str = None):
    s_time = time.time()

    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    results = []

    for algorithm in algorithms:
        count = 0
        print(algorithm)
        algorithm.speed = 6
        success = 0
        map_results = [None] * len(map_name_list)

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    evaluate_algorithm_on_map, algorithm, map, dummy_algo
                ): index
                for index, map in enumerate(map_name_list)
            }

            for future in as_completed(futures):
                count += 1
                index = futures[future]

                print(f"{count}:{map_names[index]}")

                result = future.result()
                map_results[index] = result
                if result["path"] is not None:
                    success += 1

        success_rate = success / NUM_MAPS
        result = {
            "Algorithm": algorithm.name,
            "Map Names": map_names,
            "Results": map_results,
            "Success Rate": success_rate,
        }
        results.append(result)

    print(f"TIME: {time.time() - s_time}")
    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_results(results, name):
    """
    Saves the results obtained from the evaluation script in JSON format.
    """
    with open(name, "w") as file:
        json.dump(results, file, indent=4, cls=NumpyEncoder)

    print(f"Results saved to {name}")


def measure_cpu_usage(func, *args, **kwargs):
    """
    Measures CPU utalisation of a function as a sum of the CPU percentage
    utalised at each time step.
    """

    def measure():
        while running:
            cpu_percentages.append(psutil.cpu_percent(interval=0.1))

    # process = psutil.Process()

    cpu_percentages = []
    running = True
    measurement_thread = Thread(target=measure)
    measurement_thread.start()

    result = func(*args, **kwargs)

    running = False
    measurement_thread.join()

    cpu_load = cpu_percentages

    return (result, cpu_load)


def main():
    START = (0, 0)
    END = (0, 0)
    PSEUDO_INF = 10000000

    dummy_algo = MBGuidedSRrtEdge(START, END, 0.05, time=1)

    algorithms = [
        DStar(START, END, "euclidian", time=5)
        # AdaptiveAStar(START, END, "euclidian", time=5),
        # IRrtStar(START, END, 10, 0.05, 5, iter_max=PSEUDO_INF, time=5),
        # MBGuidedSRrtEdge(START, END, 0.05, time=5),
        # RrtEdge(START, END, 0.05, iter_max=PSEUDO_INF, time=5),
        # MBGuidedSRrtEdge(START, END, 0.05, time=10),
        # MBGuidedSRrtEdge(START, END, 0.05, time=15),
        # MBGuidedSRrtEdge(START, END, 0.05, time=20),
        # IRrtStar(START, END, 10, 0.05, 5, iter_max=PSEUDO_INF, time=5)
    ]
    results = evaluate(algorithms, dummy_algo, "src/Evaluation/Maps/2D/main")
    save_results(results, "2D-static-dstar-5.json")


if __name__ == "__main__":
    main()
