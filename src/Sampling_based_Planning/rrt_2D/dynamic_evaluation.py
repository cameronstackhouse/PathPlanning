import json
import sys
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread

import numpy as np


from evaluate import measure_cpu_usage

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.dynamic_guided_srrt_edge import DynamicGuidedSRrtEdge
from rrt_2D.rrt_edge import RrtEdge
from rrt_2D.informed_rrt_star import IRrtStar
from rrt_2D.adaptive_srrt_edge import AdaptiveSRRTEdge

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_2D.D_star_Lite import DStar
from Search_2D.Adaptive_AStar import AdaptiveAStar

from glob import glob
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def load_existing_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []

def save_results(results, name):
    with open(name, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(f"Results saved to {name}")

def save_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, cls=NumpyEncoder)


def evaluate_algorithm_on_map(algorithm, map, OBJ_DIR, dummy_algo):
    algorithm.change_env(map, OBJ_DIR)

    tracemalloc.start()

    path, avg_cpu_load = measure_cpu_usage(algorithm.run)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    traversal_time = algorithm.total_time
    compute_time = algorithm.compute_time
    replan_time = algorithm.replan_time

    if path is not None:
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
        "replan_time": replan_time,
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
                    evaluate_algorithm_on_map, algorithm, map, OBJ_DIR, dummy_algo
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


def main():
    START = (0, 0)
    END = (0, 0)
    PSEUDO_INF = 10000000
    
    dummy_algo = DynamicGuidedSRrtEdge(START, END, 0.05, global_time=1)
    
    algorithms = [
        AdaptiveSRRTEdge(START, END, 0.05, 5, 5, min_edge_length=1, x=100, t=25),
        AdaptiveSRRTEdge(START, END, 0.05, 5, 5, min_edge_length=1, x=200, t=25),
        AdaptiveSRRTEdge(START, END, 0.05, 5, 5, min_edge_length=1, x=100, t=50),
        AdaptiveSRRTEdge(START, END, 0.05, 5, 5, min_edge_length=1, x=200, t=50),
        AdaptiveSRRTEdge(START, END, 0.05, 5, 5, min_edge_length=1, x=100, t=100),
        AdaptiveSRRTEdge(START, END, 0.05, 5, 5, min_edge_length=1, x=200, t=100)
    ]

    MAP_DIR = "src/Evaluation/Maps/2D/main/"
    OBJ_DIR = "src/Evaluation/Maps/2D/dynamic_obs.json"
    results = evaluate(algorithms, dummy_algo, MAP_DIR, OBJ_DIR)
    save_results(results, "5-seconds-2D-dynamic.json")
    
    # diff_time = evaluate(different_time_algos, dummy_algo, MAP_DIR, OBJ_DIR)
    # save_results(diff_time, "2d-different-time-srrt-edge.json")


if __name__ == "__main__":
    main()
