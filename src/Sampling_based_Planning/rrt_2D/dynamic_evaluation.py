import json
import sys
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread


from evaluate import save_results, measure_cpu_usage

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_2D.dynamic_guided_srrt_edge import DynamicGuidedSRrtEdge
from rrt_2D.rrt_edge import RrtEdge
from rrt_2D.informed_rrt_star import IRrtStar

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_2D.D_star_Lite import DStar
from Search_2D.Adaptive_AStar import AdaptiveAStar

from glob import glob
from pathlib import Path


def load_existing_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []


def save_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


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


def evaluate(MAP_DIR: str, OBJ_DIR: str = None):
    s_time = time.time()

    START = (0, 0)
    END = (0, 0)
    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    dummy_algo = DynamicGuidedSRrtEdge(START, END, 0.05, global_time=1)

    algorithms = [
        # DStar(START, END, "euclidian"),
        # DStar(START, END, "euclidian", 5.0),
        # AdaptiveAStar(START, END, "euclidian"),
        DynamicGuidedSRrtEdge(START, END, 0.05, global_time=5),
        RrtEdge(START, END, 0.05, float("inf"), time=5),
        AdaptiveAStar(START, END, "euclidian", time=5),
        #IRrtStar(START, END, 10, 0.05, 5, iter_max=float("inf"), time=5),
        DStar(START, END, "euclidian", time=5),
    ]

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
                print(f"COMPLETED: {count}")
                index = futures[future]
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
    MAP_DIR = "src/Evaluation/Maps/2D/main/"
    OBJ_DIR = "src/Evaluation/Maps/2D/dynamic_obs.json"
    results = evaluate(MAP_DIR, OBJ_DIR)
    save_results(results, "5-seconds-2D-dynamic.json")


if __name__ == "__main__":
    main()
