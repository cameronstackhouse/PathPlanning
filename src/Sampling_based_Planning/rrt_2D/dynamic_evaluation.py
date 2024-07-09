import sys
import os
import tracemalloc

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

from glob import glob
from pathlib import Path


def evaluate(MAP_DIR: str, OBJ_DIR: str = None):
    START = (0, 0)
    END = (0, 0)
    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    algorithms = [
        DStar(START, END, "euclidian"),
        DynamicGuidedSRrtEdge(START, END, 0.05, global_time=5),
        RrtEdge(START, END, 0.05, 2000, time=5),
        IRrtStar(START, END, 5, 0.05, 5, 2000, time=5),
    ]

    results = []
    for algorithm in algorithms:
        print(algorithm)
        path_len = []
        compute_time = []
        traversal_time = []
        energy = []
        success = 0

        cpu_usage = []
        memory_used = []

        for map in map_name_list:
            print(map)
            algorithm.change_env(map)
            algorithm.dobs_dir = OBJ_DIR

            tracemalloc.start()

            path, avg_cpu_load = measure_cpu_usage(algorithm.run)

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            traversal_time.append(algorithm.total_time)
            compute_time.append(algorithm.compute_time)

            cpu_usage.append(avg_cpu_load)
            memory_used.append(peak)

            if path is not None:
                success += 1
                path_len.append(algorithms[1].utils.path_cost(path))
                energy.append(algorithms[1].utils.path_energy(path))
            else:
                path_len.append(None)
                energy.append(None)

        success /= NUM_MAPS
        result = {
            "Algorithm": algorithm.name,
            "Map Names": map_names,
            "Success Rate": success,
            "Path Length": path_len,
            "Initial Calculation Time": compute_time,
            "Traversal Time": traversal_time,
            "CPU Usage": cpu_usage,
            "Memory Used": memory_used,
            "Replan Time": [],
        }

        results.append(result)

    return results


def main():
    MAP_DIR = "src/Evaluation/Maps/2D/main/"
    OBJ_DIR = "src/Evaluation/Maps/2D/dynamic_block_map_25/0_obs.json"
    results = evaluate(MAP_DIR, OBJ_DIR)
    save_results(results, "dynamic_eval_2D_results.json")


if __name__ == "__main__":
    main()
