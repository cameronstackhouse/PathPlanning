import json
import sys
import os
import tracemalloc

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_3D.dynamic_guided_srrt_edge3D import DynamicGuidedSrrtEdge
from rrt_3D.informed_rrt_star3D import IRRT
from rrt_3D.rrt_edge3D import RrtEdge

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_3D.Adaptive_AStar import AdaptiveAStar

from glob import glob
from pathlib import Path


def evaluate(MAP_DIR: str, OBJ_DIR: str = None):
    START = (0, 0)
    END = (0, 0)

    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    algorithms = [AdaptiveAStar(), DynamicGuidedSrrtEdge(t=5), IRRT(), RrtEdge()]

    results = []
    for algorithm in algorithms:
        print(algorithm)
        algorithm.speed = 6
        path_len = []
        compute_time = []
        traversal_time = []
        energy = []
        replan_time = []
        success = 0
        traversed_path = []

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

            replan_time.append(algorithm.replan_time)

            if path is not None:
                success += 1
                path_len.append(path_cost(path))
                energy.append(path_energy(path))
                traversed_path.append(path)
            else:
                path_len.append(None)
                energy.append(None)
                traversed_path.append(None)

        success /= NUM_MAPS
        result = {
            "Algorithm": algorithm.name,
            "Map Names": map_names,
            "Path": traversed_path,
            "Success Rate": success,
            "Path Length": path_len,
            "Initial Calculation Time": compute_time,
            "Traversal Time": traversal_time,
            "CPU Usage": cpu_usage,
            "Memory Used": memory_used,
            "Energy To Traverse": energy,
            "Replan Time": replan_time,
        }

        results.append(result)
    return results


def main():
    MAP_DIR = "src/Evaluation/Maps/3D/main/"
    OBJ_DIR = "src/Evaluation/Maps/3D/obs.json"

    results = evaluate(MAP_DIR, OBJ_DIR)
    save_results(results, "3D Dynamic 5 Seconds")


if __name__ == "__main__":
    pass
