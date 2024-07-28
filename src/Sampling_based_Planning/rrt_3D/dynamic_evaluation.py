import json
import sys
import os
from threading import Thread
import tracemalloc

import numpy as np
import psutil

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/"
)

from rrt_3D.dynamic_guided_srrt_edge3D import DynamicGuidedSrrtEdge
from rrt_3D.adapted_IRRT_star import AnytimeIRRTTStar
from rrt_3D.rrt_edge3D import RrtEdge
from rrt_3D.utils3D import getDist

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_3D.Adaptive_AStar import AdaptiveAStar

from glob import glob
from pathlib import Path


def save_results(results, name):
    with open(name, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {name}")


def path_cost(path):
    cost = 0
    for i in range(len(path) - 2):
        current = path[i]
        next = path[i + 1]

        dist = getDist(current, next)
        cost += dist

    return cost


def calculate_turn_angle(p_1, p_2, p_3):
    pass


def path_energy(path):
    # TODO in 2D utils.py

    x = np.array([0, 2, 4, 6])
    # Power required (W) for each at 0, 2, 4, 6 m/s
    P_acc = np.array([242, 235, 239, 249])
    P_dec = np.array([245, 232, 230, 239])
    P_v = [242, 245, 246, 268]
    TURN_POWER = 260
    TURN_SPEED = 2.07

    cubic_coeffs_acc = np.polyfit(x, P_acc, 3)
    cubic_poly_acc = np.poly1d(cubic_coeffs_acc)

    cubic_coeffs_dec = np.polyfit(x, P_dec, 3)
    cubic_poly_dec = np.poly1d(cubic_coeffs_dec)

    def integrand_acc(x):
        return cubic_poly_acc(x)

    def integrand_dec(x):
        return cubic_poly_dec(x)

    fixed_speed = 6  # 6 m/s
    total_energy = 0

    for i in range(1, len(path)):
        p_1 = path[i - 1]
        p_2 = path[i]

        distance = getDist(p_1, p_2)
        time_uniform = distance / fixed_speed


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


def evaluate(MAP_DIR: str, OBJ_DIR: str = None):
    START = (0, 0)
    END = (0, 0)

    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    algorithms = [
        AdaptiveAStar(time=5),
        DynamicGuidedSrrtEdge(t=5),
        AnytimeIRRTTStar(time=5),
        RrtEdge(time=5),
    ]

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
    main()
