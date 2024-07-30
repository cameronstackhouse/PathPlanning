import json
import math
import sys
import os
from threading import Thread
from scipy.integrate import quad
import tracemalloc

import numpy as np
import psutil


from dynamic_guided_srrt_edge3D import DynamicGuidedSrrtEdge
from adapted_IRRT_star import AnytimeIRRTTStar
from rrt_edge3D import RrtEdge
from utils3D import getDist

sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/"
)

from Search_3D.Adaptive_AStar import AdaptiveAStar
from Search_3D.DstarLite3D import D_star_Lite

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


def save_results(results, name):
    with open(name, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

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
    x1, y1, z1 = p_1
    x2, y2, z2 = p_2
    x3, y3, z3 = p_3

    A = (x2 - x1, y2 - y1, z2 - z1)
    B = (x3 - x2, y3 - y2, z3 - z2)

    dot_product = (A[0] * B[0]) + (A[1] * B[1]) + (A[2] * B[2])

    magnitude_a = math.sqrt(A[0] ** 2 + A[1] ** 2 + A[2] ** 2)
    magnitude_b = math.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)

    if magnitude_a * magnitude_b == 0:
        return 0

    cos_theta = dot_product / (magnitude_a * magnitude_b)

    cos_theta = max(-1, min(1, cos_theta))

    theta_radians = math.acos(cos_theta)

    theta_degrees = math.degrees(theta_radians)

    return theta_degrees


def turn_energy(turn_power, degrees, speed):
    """
    Calculates the energy consumption of turning given
    the power required to turn per second, the number of degrees to turn,
    and the speed of turning of the UAV.
    """
    return turn_power * (degrees / speed)


def path_energy(path):
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

        energy_acc, _ = quad(integrand_acc, 0, fixed_speed)
        energy_unform, _ = quad(integrand_dec, 0, fixed_speed)
        energy_dec = time_uniform * P_v[-1]

        total_energy += energy_acc + energy_unform + energy_dec

        if i < len(path) - 1:
            p_3 = path[i + 1]
            angle = calculate_turn_angle(p_1, p_2, p_3)
            total_energy += turn_energy(TURN_POWER, angle, TURN_SPEED)

    return total_energy


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


def evaluate(MAP_DIR: str, OBJ_DIR: str = None, HOUSE_OBJ_DIR: str = None):
    START = (0, 0)
    END = (0, 0)

    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    map_names = [map.stem for map in map_name_list]
    NUM_MAPS = len(map_name_list)

    algorithms = [
        # D_star_Lite(),
        D_star_Lite(time=5),
        # DynamicGuidedSrrtEdge(t=5),
        # DynamicGuidedSrrtEdge(t=1),
        # RrtEdge(time=1),
        # AnytimeIRRTTStar(time=1),
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

            if str(map).startswith("src/Evaluation/Maps/3D/main/house_"):
                algorithm.change_env(map_name=map, obs_name=HOUSE_OBJ_DIR, size=28)
            else:
                algorithm.change_env(map, OBJ_DIR)

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
    OBJ_DIR = "src/Evaluation/Maps/3D/block_obs.json"
    HOUSE_OBJ_DIR = "src/Evaluation/Maps/3D/house_obs.json"

    results = evaluate(MAP_DIR, OBJ_DIR, HOUSE_OBJ_DIR)
    save_results(results, "DStar.json")


if __name__ == "__main__":
    main()
