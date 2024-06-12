import json
import os
import sys

from ..Sampling_based_Planning.rrt_2D.mb_guided_srrt_edge import MBGuidedSRrtEdge


from glob import glob
from pathlib import Path

def comapare(algo_1, algo_2):
    pass

def gen_report(algo):
    MAP_DIR = "Evaluation/Maps/2D"
    map_name_list = list(Path(MAP_DIR).glob("*.json"))

    # TODO metrics

    # Load and evaluate each map
    for map in map_name_list:
        with open(map, 'r') as f:
            map_data = json.load(f)
            alg = MBGuidedSRrtEdge(map_data["agent"], map_data["goal"], 0.05)

def main():
    return 0

if __name__ == "__main__":
    gen_report(1)
