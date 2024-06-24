import json

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Sampling_based_Planning.rrt_2D.env import CustomEnv
# from Sampling_based_Planning.rrt_2D.dynamic_guided_srrt_edge import DynamicObj


def load(filepath):
    with open(filepath) as f:
        data = json.load(f)
        return data


def create_custom_env(filepath):
    d = load(filepath)
    return CustomEnv(d)


def create_custom_dynamic_env(filepath, obs_path):
    """
    TODO
    """
    map = load(filepath)
    objects = load(filepath)

    env = CustomEnv(map)

    dynamic_obs = []

    for json_obj in objects:
        obj = DynamicObj()
        obj.velocity = json_obj["velocity"]
        obj.size = json_obj["size"]
        obj.init_pos = json_obj["position"]

        dynamic_obs.append(obj)

        env.add_rect(obj)

    return env, dynamic_obs


if __name__ == "__main__":
    d = create_custom_env(
        "Evaluation/Maps/2D/uniform_random_fill_2D_10_perc/10_perc_2.json"
    )
