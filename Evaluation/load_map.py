import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Sampling_based_Planning.rrt_2D.env import CustomEnv

def load(filepath):
    with open(filepath) as f:
        data = json.load(f)
        return data
    
def create_custom_env(filepath):
    
    d = load(filepath)
    return CustomEnv(d)
    

if __name__ == "__main__":
    d = create_custom_env("Evaluation/Maps/2D/uniform_random_fill_2D_10_perc/10_perc_2.json")

    
    