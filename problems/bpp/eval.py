import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluations
from gpt import build_mip_model

def instance_generator(num_items, bin_capacity, min_weight, max_weight, seed=42):
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    item_weights = rng.integers(int(min_weight), int(max_weight) + 1, size=num_items)

    return item_weights, bin_capacity

def mip_model_builder(item_weights, bin_capacity, env):
    num_items = len(item_weights)
    return build_mip_model(num_items, item_weights, bin_capacity, env=env)

if __name__ == "__main__":

    problem = "bpp"
    stdout_path, mode = sys.argv[1], sys.argv[2]
    instance_params = {
        "num_items": 100,
        "bin_capacity": 150,
        "min_weight": 20,
        "max_weight": 100,
    }

    evaluations(
        mode=mode,
        stdout_path=stdout_path,
        problem=problem,
        instance_params=instance_params,
        instance_generator=instance_generator,
        mip_model_builder=mip_model_builder
    )