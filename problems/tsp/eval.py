import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluations
from gpt import build_mip_model

def instance_generator(num_cities, seed=42):
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    city_coords = rng.random(size=(num_cities, 2))
    distance_matrix = np.round(
        np.linalg.norm(
            city_coords[:, np.newaxis, :] - city_coords[np.newaxis, :, :], axis=-1
            ) * 10
        )

    return distance_matrix

def mip_model_builder(distance_matrix, env):
    num_cities = len(distance_matrix)
    return build_mip_model(num_cities, distance_matrix, env=env)

if __name__ == "__main__":

    problem = "tsp"
    stdout_path, mode = sys.argv[1], sys.argv[2]
    instance_params = {
        "num_cities": 30
    }

    evaluations(
        mode=mode,
        stdout_path=stdout_path,
        problem=problem,
        instance_params=instance_params,
        instance_generator=instance_generator,
        mip_model_builder=mip_model_builder
    )