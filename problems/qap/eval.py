import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluations
from gpt import build_mip_model

def instance_generator(num_facilities, flow_range=(1, 10), seed=42):
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    flow_matrix = rng.integers(*flow_range, size=(num_facilities, num_facilities))
    flow_matrix = np.triu(flow_matrix) + np.triu(flow_matrix).T
    np.fill_diagonal(flow_matrix, 0)

    facility_coords = rng.random(size=(num_facilities, 2))
    distance_matrix = np.round(
        np.linalg.norm(
            facility_coords[:, np.newaxis, :] - facility_coords[np.newaxis, :, :], axis=-1
            ) * 10
        )

    return num_facilities, flow_matrix, distance_matrix

def mip_model_builder(num_facilities, flow_matrix, distance_matrix, env):
    return build_mip_model(num_facilities, flow_matrix, distance_matrix, env=env)

if __name__ == "__main__":

    problem = "qap"
    stdout_path, mode = sys.argv[1], sys.argv[2]
    instance_params = {
        "num_facilities": 7,
    }

    evaluations(
        mode=mode,
        stdout_path=stdout_path,
        problem=problem,
        instance_params=instance_params,
        instance_generator=instance_generator,
        mip_model_builder=mip_model_builder
    )