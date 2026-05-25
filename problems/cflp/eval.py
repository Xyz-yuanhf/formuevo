import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluations
from gpt import build_mip_model

def instance_generator(
    num_facilities,
    num_customers,
    demand_range=(5, 35),
    capacity_range=(10, 160),
    seed=42
    ):

    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    customer_coords = rng.random(size=(num_customers, 2))
    facility_coords = rng.random(size=(num_facilities, 2))
    transport_costs = np.round(
        np.linalg.norm(
            facility_coords[:, np.newaxis, :] - customer_coords[np.newaxis, :, :], axis=2
            ) * 10
        ).astype(int)
    demands = rng.integers(*demand_range, size=num_customers)
    capacities = rng.integers(*capacity_range, size=num_facilities)
    fixed_costs = np.round(
        rng.integers(0, 90, size=num_facilities) + \
        rng.integers(100, 110, size=num_facilities) * np.sqrt(capacities)
        ).astype(int)

    return fixed_costs, capacities, demands, transport_costs

def mip_model_builder(fixed_costs, capacities, demands, transport_costs, env):
    num_facilities, num_customers = len(fixed_costs), len(demands)
    return build_mip_model(
        num_facilities,
        num_customers,
        fixed_costs,
        capacities,
        demands,
        transport_costs,
        env=env
)

if __name__ == "__main__":

    problem = "cflp"
    stdout_path, mode = sys.argv[1], sys.argv[2]
    instance_params = {
        "num_facilities": 100,
        "num_customers": 200,
    }

    evaluations(
        mode=mode,
        stdout_path=stdout_path,
        problem=problem,
        instance_params=instance_params,
        instance_generator=instance_generator,
        mip_model_builder=mip_model_builder
    )