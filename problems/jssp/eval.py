import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluations
from gpt import build_mip_model

def instance_generator(num_jobs, num_machines, min_time=1, max_time=100, seed=42):
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    machine_assign = np.zeros((num_jobs, num_machines), dtype=int)
    for j in range(num_jobs):
        machine_assign[j] = rng.permutation(num_machines)
    process_time = rng.integers(min_time, max_time + 1, size=(num_jobs, num_machines))
    
    return machine_assign, process_time

def mip_model_builder(machine_assign, process_time, env):
    num_jobs = machine_assign.shape[0]
    num_machines = machine_assign.shape[1]
    return build_mip_model(num_jobs, num_machines, machine_assign, process_time, env=env)

if __name__ == "__main__":

    problem = "jssp"
    stdout_path, mode = sys.argv[1], sys.argv[2]
    instance_params = {
        "num_jobs": 10,
        "num_machines": 10
    }    
    
    evaluations(
        mode=mode,
        stdout_path=stdout_path,
        problem=problem,
        instance_params=instance_params,
        instance_generator=instance_generator,
        mip_model_builder=mip_model_builder
    )
