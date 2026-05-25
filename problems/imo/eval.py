import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluations
from gpt import build_mip_model

def instance_generator(N, seed=42):
    return N

def mip_model_builder(N, env):
    return build_mip_model(N, env=env)

if __name__ == "__main__":

    problem = "imo"
    stdout_path, mode = sys.argv[1], sys.argv[2]
    instance_params = {
        "N": [4, 5, 6, 7, 8, 9],
    }

    evaluations(
        mode=mode,
        stdout_path=stdout_path,
        problem=problem,
        instance_params=instance_params,
        instance_generator=instance_generator,
        mip_model_builder=mip_model_builder
    )