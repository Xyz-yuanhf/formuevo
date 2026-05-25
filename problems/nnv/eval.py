import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluations
from gpt import build_mip_model

def instance_generator(layer_dims, seed=42):
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    weights = []
    biases = []

    for k in range(1, len(layer_dims)):
        dim_in = layer_dims[k - 1]
        dim_out = layer_dims[k]
        W = rng.normal(loc=0.0, scale=np.sqrt(2.0 / dim_in), size=(dim_out, dim_in))
        b = rng.normal(loc=0.0, scale=0.1, size=(dim_out,))
        weights.append(W)
        biases.append(b)

    n_in = layer_dims[0]
    x0 = rng.uniform(0.0, 1.0, size=(n_in,))

    z = x0
    L = len(weights)
    for k in range(L):
        W, b = weights[k], biases[k]
        z_hat = W @ z + b
        if k < L - 1:
            z = np.maximum(z_hat, 0)
        else:
            logits = z_hat

    true_label = int(np.argmax(logits))
    num_classes = layer_dims[-1]
    available_targets = [c for c in range(num_classes) if c != true_label]
    target_label = int(rng.choice(available_targets))

    return weights, biases, x0, true_label, target_label

def mip_model_builder(weights, biases, x0, true_label, target_label, env):
    return build_mip_model(weights, biases, x0, true_label, target_label, env=env)

if __name__ == "__main__":

    problem = "nnv"
    stdout_path, mode = sys.argv[1], sys.argv[2]
    instance_params = {
        "layer_dims": [20, 80, 10],
    }

    evaluations(
        mode=mode,
        stdout_path=stdout_path,
        problem=problem,
        instance_params=instance_params,
        instance_generator=instance_generator,
        mip_model_builder=mip_model_builder
    )