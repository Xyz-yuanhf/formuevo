import re
import os
import json
import pickle
import numpy as np
import gurobipy as gp

# compute shifted geometric mean of runtime list
def shifted_geometric_mean(runtime_ls, shift=1.0):
    runtime_arr = np.array(runtime_ls)
    return np.exp(np.mean(np.log(runtime_arr + shift))) - shift

# check if is seed evaluation
def check_seed(stdout_file):
    try:
        match = re.search(r'(\d+).*?(\d+)', stdout_file)
        iter_id = int(match.group(1))
        resp_id = int(match.group(2))
        return iter_id == 0 and resp_id == 0
    except:
        return False

def _evaluation(instances, wls_config, mip_model_builder):

    with gp.Env(params=wls_config) as env:
        
        objval_ls, runtime_ls, optimal_ls = [], [], []
        solver_info = {
            "num_vars_ls": [],
            "num_constrs_ls": [],
            "root_bound_ls": [],
            "root_gap_ls": [],
            "node_count_ls": [],
            "pres_var_ls": [],
            "pres_constr_ls": [],
            "pres_bound_ls": [],
        }

        for instance in instances:

            if isinstance(instance, tuple):
                model = mip_model_builder(*instance, env=env)
            else:
                model = mip_model_builder(instance, env=env)
            
            model.setParam("Threads", 1)
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 60)

            model.update()
            solver_info["num_vars_ls"].append(model.NumVars)
            solver_info["num_constrs_ls"].append(model.NumConstrs)

            root_record, pres_record = False, False
            def _callback(model, where):
                nonlocal root_record, pres_record
                if where == gp.GRB.Callback.MIP:
                    if not root_record:
                        root_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                        solver_info["root_bound_ls"].append(root_bound)
                        root_record = True
                if where == gp.GRB.Callback.PRESOLVE:
                    if not pres_record:
                        solver_info["pres_constr_ls"].append(model.cbGet(gp.GRB.Callback.PRE_ROWDEL))
                        solver_info["pres_var_ls"].append(model.cbGet(gp.GRB.Callback.PRE_COLDEL))
                        solver_info["pres_bound_ls"].append(model.cbGet(gp.GRB.Callback.PRE_BNDCHG))
                        pres_record = True
                    else:
                        solver_info["pres_constr_ls"][-1] = model.cbGet(gp.GRB.Callback.PRE_ROWDEL)
                        solver_info["pres_var_ls"][-1] = model.cbGet(gp.GRB.Callback.PRE_COLDEL)
                        solver_info["pres_bound_ls"][-1] = model.cbGet(gp.GRB.Callback.PRE_BNDCHG)

            model.optimize(_callback)

            runtime_ls.append(model.RunTime)
            objval_ls.append(model.ObjVal)
            optimal_ls.append(model.Status == gp.GRB.OPTIMAL)

            solver_info["node_count_ls"].append(model.NodeCount)
            solver_info["root_gap_ls"].append(
                abs(solver_info["root_bound_ls"][-1] - model.ObjVal) / (abs(model.ObjVal) + 1e-8)
            )

        return runtime_ls, objval_ls, optimal_ls, solver_info

def evaluations(
    mode,
    stdout_path,
    problem,
    instance_params,
    instance_generator,
    mip_model_builder,
):

    # num of instances
    with open("configs/run_cfg.json", 'r') as f:
        run_cfg = json.load(f)
        num_train = run_cfg.get("num_train", 100)
        num_validate = run_cfg.get("num_val", 100)
    # special case
    if problem == "imo":
        num_train, num_validate = 1, 1

    # seed setting
    base_seed = 2026

    # Gurobi API
    with open("configs/llm_api.json", 'r') as f:
        llm_api = json.load(f)
        wls_config = llm_api["gurobi_api"]

    # related directory
    output_dir = os.path.dirname(stdout_path)
    is_init = check_seed(os.path.basename(stdout_path))
    instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), problem, "instances")
    os.makedirs(instance_dir, exist_ok=True)

    assert mode in ["train", "validate"]
    if mode == "train":
        print("[*] Run training...")
    elif mode == "validate":
        print("[*] Run validation...")

    # generate instances
    if is_init and mode == "train":
        instances = []
        for i in range(num_train + num_validate):
            seed = base_seed + i
            instance = instance_generator(seed=seed, **instance_params)
            instances.append(instance)
        # special case
        if problem == "imo":
            instances = instance_params["N"]
        # save instances
        with open(os.path.join(instance_dir, "train.pkl"), 'wb') as f:
            pickle.dump(instances[:num_train], f)
        with open(os.path.join(instance_dir, "validate.pkl"), 'wb') as f:
            pickle.dump(instances[num_train:], f)
    else:
        # read instances from files
        if mode == "train":
            with open(os.path.join(instance_dir, "train.pkl"), 'rb') as f:
                instances = pickle.load(f)
        elif mode == "validate":
            with open(os.path.join(instance_dir, "validate.pkl"), 'rb') as f:
                instances = pickle.load(f)

    # evaluate instances
    if not (is_init and mode == "validate"):
        runtime_ls, objval_ls, optimal_ls, solver_info = _evaluation(instances, wls_config, mip_model_builder)
    else:
        with open(os.path.join(output_dir, "baseline.json"), 'r') as f:
            baseline = json.load(f)
            runtime_ls = baseline[mode]["runtime"]
            objval_ls = baseline[mode]["objval"]
            optimal_ls = baseline[mode]["optimal"]
        solver_info = {
            "num_vars_ls": [],
            "num_constrs_ls": [],
            "root_bound_ls": [],
            "root_gap_ls": [],
            "node_count_ls": [],
            "pres_var_ls": [],
            "pres_constr_ls": [],
            "pres_bound_ls": [],
        }  # placeholder for solver info

    # save or read baseline results
    if is_init and mode == "train":
        baseline = {
            "train": {
                "runtime": runtime_ls[:num_train],
                "objval": objval_ls[:num_train],
                "optimal": optimal_ls[:num_train]},
            "validate": {
                "runtime": runtime_ls[num_train:],
                "objval": objval_ls[num_train:],
                "optimal": optimal_ls[num_train:]}
        }
        with open(os.path.join(output_dir, "baseline.json"), 'w') as f:
            json.dump(baseline, f, indent=4)
        runtime_ls = baseline[mode]["runtime"]
        objval_ls = baseline[mode]["objval"]
        optimal_ls = baseline[mode]["optimal"]
        # truncate solver info
        for key in solver_info:
            solver_info[key] = solver_info[key][:num_train]
    else:
        with open(os.path.join(output_dir, "baseline.json"), 'r') as f:
            baseline = json.load(f)

    # check objval equality in train mode
    if mode == "train":
        baseline_objval = [objval for idx, objval in enumerate(baseline[mode]["objval"]) if baseline[mode]["optimal"][idx]]
        current_objval = [objval for idx, objval in enumerate(objval_ls) if baseline[mode]["optimal"][idx]]
        for b_obj, c_obj in zip(baseline_objval, current_objval):
            if problem in ["jssp", "bpp", "imo"]:
                assert abs(b_obj - c_obj) < 0.5, "Wrong ObjVal! Incorrect formulation!"
            elif problem in ["cflp", "tsp", "clsp", "nnv", "qap"]:
                assert abs(b_obj - c_obj) / (abs(b_obj) + 1e-12) < 1e-3, "Wrong ObjVal! Incorrect formulation!"
            else:
                raise ValueError(f"Unknown problem {problem}")
    
    # output average runtime
    runtime_ls = [runtime for idx, runtime in enumerate(runtime_ls) if baseline[mode]["optimal"][idx]]
    print("[*] Solver Info:")
    print(json.dumps({
        "num_vars": np.mean(solver_info['num_vars_ls']),
        "num_constrs": np.mean(solver_info['num_constrs_ls']),
        "root_bound": np.mean(solver_info['root_bound_ls']),
        "root_gap": f"{np.mean(solver_info['root_gap_ls']) * 100:.4f}%",
        "node_count": np.mean(solver_info['node_count_ls']),
        "presolve_rows_removed": np.mean(solver_info['pres_var_ls']),
        "presolve_cols_removed": np.mean(solver_info['pres_constr_ls']),
        "presolve_bounds_changed": np.mean(solver_info['pres_bound_ls']),
    }))
    print("[*] SGM Runtime:")
    print(shifted_geometric_mean(runtime_ls))