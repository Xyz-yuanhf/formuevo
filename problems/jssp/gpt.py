import scipy as sp
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def build_mip_model(num_jobs: int, num_machines: int, machine_assign: np.ndarray, process_time: np.ndarray, env: gp.Env):
    """Build an improved MIP model for Job-Shop Scheduling Problem (JSSP).

    Key ideas (complementary crossover):
      1) Use Parent B's compact disjunctive ordering MILP with precedence and makespan.
      2) Replace the remaining weak/loose relaxation with deterministic, callback-free
         strengthening via ES/LS-based propagation to add implied machine-order constraints
         (a static "edge-fixing" / precedence propagation, valid for any feasible schedule).

    Returns a Gurobi model (not optimized)."""

    J = int(num_jobs)
    M = int(num_machines)

    # Precompute processing time p[i,k] and the operation position for each (job i, machine k)
    job_op_on_machine = -np.ones((J, M), dtype=int)
    p = np.zeros((J, M), dtype=float)
    for i in range(J):
        for t in range(M):
            k = int(machine_assign[i, t])
            job_op_on_machine[i, k] = t
            p[i, k] = float(process_time[i, t])
    if np.any(job_op_on_machine < 0):
        raise ValueError("machine_assign must specify exactly one operation per machine for each job.")

    total_p = float(np.sum(process_time))
    # A safe horizon upper bound on completion times.
    horizon = total_p

    model = gp.Model("jssp_strong_propagation", env=env)

    # ---- Compute ES/LS per operation along the fixed technological route (Parent B style) ----
    ES_op = np.zeros((J, M), dtype=float)
    LS_op = np.zeros((J, M), dtype=float)

    for i in range(J):
        cum = 0.0
        for t in range(M):
            ES_op[i, t] = cum
            k = int(machine_assign[i, t])
            cum += p[i, k]
        cum = 0.0
        for t in range(M - 1, -1, -1):
            LS_op[i, t] = horizon - cum
            k = int(machine_assign[i, t])
            cum += p[i, k]

    ES = np.zeros((J, M), dtype=float)
    LS = np.zeros((J, M), dtype=float)
    for i in range(J):
        for k in range(M):
            t = int(job_op_on_machine[i, k])
            ES[i, k] = ES_op[i, t]
            LS[i, k] = LS_op[i, t]

    ES = np.maximum(ES, 0.0)
    LS = np.minimum(LS, horizon)

    # ---- Variables ----
    # Start times per (job, machine)
    s = {}
    for i in range(J):
        for k in range(M):
            lb = float(ES[i, k])
            ub = float(LS[i, k])
            if ub < lb - 1e-9:
                raise ValueError("Inconsistent ES/LS bounds. Check horizon or input data.")
            s[(i, k)] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"s_{i}_{k}")

    C_max = model.addVar(lb=0.0, ub=horizon, vtype=GRB.CONTINUOUS, name="C_max")

    # Disjunctive ordering binaries only for i1<i2 (compact)
    y = {}
    for k in range(M):
        for i1 in range(J):
            for i2 in range(i1 + 1, J):
                y[(i1, i2, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i1}_{i2}_{k}")

    model.setObjective(C_max, GRB.MINIMIZE)

    # ---- Precedence constraints along each job route ----
    for i in range(J):
        for t in range(M - 1):
            k1 = int(machine_assign[i, t])
            k2 = int(machine_assign[i, t + 1])
            model.addConstr(s[(i, k2)] >= s[(i, k1)] + p[i, k1], name=f"prec_{i}_{t}")

    # ---- Makespan constraints ----
    for i in range(J):
        for k in range(M):
            model.addConstr(C_max >= s[(i, k)] + p[i, k], name=f"mk_{i}_{k}")

    # ---- Machine non-overlap with tighter pair-specific Big-M (Parent B style) ----
    # For each machine k and pair i1<i2:
    #   y=1 => i1 before i2
    #   y=0 => i2 before i1
    for k in range(M):
        for i1 in range(J):
            p1 = p[i1, k]
            for i2 in range(i1 + 1, J):
                p2 = p[i2, k]
                yvar = y[(i1, i2, k)]

                # i1 before i2 active when yvar=1
                # s[i1]+p1 <= s[i2] + M12*(1-y)
                M12 = float(LS[i1, k] + p1 - ES[i2, k])
                if M12 < 0.0:
                    M12 = 0.0
                model.addConstr(
                    s[(i1, k)] + p1 <= s[(i2, k)] + M12 * (1 - yvar),
                    name=f"mach_fwd_{k}_{i1}_{i2}"
                )

                # i2 before i1 active when yvar=0
                # s[i2]+p2 <= s[i1] + M21*y
                M21 = float(LS[i2, k] + p2 - ES[i1, k])
                if M21 < 0.0:
                    M21 = 0.0
                model.addConstr(
                    s[(i2, k)] + p2 <= s[(i1, k)] + M21 * (yvar),
                    name=f"mach_bwd_{k}_{i1}_{i2}"
                )

    # ---- Complementary strengthening: ES/LS-based edge fixing (no callbacks) ----
    # If earliest of i1 is so late that i2 must precede i1 on machine k, we fix y accordingly.
    # For i1<i2 on machine k:
    #   If ES[i2,k] + p[i2,k] > LS[i1,k] then i2 cannot be after i1 -> must be i2 before i1 -> y=0.
    #   If ES[i1,k] + p[i1,k] > LS[i2,k] then i1 cannot be after i2 -> must be i1 before i2 -> y=1.
    # This is valid because any feasible start times satisfy ES<=s<=LS.
    for k in range(M):
        for i1 in range(J):
            for i2 in range(i1 + 1, J):
                yvar = y[(i1, i2, k)]
                # i1<i2, y=1 means i1 before i2
                # Check necessity of i1 before i2
                if ES[i1, k] + p[i1, k] > LS[i2, k] + 1e-9:
                    model.addConstr(yvar == 1, name=f"edgefix_{k}_{i1}_{i2}_must_fwd")
                # Check necessity of i2 before i1
                if ES[i2, k] + p[i2, k] > LS[i1, k] + 1e-9:
                    model.addConstr(yvar == 0, name=f"edgefix_{k}_{i1}_{i2}_must_bwd")

    # ---- Return model (no optimize) ----
    return model
