import os
import sys
import json
from pyomo.environ import *
from pyomo.opt import TerminationCondition

def load_instance(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_and_solve_hybrid_window(inst_path, w=1):
    # Загружаем данне
    data = load_instance(inst_path)
    U_data = data["U_data"]
    I_data = data["I_data"]
    tau = data["tau_data"]
    Bmin = data["B_min"]
    Bmax = data["B_max"]
    H = data["H"]
    P = data["P"]

    # Составим жадный план, когда задачи последовательно исполненяются на каждом узле
    rough_start = {}
    for u in U_data:
        t = 0.0
        for i in I_data[u]:
            rough_start[(u, i)] = t
            t += tau[u][i]

    # Для каждого узла распределим центры событий равномерно
    W = {}
    for u in U_data:
        tasks_u = sorted(I_data[u], key=lambda i: rough_start[(u,i)])
        n_u = len(tasks_u)
        for idx, i in enumerate(tasks_u):
            if n_u == 1:
                p_center = 1
            else:
                p_center = 1 + round(idx * (P - 1) / (n_u - 1))
            low  = max(1, p_center - w)
            high = min(P, p_center + w)
            W[(u, i)] = list(range(low, high+1))

    # Зададим модель
    model = ConcreteModel()
    model.U = Set(initialize=U_data)
    model.K = Set(initialize=[(u,i) for u in U_data for i in I_data[u]], dimen=2)
    model.P = RangeSet(1, P)

    model.tau  = Param(model.K, initialize={(u,i): tau[u][i] for u,i in model.K})
    model.Bmin = Param(model.K, initialize={(u,i): Bmin[u][i] for u,i in model.K})
    model.Bmax = Param(model.K, initialize={(u,i): Bmax[u][i] for u,i in model.K})
    max_tau    = max(tau[u][i] for u,i in model.K)
    model.Mbig = Param(initialize=H + max_tau)

    model.z  = Var(model.K, model.P, within=Binary)
    model.S  = Var(model.K, model.P, within=NonNegativeReals)
    model.F  = Var(model.K, model.P, within=NonNegativeReals)
    model.Q  = Var(model.K, model.P, within=NonNegativeReals)
    model.MS = Var(within=NonNegativeReals)

    model.obj = Objective(expr=model.MS, sense=minimize)

    # Зададим ограничения
    def window_rule(m, u, i, p):
        if p not in W[(u, i)]:
            return m.z[(u, i), p] == 0
        else:
            return Constraint.Skip
    model.window = Constraint(model.K, model.P, rule=window_rule)

    def assign_rule(m, u, i):
        return sum(m.z[(u, i), p] for p in m.P) == 1
    model.assign = Constraint(model.K, rule=assign_rule)

    def machine_rule(m, u, p):
        return sum(m.z[(u, i), p] for i in I_data[u]) <= 1
    model.machine = Constraint(model.U, model.P, rule=machine_rule)

    def size_lower(m, u, i, p):
        return m.Q[(u, i), p] >= m.Bmin[(u, i)] * m.z[(u, i), p]
    model.size_lower = Constraint(model.K, model.P, rule=size_lower)

    def size_upper(m, u, i, p):
        return m.Q[(u, i), p] <= m.Bmax[(u, i)] * m.z[(u, i), p]
    model.size_upper = Constraint(model.K, model.P, rule=size_upper)

    def time_constr(m, u, i, p):
        return m.F[(u, i), p] >= m.S[(u, i), p] + m.tau[(u, i)] - m.Mbig * (1 - m.z[(u, i), p])
    model.time_constr = Constraint(model.K, model.P, rule=time_constr)

    def start_constr(m, u, i, p):
        return m.S[(u, i), p] <= m.Mbig * m.z[(u, i),p]
    model.start_constr = Constraint(model.K, model.P, rule=start_constr)

    def makespan_constr(m, u, i, p):
        return m.MS >= m.F[(u, i), p] - m.Mbig * (1 - m.z[(u, i), p])
    model.makespan_constr = Constraint(model.K, model.P, rule=makespan_constr)

    # MIP‐start: сразу пометим центральное событие каждой задачи
    for (u,i), window in W.items():
        p0 = window[len(window) // 2]
        model.z[(u, i), p0].value = 1

    # Настроим решатель
    solver = SolverFactory('cbc')
    solver.options['sec'] = 60
    solver.options['ratio'] = 1e-2
    solver.options['threads'] = 8

    res = solver.solve(model, tee=False)
    status = res.solver.termination_condition
    if status in (TerminationCondition.optimal, TerminationCondition.feasible):
        print(f"найдено feasible при w={w}, makespan = {value(model.MS):.2f}")
    else:
        print(f"Статус решателя: {status}")

if __name__ == '__main__':
    inst   = sys.argv[1] if len(sys.argv)>1 else 'data/small_1.json'
    init_w = int(sys.argv[2]) if len(sys.argv)>2 else 1
    build_and_solve_hybrid_window(inst, w=init_w)
