import os, sys, json
from pyomo.environ import *

def load_instance(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_and_solve(inst_path, solver_name='cbc'):
    # Загрузим сгенерированные входные данные
    data = load_instance(inst_path)
    U_data = data["U_data"]
    I_data = data["I_data"]
    P = data["P"]
    H = data["H"]
    tau_data = data["tau_data"]
    B_min = data["B_min"]
    B_max = data["B_max"]

    # Изменим формат полученных данных, чтобы их было проще использовать в pyomo
    KI = [(u,i) for u in U_data for i in I_data[u]]
    tau_flat = {(u,i): tau_data[u][i] for u,i in KI}
    Bmin_flat = {(u,i): B_min[u][i] for u,i in KI}
    Bmax_flat = {(u,i): B_max[u][i] for u,i in KI}
    max_tau = max(tau_flat.values())

    # Зададим модель
    model = ConcreteModel()
    model.U = Set(initialize=U_data)
    model.K = Set(initialize=KI, dimen=2)
    model.P = RangeSet(1, P)

    model.tau = Param(model.K, initialize=tau_flat)
    model.Bmin = Param(model.K, initialize=Bmin_flat)
    model.Bmax = Param(model.K, initialize=Bmax_flat)
    model.M = Param(initialize=H + max_tau)

    model.z = Var(model.K, model.P, within=Binary)
    model.S = Var(model.K, model.P, within=NonNegativeReals)
    model.F = Var(model.K, model.P, within=NonNegativeReals)
    model.Q = Var(model.K, model.P, within=NonNegativeReals)
    model.MS = Var(within=NonNegativeReals)

    model.obj = Objective(expr=model.MS, sense=minimize)

    # Зададим теперь ограничения, которые мы описывали в модели
    def assign_rule(m, u, i):
        return sum(m.z[(u,i),p] for p in m.P) == 1
    model.assign_constr = Constraint(model.K, rule=assign_rule)

    def machine_rule(m, u, p):
        return sum(m.z[(u,i),p] for i in I_data[u]) <= 1
    model.machine_constr = Constraint(model.U, model.P, rule=machine_rule)

    def size_lower_rule(m, u, i, p):
        return m.Q[(u, i), p] >= m.Bmin[(u, i)] * m.z[(u, i), p]

    model.size_lower = Constraint(model.K, model.P, rule=size_lower_rule)

    def size_upper_rule(m, u, i, p):
        return m.Q[(u, i), p] <= m.Bmax[(u, i)] * m.z[(u, i), p]

    model.size_upper = Constraint(model.K, model.P, rule=size_upper_rule)

    def time_rule(m, u, i, p):
        return m.F[(u,i),p] >= m.S[(u,i),p] + m.tau[(u,i)] - m.M*(1-m.z[(u,i),p])
    model.time_constr = Constraint(model.K, model.P, rule=time_rule)

    def start_rule(m, u, i, p):
        return m.S[(u,i),p] <= m.M * m.z[(u,i),p]
    model.start_constr = Constraint(model.K, model.P, rule=start_rule)

    def makespan_rule(m, u, i, p):
        return m.MS >= m.F[(u,i),p] - m.M*(1-m.z[(u,i),p])
    model.ms_constr = Constraint(model.K, model.P, rule=makespan_rule)

    # Теперь настроим решатель
    solver = SolverFactory(solver_name)
    solver.options['sec'] = 3600
    solver.options['ratio'] = 0.0001
    solver.options['threads'] = 8

    # Решим задачу
    solver.solve(model, tee=False)
    print(f"makespan: {value(model.MS):.2f}")

if __name__=='__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data','large_1.json')
    build_and_solve(inst)
