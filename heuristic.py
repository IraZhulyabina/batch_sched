import os, sys, json
from pyomo.environ import *
from pyomo.opt import TerminationCondition

def load_instance(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_and_solve_heuristic(inst_path, solver_name='cbc'):
    # Загрузим сгенерированные входные данные
    data = load_instance(inst_path)
    U_data = data['U_data']
    I_data = data['I_data']
    tau_data = data['tau_data']
    H = data['H']
    P = data['P']

    # Ограничим число слотов P_heu << P
    P_heu = min(P, 8)
    if P_heu > 1:
        grid = [round(H * k / (P_heu - 1)) for k in range(P_heu)]
    else:
        grid = [0]

    KI = [(u, i) for u in U_data for i in I_data[u]]

    # Запустим первую фазу эвристика - дискретную MILP на сжатой сетке
    # Зададим модель
    model1 = ConcreteModel()
    model1.K = Set(initialize=KI, dimen=2)
    model1.G = RangeSet(0, len(grid) - 1)
    model1.t = Param(model1.G, initialize=lambda m, g: grid[g])
    model1.tau = Param(model1.K, initialize={(u, i): tau_data[u][i] for u, i in KI})
    model1.x = Var(model1.K, model1.G, within=Binary)
    model1.M = Var(within=NonNegativeReals)
    model1.obj = Objective(expr=model1.M, sense=minimize)

    # Теперь зададим ограничения
    def one_start(m, u, i):
        return sum(m.x[(u, i), g] for g in m.G) == 1

    model1.one = Constraint(model1.K, rule=one_start)

    model1.no = ConstraintList()
    for u in U_data:
        for i1 in I_data[u]:
            for i2 in I_data[u]:
                if i1 >= i2:
                    continue
                d1 = tau_data[u][i1]
                d2 = tau_data[u][i2]
                for g1 in model1.G:
                    t1, e1 = grid[g1], grid[g1] + d1
                    for g2 in model1.G:
                        t2, e2 = grid[g2], grid[g2] + d2
                        if not (t2 >= e1 or t1 >= e2):
                            model1.no.add(
                                model1.x[(u, i1), g1] + model1.x[(u, i2), g2] <= 1
                            )

    def ms_bound(m, u, i):
        return m.M >= sum((m.t[g] + m.tau[(u, i)]) * m.x[(u, i), g] for g in m.G)

    model1.ms = Constraint(model1.K, rule=ms_bound)

    # Решаем
    solver1 = SolverFactory(solver_name)
    solver1.options['sec'] = 1
    solver1.options['ratio'] = 0.05
    solver1.options['threads'] = 4
    result1 = solver1.solve(model1, tee=False)

    # Проверяем, нашлось ли feasible/optimal
    tc1 = result1.solver.termination_condition
    if tc1 not in (TerminationCondition.optimal, TerminationCondition.feasible):
        # Если не нашлось, то запускаем жадный алгоритм вместо этого
        starts = {}
        for u in U_data:
            t = 0
            for i in I_data[u]:
                starts[(u, i)] = t
                t += tau_data[u][i]
    else:
        # Извлекаем старты
        starts = {}
        for u, i in KI:
            for g in model1.G:
                if value(model1.x[(u, i), g]) > 0.5:
                    starts[(u, i)] = grid[g]
                    break

    # Запускаем вторую фазу эвристики - left-shift
    # Определяем модель
    model2 = ConcreteModel()
    model2.K = Set(initialize=KI, dimen=2)
    model2.S = Var(model2.K, within=NonNegativeReals)
    model2.MS = Var(within=NonNegativeReals)
    model2.obj = Objective(expr=model2.MS, sense=minimize)

    # Задаем ограничения
    def lb_rule(m, u, i):
        return m.S[(u, i)] >= starts[(u, i)]

    model2.lb = Constraint(model2.K, rule=lb_rule)

    model2.chain = ConstraintList()
    for u in U_data:
        seq = sorted(I_data[u], key=lambda i: starts[(u, i)])
        for prev, nxt in zip(seq, seq[1:]):
            model2.chain.add(
                model2.S[(u, nxt)] >= model2.S[(u, prev)] + tau_data[u][prev]
            )

    def ms2_rule(m, u, i):
        return m.MS >= m.S[(u, i)] + tau_data[u][i]

    model2.ms2 = Constraint(model2.K, rule=ms2_rule)

    # Решаем
    solver2 = SolverFactory(solver_name)
    solver2.options['sec'] = 60
    solver2.options['threads'] = 8
    solver2.solve(model2, tee=False)

    print(f"makespan: {value(model2.MS):.2f}")

if __name__ == '__main__':
    inst_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'small_1.json')
    build_and_solve_heuristic(inst_path)
