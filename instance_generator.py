import os
import json
import random


def generate_instance(U, I, P, H, out_path):
    # Вводим узлы и операции
    U_data = [f"u{u + 1}" for u in range(U)]
    I_data = {u: [f"i{u}_{j + 1}" for j in range(I)] for u in U_data}

    # Добавляем вариативные времена обработки
    tau_data = {}
    for u in U_data:
        tau_data[u] = {}
        for i in I_data[u]:
            if random.random() < 0.2:
                # 20% задач длятся долго
                tau_data[u][i] = random.randint(80, 150)
            else:
                tau_data[u][i] = random.randint(5, 50)

    # Параметры партии
    B_min = {u: {i: 1 for i in I_data[u]} for u in U_data}
    B_max = {u: {i: 10 for i in I_data[u]} for u in U_data}
    C_min = {u: 1 for u in U_data}
    C_max = {u: 20 for u in U_data}

    # Генерируем технологические зависимости
    tasks = [(u, i) for u in U_data for i in I_data[u]]
    precedences = []
    num_prec = int(len(tasks) * 0.3)
    for _ in range(num_prec):
        (u1, i1), (u2, i2) = random.sample(tasks, 2)
        if (u1, i1) != (u2, i2):
            precedences.append({
                "before": [u1, i1],
                "after": [u2, i2]
            })

    instance = {
        "U_data": U_data,
        "I_data": I_data,
        "P": P,
        "H": H,
        "tau_data": tau_data,
        "B_min": B_min,
        "B_max": B_max,
        "C_min": C_min,
        "C_max": C_max,
        "alpha_in": {},
        "alpha_out": {},
        "d_data": {},
        "e_data": {},
        "Pj_max": {},
        "p_init": {},
        "precedences": precedences
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(instance, f, indent=2, ensure_ascii=False)


def main():
    random.seed(42)
    sizes = {
        "small": (5, 5, 10, 200),
        "medium": (10, 10, 15, 200),
        "large": (15, 10, 20, 200)
    }
    for size, (U, I, P, H) in sizes.items():
        for idx in range(1, 6):
            fn = os.path.join("data", f"{size}_{idx}.json")
            generate_instance(U, I, P, H, fn)
            print(f"Created {fn}")


if __name__ == "__main__":
    main()