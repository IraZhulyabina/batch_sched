import os, time, subprocess, csv
import pandas as pd
import matplotlib.pyplot as plt

SOLVER_SCRIPT = 'batch_sched.py'
HEUR_SCRIPT = 'heuristic.py'
WINDOW_SCRIPT = 'hybrid_window.py'

DATA_DIR = 'data'
RESULTS_CSV = 'results.csv'

def measure(script, inst_path, label):
    # Здесь мы запускаем решение и измеряем время
    t0 = time.time()
    res = subprocess.run(
        ['python3', script, inst_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    elapsed = time.time() - t0

    # Ищем в самом конце вывода строку с makespan
    ms = None
    for line in reversed(res.stdout.splitlines()):
        if 'makespan' in line.lower():
            try:
                ms = float(line.split()[-1])
            except ValueError:
                pass
            break

    if ms is None:
        raise RuntimeError(f"Не удалось извлечь makespan из вывода ({label})")
    return elapsed, ms

def main():
    # Собираем данные по сгенерированных инстансам
    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class','instance','time_milp','ms_milp','time_heur','ms_heur','time_window','ms_window'])

        for filename in sorted(os.listdir(DATA_DIR)):
            cls = filename.split('_')[0]
            path = os.path.join(DATA_DIR, filename)

            print(f"=== {filename} ({cls}) ===")
            print(" MILP: ", end='', flush=True)
            t_m, ms_m = measure(SOLVER_SCRIPT, path, 'MILP')
            print(f"{t_m:.2f}s, makespan={ms_m:.2f}")

            print(" Heuristic: ", end='', flush=True)
            t_h, ms_h = measure(HEUR_SCRIPT, path, 'heuristic')
            print(f"{t_h:.2f}s, makespan={ms_h:.2f}")

            print(" Window: ", end='', flush=True)
            t_w, ms_w = measure(WINDOW_SCRIPT, path, 'window')
            print(f"{t_w:.2f}s, makespan={ms_w:.2f}")
            print()

            writer.writerow([
                cls, filename,
                f"{t_m:.2f}", f"{ms_m:.2f}",
                f"{t_h:.2f}", f"{ms_h:.2f}",
                f"{t_w:.2f}", f"{ms_w:.2f}"
            ])

    # Выведем средние результаты
    df = pd.read_csv(RESULTS_CSV)
    summary = df.groupby('class').agg({
        'time_milp': 'mean',
        'ms_milp': 'mean',
        'time_heur': 'mean',
        'ms_heur': 'mean',
        'time_window': 'mean',
        'ms_window': 'mean',
    }).round(2)
    print("Средние по классам:")
    print(summary, "\n")

if __name__ == '__main__':
    main()