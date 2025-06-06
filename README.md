# Batch Scheduling MILP
Репозиторий содержит реализацию гибридной MILP–модели составления расписаний для batch-производства по статье Blömer & Günther, а также несколько эвристик и ускоренных версий.

## Зависимости
- Python 3.8+  
- [Pyomo](http://www.pyomo.org/)  
- CBC  
- (опционально) pandas, matplotlib, если хотите анализировать `results.csv` и строить графики

## Запуск
Сначала нужно запустить код instance_generator.py, чтобы были сгенерированы инстансы.
Затем тестировочная логика собрана в скрипте run.py. Он:
- читает тестовые инстансы;
- запускает на каждом инстансе batch_sched.py (точная MILP–модель), heuristic.py (двухэтапная LP-эвристика) и hybrid_window.py (гибридная модель с локальными окнами).
- собирает время решения и значение makespan в results.csv
- выводит средние значения по классам.
