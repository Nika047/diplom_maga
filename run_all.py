"""
Все этапы подряд для выбранного варианта данных (решение пользователя 23.10.2026: на варианте
merged пересчитать все этапы).

  .venv\\Scripts\\python.exe run_all.py --data merged

Этапы идут в порядке зависимостей. Этап пропускается, если его сводка уже есть (повторный
запуск продолжает с первого незавершённого; --force - запустить всё заново). Сводка с FAIL
считается завершённой: FAIL - это результат, его разбирают по сводке. Если после этапа сводки
нет (необработанное исключение), запуск останавливается: следующие этапы от него зависят.
Подбор этапа 4 и 4г сам продолжает прерванные попытки (cache/.../optuna.db).

Журнал каждого этапа - как обычно в log/ (log_merged/ для merged).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# (команда, сводка относительно папки результатов)
STAGES = [
    ("run_stage0.py", "stage0/summary.txt"),
    ("run_stage1.py", "stage1/summary.txt"),
    ("run_stage2.py", "stage2/summary.txt"),
    ("run_stage3.py", "stage3/summary.txt"),
    ("run_stage3_window.py", "stage3_window/summary.txt"),
    ("run_stage4.py --phase all", "stage4/summary_all.txt"),
    ("run_stage4_baselines.py", "stage4_baselines/summary.txt"),
    ("run_stage4b.py --phase all", "stage4b/summary_all.txt"),
    ("run_stage4c.py", "stage4c/summary.txt"),
    ("run_stage4d.py --synth 4", "stage4d/summary_x4.txt"),
    ("run_stage5.py", "stage5/summary.txt"),
    ("run_stage6a.py", "stage6a/summary.txt"),
    ("run_stage6b.py", "stage6b/summary.txt"),
    ("run_stage6c.py", "stage6c/summary.txt"),
    ("run_stage6d.py", "stage6d/summary.txt"),
    ("run_stage8.py", "stage8/summary.txt"),
    ("run_stage9.py", "stage9/summary.txt"),
    ("run_stage10.py", "stage10/summary.txt"),
    ("run_stage11.py", "stage11/summary.txt"),
    ("run_stage12.py", "stage12/summary.txt"),
    ("run_stage13.py", "stage13/summary.txt"),
    ("run_stage14.py", "stage14/summary.txt"),
    ("run_stage15.py", "stage15/summary.txt"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Все этапы подряд для варианта данных")
    parser.add_argument("--data", choices=("base", "merged"), required=True)
    parser.add_argument("--force", action="store_true", help="запускать этапы, даже если сводка уже есть")
    parser.add_argument("--from", dest="start", default="", help="начать с этого скрипта, например run_stage4c.py")
    args = parser.parse_args()
    # Свои строки - в UTF-8, как у этапов: при выводе в файл иначе кодировка системы, и файл
    # получался в двух кодировках (дома строки [готово] и [стоп] читались кракозябрами)
    sys.stdout.reconfigure(encoding="utf-8")
    env = {**os.environ, "DIPLOM_DATA": args.data, "PYTHONIOENCODING": "utf-8"}
    results = PROJECT_DIR / ("results" if args.data == "base" else "results_merged")
    started = not args.start
    for cmd, rel in STAGES:
        started = started or cmd.split()[0] == args.start
        summary = results / rel
        if not started or (summary.exists() and not args.force):
            print(f"[пропуск] {cmd}: {'до --from' if not started else 'сводка уже есть'}", flush=True)
            continue
        t0 = time.time()
        print(f"[запуск] {cmd} (DIPLOM_DATA={args.data})", flush=True)
        code = subprocess.call([sys.executable, "-u", *cmd.split()], cwd=PROJECT_DIR, env=env)
        print(f"[готово] {cmd}: код {code}, {(time.time() - t0) / 60:.1f} мин", flush=True)
        if not summary.exists():
            print(f"[стоп] {cmd} не записал сводку {summary} - смотреть журнал этапа", flush=True)
            return 1
    print("[все этапы завершены]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
