"""
Все этапы версии 2 подряд для одного прогона (вариант данных x объём синтетики).

  .venv\\Scripts\\python.exe run_all.py --data base --synth 1
  .venv\\Scripts\\python.exe run_all.py --data merged --synth 1
  (если хватит времени)  --data base --synth 4,  --data merged --synth 4

Остановка: Ctrl+C (или закрыть окно). Продолжение: та же команда. Этап пропускается, если сводки всех его
случаев уже есть; прерванный этап продолжается без потери: обученные сети лежат в model_store/<прогон>/,
прерванное обучение сети продолжается с последней эпохи, попытки подбора - в базе Optuna
(cache/<прогон>/<этап>/<случай>/optuna.db). --force - заново, --from run_stage4.py - начать с этапа
(после правки этапа пересчитать его и следующие: --from run_stageN.py --force; готовые сети берутся из хранилища),
--case dop_tele - только один случай доплера (nodop, dop_traj, dop_tele; по умолчанию все три),
--smoke - проверка конвейера на подвыборке (результаты в папках _smoke, для диплома не годятся).

Сводка этапа: results/<прогон>/stageN/<случай>/summary.txt (этап 0 - stage0/summary.txt). Этап без сводки после
запуска (необработанное исключение) останавливает прогон: следующие зависят от него. Сводка с FAIL
считается завершённой: FAIL - результат, его разбирают по сводке.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CASES = ("nodop", "dop_traj", "dop_tele")

# (скрипт, зависит ли от случая)
STAGES = [
    ("run_stage0.py", False),
    ("run_stage1.py", True),
    ("run_stage2.py", True),
    ("run_stage3.py", True),
    ("run_stage4.py", True),
    ("run_stage5.py", True),
    ("run_stage6.py", True),
    ("run_stage7.py", True),
    ("run_stage8.py", True),
    ("run_stage9.py", True),
    ("run_stage10.py", True),
    ("run_stage11.py", True),
]


def summary_path(results: Path, script: str, per_case: bool, case: str = None) -> Path:
    stage = script.replace("run_", "").replace(".py", "")
    return results / stage / (case or "") / "summary.txt" if per_case else results / stage / "summary.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Все этапы версии 2 подряд для прогона")
    parser.add_argument("--data", choices=("base", "merged"), required=True)
    parser.add_argument("--synth", choices=("1", "4"), default="1", help="объём синтетики: x1 или x4 (DIPLOM_SYNTH)")
    parser.add_argument("--case", choices=("all",) + CASES, default="all")
    parser.add_argument("--force", action="store_true", help="запускать этапы, даже если сводка уже есть")
    parser.add_argument("--from", dest="start", default="", help="начать с этого скрипта, например run_stage4.py")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    env = {**os.environ, "DIPLOM_DATA": args.data, "DIPLOM_SYNTH": args.synth, "PYTHONIOENCODING": "utf-8"}
    run = f"{args.data}_x{args.synth}"
    results_root = PROJECT_DIR / "results" / run
    results = results_root / ("_smoke" if args.smoke else "")
    cases = CASES if args.case == "all" else (args.case,)
    started = not args.start
    for script, per_case in STAGES:
        started = started or script == args.start
        for case in (cases if per_case else (None,)):
            summary = summary_path(results_root if script == "run_stage0.py" else results, script, per_case, case)
            title = f"{script}" + (f" --case {case}" if case else "")
            if not started or (summary.exists() and not args.force):
                print(f"[пропуск] {title}: {'до --from' if not started else 'сводка уже есть'}", flush=True)
                continue
            cmd = [sys.executable, "-u", script] + (["--case", case] if case else []) + (["--smoke"] if args.smoke and script != "run_stage0.py" else [])
            t0 = time.time()
            print(f"[запуск] {title} (прогон {run})", flush=True)
            try:
                code = subprocess.call(cmd, cwd=PROJECT_DIR, env=env)
            except KeyboardInterrupt:
                print(f"[остановлено] {title}: продолжить - той же командой", flush=True)
                return 130
            print(f"[готово] {title}: код {code}, {(time.time() - t0) / 60:.1f} мин", flush=True)
            if not summary.exists():
                print(f"[стоп] {title} не записал сводку {summary} - смотреть журнал этапа "
                      f"(log/{run}/); если GPU нет, этап 2-9 запускаются только дома", flush=True)
                return 1
    print("[все этапы завершены]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
