"""
Выгрузка результатов прогона в home_results/<прогон>/ (решение пользователя 27.10.2026: в репозиторий идут
все сводки, метрики и все PNG, включая графики обучения loss_*; HTML и CSV с координатами полётов - нет, репозиторий
открытый).

  .venv\\Scripts\\python.exe export_results.py --run base_x1 [--dest home_results]

Копируется из results/<прогон>/: summary*.txt, metrics*.json, final_models.json, *.png (без папок _smoke и predict);
из cache/<прогон>/state/: JSON состояния этапов (подобранные параметры, основа, выжившие гипотезы, итоговая
модель), кроме реестров (они большие). Повторная выгрузка перезаписывает файлы. Дальше -
git add home_results/<прогон>; git commit; git push (или копия папки на рабочий ПК).
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_GLOBS = ("summary*.txt", "metrics*.json", "final_models.json", "*.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Выгрузка результатов прогона в home_results")
    parser.add_argument("--run", required=True, help="base_x1, merged_x1, base_x4 или merged_x4")
    parser.add_argument("--dest", type=Path, default=PROJECT_DIR / "home_results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    src = PROJECT_DIR / "results" / args.run
    if not src.exists():
        print(f"Нет {src}")
        return 2
    out = args.dest / args.run
    n = 0
    for pattern in RESULT_GLOBS:
        for path in src.rglob(pattern):
            rel = path.relative_to(src)
            if rel.parts[0] in ("_smoke", "predict"):
                continue
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            n += 1
    state = PROJECT_DIR / "cache" / args.run / "state"
    for path in state.glob("*.json") if state.exists() else []:
        if path.name.startswith("registry_"):
            continue
        target = out / "state" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        n += 1
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Скопировано файлов: {n}, в {out} всего {size:.1f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
