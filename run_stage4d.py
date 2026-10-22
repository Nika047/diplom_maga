"""
Этап 4г: повтор генетического подбора гиперпараметров IMM+GRU на расширенной синтетике.

На этапе 4в сети на синтетике x2 и x4 обучались с гиперпараметрами, подобранными на x1,
и точнее не стали. Возможно, на большем наборе нужны другие (шаг обучения, батч, dropout,
размер сети). Здесь подбор этапа 4 повторяется на наборе с синтетикой --synth (по
умолчанию x4): тот же генетический алгоритм (Optuna, NSGA-II), та же цель (ADE на val),
тот же бюджет TUNE_TRIALS. Первая попытка - лучшая конфигурация этапа 4, поэтому
найденное не хуже неё на этом наборе. Затем итог на 3 seed - val, test, holdout - рядом с
IMM+GRU этапа 4 (подбор и обучение на x1) и этапа 4в (параметры x1, обучение на этом
наборе).

Новые партии синтетики и окна по ним - из кэша этапа 4в (cache/stage4c/): этап 4в должен
быть выполнен. --synth 0 - то же для набора только из реальных полётов.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage4d.py
Смоук (рабочий ПК, нужны смоуки этапов 4 и 4в): .venv\\Scripts\\python.exe run_stage4d.py --smoke --synth 2

Проверки: бюджет выполнен, лучшее не хуже конфигурации этапа 4, проверки обучения.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import mask_forecast  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from tuning import search  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"trials": 3, "epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage4d"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.PROJECT_DIR / "artifacts" / STAGE
EVAL_SETS = ("val", "test", "holdout")
HYB = "IMM+GRU"


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 4г: подбор IMM+GRU на расширенной синтетике")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synth", type=int, default=4, help="синтетика: 0 - только реальные, 1..4 - x1..x4")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    tag = ("_smoke" if args.smoke else "") + f"_x{args.synth}"
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary(f"Этап 4г: подбор IMM+GRU, синтетика x{args.synth}"
                           + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Подбор и обучение этапа 4г - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache4c = config.CACHE_DIR / ("stage4c_smoke" if args.smoke else "stage4c")
    cache = config.CACHE_DIR / (STAGE + ("_smoke" if args.smoke else ""))
    trials = args.trials or (SMOKE["trials"] if args.smoke else config.TUNE_TRIALS)
    epochs_t = SMOKE["epochs"] if args.smoke else config.TUNE_EPOCHS
    patience_t = SMOKE["patience"] if args.smoke else config.TUNE_PATIENCE
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    t0 = time.time()

    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    params4 = best_all[HYB]["params"]
    window = params4["window"]

    # Набор train: реальные + синтетика этапа 4 + партии этапа 4в
    sets = load_data(args.smoke)
    real, synth = sets["train"], sets["synth"]
    extra_path = cache4c / "synth_extra.pkl"
    extra = []
    if extra_path.exists():
        with open(extra_path, "rb") as fh:
            extra = pickle.load(fh)
    batches = [[tr for tr in extra if tr.flight_id.startswith(f"synthX{b}_")] for b in range(1, 4)]
    batches = [b for b in batches if b]
    if args.synth - 1 > len(batches):
        summary.check("Партии синтетики этапа 4в найдены", False,
                      f"нужно {args.synth - 1} партий, в {extra_path} их {len(batches)} - сначала run_stage4c.py")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    with search.config_override(imm):
        ws_base = dataset.load_or_build(f"train{'_smoke' if args.smoke else ''}_w{window}", real + synth, cache4,
                                        window=window)
        ws_extra = [dataset.load_or_build(f"synth_extra{b + 1}_w{window}", batches[b], cache4c, window=window)
                    for b in range(max(args.synth - 1, 0))]
        ws_val = dataset.load_or_build(f"val_w{window}", sets["val"], cache4, window=window)
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
    ws_tr = ws_base.subset_tracks(np.arange(len(real))) if args.synth == 0 else dataset.concat([ws_base] + ws_extra)
    log.info("Набор train: синтетика x%d, %d окон", args.synth, len(ws_tr))

    # Подбор
    study = search.open_study(f"{HYB}_x{args.synth}", cache / "optuna.db")
    objective = search.CorrectorObjective("GRU", "hybrid", {window: (ws_tr, ws_val)}, device, epochs=epochs_t,
                                          patience=patience_t)
    search.run_budget(study, objective, trials, params4)
    done = search.completed(study)
    best, default = study.best_trial, search.default_trial(study)
    summary.check("Бюджет подбора выполнен", len(done) >= trials, f"завершено {len(done)} из {trials}")
    summary.check("Лучшее не хуже конфигурации этапа 4 на этом наборе", best.value <= default.value + 1e-9,
                  f"ADE val: конфигурация этапа 4 {default.value:.2f} м, лучшая {best.value:.2f} м")
    fmt = lambda p: ", ".join(f"{k}={round(v, 5) if isinstance(v, float) else v}" for k, v in p.items())  # noqa: E731
    spent = sum(t.user_attrs.get("seconds", 0.0) for t in done) / 60
    summary.section(f"Подбор на синтетике x{args.synth}: {len(ws_tr)} окон, {len(done)} попыток, {spent:.0f} мин", [
        f"конфигурация этапа 4: {fmt(params4)} -> ADE val {default.value:.2f} м",
        f"лучшая:               {fmt(best.params)} -> ADE val {best.value:.2f} м ({best.value / default.value - 1:+.1%})"])

    # Итог на 3 seed
    _, hp, train_params = search.split_params(best.params)
    params = {**train_params, "epochs": epochs, "patience": patience}
    ks = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
          for name, b in built.items()}
    reps = {name: [] for name in EVAL_SETS}
    with search.config_override(imm):
        for s_i, seed in enumerate(seeds):
            res = run_experiment(f"{HYB} x{args.synth}, seed {seed}", "GRU", "hybrid", ws_tr, built["val"][0], params,
                                 device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                 model_kwargs=hp, overfit=s_i == 0,
                                 extra_sets={n: built[n][0] for n in ("test", "holdout")})
            preds = {"val": res["pred"], **res["extra"]}
            for name in EVAL_SETS:
                ws, rts, runs = built[name]
                fcs = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                      preds[name][ws.track == i]), ks[name][i])
                       for i, run in enumerate(runs)]
                reps[name].append(summarize_method(rts, fcs))

    def cell(rs, key):
        v = np.array([r[key] for r in rs], dtype=float)
        return f"{v.mean():.1f}±{v.std():.1f}" if len(v) > 1 else f"{v.mean():.1f}"

    cols = ("ade", "fde", "straight", "entry", "turn", "exit")
    rows = {f"подбор на x{args.synth}, обучение на x{args.synth}": reps}
    s4 = config.RESULTS_DIR / "stage4" / "final_metrics.json"   # этап 4 пишет его и в смоуке
    if s4.exists():
        m4 = json.loads(s4.read_text(encoding="utf-8"))
        if all(HYB in m4.get(n, {}) for n in EVAL_SETS):
            rows["этап 4: подбор и обучение на x1"] = {n: m4[n][HYB] for n in EVAL_SETS}
    s4c = config.RESULTS_DIR / "stage4c" / f"metrics{'_smoke' if args.smoke else ''}.json"
    if s4c.exists():
        rs = [r for r in json.loads(s4c.read_text(encoding="utf-8"))
              if r["arch"] == "GRU" and r["frac"] == float(args.synth)
              and (r["hidden"], r["layers"]) == (params4["hidden"], params4["layers"])]
        if rs:
            rows[f"этап 4в: параметры x1, обучение на x{args.synth}"] = {n: [r[n] for r in rs] for n in EVAL_SETS}
    for name in EVAL_SETS:
        lines = [f"{'IMM+GRU':44s}" + "".join(f" {c:>11s}" for c in ("ADE", "FDE", "прямая", "вход", "маневр", "выход"))]
        for label, by_set in rows.items():
            lines.append(f"{label:44s}" + "".join(f" {cell(by_set[name], k):>11s}" for k in cols))
        summary.section(f"{name}: ADE (м), среднее ± СКО по seed", lines)

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps({"best_params": best.params, "reports": reps},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
