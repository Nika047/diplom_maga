"""
Этап 4б: эволюционный многокритериальный поиск архитектуры гибрида IMM + корректор.

Генетический алгоритм (NSGA-II) ищет не только числа, но и устройство сети: стопку
из блоков GRU / LSTM / Mamba / внимания, ширину, отбор групп признаков, параметры
обучения. Цели - ADE на val и время прогноза на пример: результат - фронт Парето
«точность - стоимость». По всем попыткам считается важность групп признаков и
типов блоков.

Шаги (--phase):
  search - поиск (ARCH_TRIALS попыток), фронт Парето, важность признаков и блоков;
           кандидаты - лучшая по ADE и самая быстрая в пределах ARCH_NEAR_BEST от неё;
  final  - кандидаты обучаются с FINAL_SEEDS, оценка на val, test, holdout рядом с IMM
           и итогом этапа 4 (если он уже есть);
  all    - оба шага.
Использует подобранный IMM и окна этапа 4 (cache/stage4/), поэтому запускается после
run_stage4.py --phase imm. Попытки - в cache/stage4/optuna.db, остановка и продолжение
как на этапе 4.

Запуск (домашний ПК, в фоне): .venv\\Scripts\\python.exe run_stage4b.py --phase all
Смоук (рабочий ПК): .venv\\Scripts\\python.exe run_stage4b.py --phase all --smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import splits as splits_mod  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import ade_by_source, mask_forecast, source_section  # noqa: E402
from models import StackCorrector  # noqa: E402
from training import dataset  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from tuning import arch_search, search  # noqa: E402
from viz import train_plots  # noqa: E402
from viz.plots import _save  # noqa: E402

FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"trials": 4, "population": 4, "epochs": 2, "patience": 2, "seeds": (42, 43), "real_step": 8,
         "synth_step": 25}

STAGE = "stage4b"
OUT_DIR = config.RESULTS_DIR / STAGE
CACHE = config.CACHE_DIR / "stage4"
ART = config.ARTIFACTS_DIR / STAGE
CACHE0 = config.CACHE_DIR / "stage0"


def load_sets(smoke: bool) -> dict:
    sets = splits_mod.load_sets(CACHE0)
    if smoke:
        sets["train"] = sets["train"][::SMOKE["real_step"]]
        sets["synth"] = sets["synth"][::SMOKE["synth_step"]]
    return sets


def arch_label(trial_params: dict, n_models: int) -> str:
    a = arch_search.decode(trial_params, n_models)
    return f"[{'+'.join(a['blocks'])}] x{a['hidden']}, признаков групп {len(a['groups'])}"


def plot_pareto(trials: list, front: list, default, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([t.values[1] for t in trials], [t.values[0] for t in trials], s=14, color="#9e9e9e",
               label="все попытки")
    pts = sorted(front, key=lambda t: t.values[1])
    ax.plot([t.values[1] for t in pts], [t.values[0] for t in pts], "-o", color="#d32f2f", ms=5,
            label="фронт Парето")
    if default is not None:
        ax.scatter([default.values[1]], [default.values[0]], s=80, marker="*", color="#1976d2",
                   label="исходная (GRU, GRU, все признаки)", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("время прогноза на пример, мкс (лог. шкала)")
    ax.set_ylabel("ADE на val, м")
    ax.legend(fontsize=8)
    ax.set_title("Эволюционный поиск архитектуры: точность и стоимость", fontsize=10)
    _save(fig, path)


# ============================================================================
# Поиск
# ============================================================================

def phase_search(sets, n_trials, population, epochs, patience, device, smoke, summary, log) -> bool:
    imm_path = CACHE / "imm_best.json"
    if not imm_path.exists():
        summary.check("Подобранный IMM найден", False, f"нет {imm_path}: сначала run_stage4.py --phase imm")
        return False
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    w = config.TUNE_WINDOWS[0]
    n_models = len(config.IMM_MODELS)
    with search.config_override(imm):
        ws_tr = dataset.load_or_build(f"train{'_smoke' if smoke else ''}_w{w}", sets["train"] + sets["synth"],
                                      CACHE, window=w)
        ws_val = dataset.load_or_build(f"val_w{w}", sets["val"], CACHE, window=w)
    study = search.open_study(f"arch{'_smoke' if smoke else ''}", CACHE / "optuna.db", n_objectives=2,
                              population=population)
    objective = arch_search.ArchObjective(ws_tr, ws_val, device, n_models, w, epochs=epochs, patience=patience)
    t0 = time.time()
    log.info("Поиск архитектуры: бюджет %d попыток, популяция %d", n_trials, population)
    search.run_budget(study, objective, n_trials, arch_search.default_params(n_models))
    done = search.completed(study)
    front = study.best_trials
    default = search.default_trial(study)
    summary.check("Бюджет поиска выполнен", len(done) >= n_trials, f"завершено {len(done)} из {n_trials}")
    summary.check("Фронт Парето не пуст, исходная архитектура оценена", len(front) > 0 and default is not None,
                  f"точек фронта {len(front)}, исходная: ADE {default.values[0]:.2f} м, "
                  f"{default.values[1]:.2f} мкс" if default else "исходной нет")

    lines = [f"{'ADE, м':>7s} {'мкс':>7s} {'парам.':>8s}  архитектура / группы признаков"]
    for t in sorted(front, key=lambda t: t.values[0]):
        lines.append(f"{t.values[0]:7.2f} {t.values[1]:7.2f} {t.user_attrs.get('n_params', 0):8d}  "
                     f"{'+'.join(t.user_attrs.get('blocks', []))} x{t.params['hidden']}; "
                     f"{', '.join(t.user_attrs.get('groups', []))}")
    summary.section(f"Фронт Парето ({len(front)} точек из {len(done)} попыток, {(time.time() - t0) / 60:.0f} мин)",
                    lines)
    rows = arch_search.feature_importance(done, n_models)
    summary.section("Важность групп признаков (по всем попыткам)", [
        f"{'группа':22s} {'ADE с ней':>10s} {'ADE без':>9s} {'доля в лучших 20%':>18s}"] + [
        f"{r['group']:22s} {r['ade_with']:10.2f} {r['ade_without']:9.2f} {r['share_in_top']:18.0%}" for r in rows])
    rows_b = arch_search.block_statistics(done)
    summary.section("Типы блоков (по всем попыткам)", [
        f"{'блок':10s} {'ADE с ним':>10s} {'ADE без':>9s} {'доля среди блоков лучших 20%':>30s}"] + [
        f"{r['block']:10s} {r['ade_with']:10.2f} {r['ade_without']:9.2f} {r['share_in_top_blocks']:30.0%}"
        for r in rows_b])

    best = min(done, key=lambda t: t.values[0])
    near = [t for t in done if t.values[0] <= best.values[0] * (1 + config.ARCH_NEAR_BEST)]
    fast = min(near, key=lambda t: t.values[1])
    candidates = {"лучшая по ADE": best.number}
    if fast.number != best.number:
        candidates[f"быстрая (ADE до +{config.ARCH_NEAR_BEST:.0%})"] = fast.number
    out = {name: {"trial": num, "params": study.trials[num].params, "ade_val": study.trials[num].values[0],
                  "latency_us": study.trials[num].values[1]} for name, num in candidates.items()}
    out["_default"] = {"params": default.params, "ade_val": default.values[0], "latency_us": default.values[1]}
    out["_importance"] = {"groups": rows, "blocks": rows_b}
    (CACHE / f"arch_best{'_smoke' if smoke else ''}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                     encoding="utf-8")
    summary.section("Кандидаты для итога", [f"{name}: попытка {v['trial']}, ADE {v['ade_val']:.2f} м, "
                                             f"{v['latency_us']:.2f} мкс, {arch_label(v['params'], n_models)}"
                                             for name, v in out.items() if not name.startswith("_")])
    plot_pareto(done, front, default, OUT_DIR / "pareto.png")
    return True


# ============================================================================
# Итог
# ============================================================================

def summarize_method(rts, fcs) -> dict:
    pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)])
    s = metrics.summarize(pairs)
    out = {"ade": s["ade"], "fde": s["fde"], "r2": s["r2"]}
    for p in ("straight", "entry", "turn"):
        out[p] = metrics.summarize(metrics.select(pairs, pairs["phase"] == PHASE_CODE[p]))["ade"]
    out.update({f"ade_{s}": v for s, v in ade_by_source(rts, fcs).items()})     # только в merged
    return out


def phase_final(sets, seeds, epochs, patience, device, smoke, summary, log) -> bool:
    best_path = CACHE / f"arch_best{'_smoke' if smoke else ''}.json"
    imm_path = CACHE / "imm_best.json"
    if not best_path.exists() or not imm_path.exists():
        summary.check("Результаты поиска найдены", False, "сначала --phase search")
        return False
    found = json.loads(best_path.read_text(encoding="utf-8"))
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    w = config.TUNE_WINDOWS[0]
    n_models = len(config.IMM_MODELS)
    with search.config_override(imm):
        ws_tr = dataset.load_or_build(f"train{'_smoke' if smoke else ''}_w{w}", sets["train"] + sets["synth"],
                                      CACHE, window=w)
        built = {name: dataset.build(sets[name], window=w) for name in config.EVAL_SETS}
    norm = dataset.Normalizer.fit(ws_tr)
    x_tr, y_tr = norm.x(ws_tr.x), norm.y(ws_tr.y)

    table = {name: {"IMM": [summarize_method(b[1], [mask_forecast(r.forecast, b[0].k[b[0].track == i])
                                                    for i, r in enumerate(b[2])])]}
             for name, b in built.items()}
    for cand, info in found.items():
        if cand.startswith("_"):
            continue
        arch = arch_search.decode(info["params"], n_models)
        cols = arch["columns"]
        params = {**arch["train"], "epochs": epochs, "patience": patience}
        for s_i, seed in enumerate(seeds):
            key = f"{cand}, seed {seed}"
            if s_i == 0:
                rng = make_rng("training", seed)
                idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
                torch.manual_seed(seed)
                m = StackCorrector(len(cols), config.HORIZON_SCANS, arch["blocks"], arch["hidden"], arch["dropout"], w)
                first, last = overfit_one_batch(m, x_tr[idx][..., cols], y_tr[idx], config.OVERFIT_STEPS, 3e-3, device)
                summary.check(f"{cand}: overfit one batch", last < config.OVERFIT_TARGET * first,
                              f"loss {first:.4f} -> {last:.4f}")
            rng = make_rng("training", seed)
            torch.manual_seed(seed)
            model = StackCorrector(len(cols), config.HORIZON_SCANS, arch["blocks"], arch["hidden"], arch["dropout"], w)
            ws_val = built["val"][0]
            history, _ = train(model, x_tr[..., cols], y_tr, norm.x(ws_val.x)[..., cols], norm.y(ws_val.y), params,
                               device, rng)
            summary.check(f"{key}: модель обучилась", history["best_val_loss"] < history["val_loss_initial"],
                          f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
            train_plots.plot_history(history, OUT_DIR / f"loss_{cand.split(' ')[0]}_{seed}.png", f"Обучение: {key}")
            ART.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "arch": arch, "seed": seed,
                        "window": w}, ART / f"{cand.split(' ')[0]}_{seed}{'_smoke' if smoke else ''}.pt")
            for name, (ws, rts, runs) in built.items():
                corr = norm.y_inv(predict(model, norm.x(ws.x)[..., cols], device))
                fcs = []
                for i, (rt, run) in enumerate(zip(rts, runs)):
                    sel = ws.track == i
                    fcs.append(mask_forecast(apply_correction(run.forecast, ws.k[sel], ws.psi[sel], corr[sel]),
                                             ws.k[sel]))
                table[name].setdefault(cand, []).append(summarize_method(rts, fcs))
            if s_i == 0:
                got, want = table["val"][cand][0]["ade"], info["ade_val"]
                summary.check(f"{cand}: результат поиска воспроизводится", abs(got / want - 1) <= 0.15,
                              f"ADE val при поиске {want:.2f} м, при итоговом обучении {got:.2f} м")

    stage4 = config.RESULTS_DIR / "stage4" / "final_metrics.json"
    ref = json.loads(stage4.read_text(encoding="utf-8")) if stage4.exists() and not smoke else {}
    for name in config.EVAL_SETS:
        lines = [f"{'метод':34s} {'ADE':>12s} {'FDE':>12s} {'R2':>12s} {'прямая':>12s} {'вход':>12s} {'маневр':>12s}"]
        rows = dict(table[name])
        for label, reps in ref.get(name, {}).items():
            rows[f"этап 4: {label}"] = reps
        for label, reps in rows.items():
            row = f"{label:34s}"
            for key in ("ade", "fde", "r2", "straight", "entry", "turn"):
                vals = np.array([r[key] for r in reps])
                fmt = "{:6.3f}±{:<5.3f}" if key == "r2" else "{:6.1f}±{:<5.1f}"
                row += " " + fmt.format(vals.mean(), vals.std())
            lines.append(row)
        summary.section(f"{name}: среднее ± СКО по seed (м){' - СМОУК' if smoke else ''}", lines)
        source_section(summary, name, rows)
    (OUT_DIR / "final_metrics.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 4б: эволюционный поиск архитектуры")
    parser.add_argument("--phase", choices=("search", "final", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    global CACHE
    if args.smoke:
        CACHE = config.CACHE_DIR / "stage4_smoke"
    log = setup_logging(f"{STAGE}_{args.phase}" + ("_smoke" if args.smoke else ""))
    summary = StageSummary(f"Этап 4б, шаг {args.phase}" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Полный поиск - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sets = load_sets(args.smoke)
    trials = args.trials or (SMOKE["trials"] if args.smoke else config.ARCH_TRIALS)
    population = SMOKE["population"] if args.smoke else config.ARCH_POPULATION
    epochs_s = SMOKE["epochs"] if args.smoke else config.TUNE_EPOCHS
    patience_s = SMOKE["patience"] if args.smoke else config.TUNE_PATIENCE
    epochs_f = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience_f = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS

    ok = True
    t0 = time.time()
    if args.phase in ("search", "all"):
        ok &= phase_search(sets, trials, population, epochs_s, patience_s, device, args.smoke, summary, log)
    if ok and args.phase in ("final", "all"):
        ok &= phase_final(sets, seeds, epochs_f, patience_f, device, args.smoke, summary, log)
    summary.section("Время", [f"шаг {args.phase}: {(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary_{args.phase}{'_smoke' if args.smoke else ''}.txt")
    return 0 if (ok and summary.all_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
