"""
Этап 4: генетический подбор и итоговое сравнение.

Шаги (--phase):
  imm   - подбор параметров IMM (шумы CV и CT, вероятность остаться в модели) по ADE
          IMM на реальных трассах train. Итог - cache/stage4/imm_best.json;
  tune  - подбор гиперпараметров каждой модели (LSTM, GRU, Transformer, Mamba в гибриде,
          GRU и Mamba без IMM) с одинаковым бюджетом попыток, при подобранном IMM; цель -
          ADE на val. Итог - cache/stage4/best_params.json;
  final - лучшие параметры, обучение с несколькими seed (FINAL_SEEDS), оценка на test и
          holdout вместе с базовыми методами (CV, KF, IMM исходный и подобранный);
  all   - все три шага подряд.
Попытки подбора хранятся в cache/stage4/optuna.db: прерванный запуск продолжается.
test и holdout впервые используются здесь, в шаге final.

Запуск (домашний ПК, в фоне - см. PLATFORM.md):
  .venv\\Scripts\\python.exe run_stage4.py --phase all
Смоук (рабочий ПК): .venv\\Scripts\\python.exe run_stage4.py --phase all --smoke --only GRU
  --only GRU,Mamba - только эти корректоры (в гибриде и без IMM);
  --trials N, --imm-trials N - изменить бюджет (одинаково для всех моделей).
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from data import splits as splits_mod  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction, apply_e2e  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from filters.extrapolation import cv_extrapolate  # noqa: E402
from filters.imm import IMM  # noqa: E402
from radar.model import observe  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from tuning import search  # noqa: E402
from viz.filter_plots import METHOD_COLORS  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"imm_trials": 3, "trials": 3, "epochs": 2, "patience": 2, "seeds": (42, 43),
         "real_step": 8, "synth_step": 25}

EXPERIMENTS = [("IMM+LSTM", "LSTM", "hybrid"), ("IMM+GRU", "GRU", "hybrid"),
               ("IMM+Transformer", "Transformer", "hybrid"), ("IMM+Mamba", "Mamba", "hybrid"),
               ("GRU (без IMM)", "GRU", "e2e"), ("Mamba (без IMM)", "Mamba", "e2e")]

STAGE = "stage4"
OUT_DIR = config.RESULTS_DIR / STAGE
CACHE = config.CACHE_DIR / STAGE
ART = config.PROJECT_DIR / "artifacts" / STAGE
CACHE0 = config.CACHE_DIR / "stage0"
TURN_CODES = (PHASE_CODE["entry"], PHASE_CODE["turn"])


def load_data(smoke: bool) -> dict:
    with open(CACHE0 / "tracks_real.pkl", "rb") as fh:
        real = pickle.load(fh)
    with open(CACHE0 / "tracks_synth_train.pkl", "rb") as fh:
        synth = pickle.load(fh)
    splits = splits_mod.load_splits(CACHE0 / "splits.json")
    sets = {name: [tr for tr in real if splits.get(tr.log_file) == name] for name in ("train", "val", "test", "holdout")}
    sets["synth"] = synth
    if smoke:
        sets["train"] = sets["train"][::SMOKE["real_step"]]
        sets["synth"] = sets["synth"][::SMOKE["synth_step"]]
    sets["_splits"] = splits
    return sets


def plot_convergence(studies: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, values in studies.items():
        best = np.minimum.accumulate(values)
        ax.plot(np.arange(1, len(best) + 1), best, "-", color=METHOD_COLORS.get(name), label=name)
    ax.set_xlabel("номер попытки")
    ax.set_ylabel("лучшая ADE, м")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def imm_values() -> dict:
    path = CACHE / "imm_best.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["params"]


# ============================================================================
# Шаг 1: IMM
# ============================================================================

def phase_imm(sets: dict, n_trials: int, summary: StageSummary, log) -> bool:
    tracks = sets["train"][::config.TUNE_IMM_TRACK_STEP]
    rts = [observe(tr) for tr in tracks]
    log.info("Подбор IMM: %d трасс train, %d обзоров, бюджет %d попыток", len(rts), sum(len(r.t) for r in rts), n_trials)
    study = search.open_study("imm", CACHE / "optuna.db")
    t0 = time.time()
    search.run_budget(study, search.imm_objective(rts), n_trials, search.default_imm_params())
    done = search.completed(study)
    default = search.default_trial(study).value
    best = study.best_trial
    summary.check("IMM: бюджет выполнен", len(done) >= n_trials, f"завершено {len(done)} из {n_trials}")
    summary.check("IMM: лучшее не хуже исходного", best.value <= default + 1e-9,
                  f"ADE train: исходные {default:.2f} м, лучшие {best.value:.2f} м")

    # Проверка на val, который в подборе IMM не участвовал
    val_rts = [observe(tr) for tr in sets["val"]]
    val_ade = {}
    for name, values in (("исходный", search.default_imm_params()), ("подобранный", best.params)):
        with search.config_override(values):
            imm = IMM()
            val_ade[name] = metrics.summarize(metrics.concat_pairs(
                [metrics.forecast_pairs(rt, imm.run(rt).forecast) for rt in val_rts]))["ade"]
    summary.section("Шаг 1: подбор параметров IMM", [
        f"трасс train {len(rts)}, попыток {len(done)}, время {time.time() - t0:.0f} с",
        f"исходные: {search.default_imm_params()} -> ADE train {default:.2f} м",
        f"лучшие:   { {k: round(v, 4) for k, v in best.params.items()} } -> ADE train {best.value:.2f} м "
        f"({best.value / default - 1:+.1%}), на маневре {best.user_attrs.get('maneuver_ade', float('nan')):.1f} м",
        f"проверка на val (в подборе не участвовал): исходный IMM {val_ade['исходный']:.2f} м, "
        f"подобранный {val_ade['подобранный']:.2f} м",
    ])
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "imm_best.json").write_text(json.dumps({"params": best.params, "ade_train": best.value,
                                                     "ade_train_default": default, "ade_val": val_ade},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    plot_convergence({"IMM": [t.value for t in done]}, OUT_DIR / "convergence_imm.png", "Генетический подбор IMM")
    return True


# ============================================================================
# Шаг 2: корректоры
# ============================================================================

def build_windows_for(sets: dict, windows: list, smoke: bool) -> dict:
    out = {}
    tag = "_smoke" if smoke else ""
    for w in windows:
        ws_tr = dataset.load_or_build(f"train{tag}_w{w}", sets["train"] + sets["synth"], CACHE, window=w)
        ws_val = dataset.load_or_build(f"val_w{w}", sets["val"], CACHE, window=w)
        out[w] = (ws_tr, ws_val)
    return out


def phase_tune(sets: dict, experiments: list, n_trials: int, epochs: int, patience: int, device, smoke: bool,
               summary: StageSummary, log) -> bool:
    imm = imm_values()
    if imm is None:
        summary.check("Подобранный IMM найден", False, "сначала --phase imm")
        return False
    with search.config_override(imm):
        windows = build_windows_for(sets, list(config.TUNE_WINDOWS), smoke)
        best_path = CACHE / "best_params.json"
        best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
        curves, lines = {}, [f"{'модель':16s} {'попыток':>7s} {'исходные':>9s} {'лучшие':>7s} {'изм.':>7s} "
                             f"{'время, мин':>10s}  лучшие параметры"]
        counts = []
        for label, corrector, mode in experiments:
            study = search.open_study(f"{label}{'_smoke' if smoke else ''}", CACHE / "optuna.db")
            objective = search.CorrectorObjective(corrector, mode, windows, device, epochs=epochs, patience=patience)
            t0 = time.time()
            log.info("Подбор %s: бюджет %d попыток", label, n_trials)
            search.run_budget(study, objective, n_trials, search.default_corrector_params())
            done = search.completed(study)
            counts.append(len(done))
            best = study.best_trial
            default = search.default_trial(study).value
            best_all[label] = {"params": best.params, "ade_val": best.value, "ade_val_default": default,
                               "trials": len(done)}
            summary.check(f"{label}: лучшее не хуже исходного", best.value <= default + 1e-9,
                          f"ADE val: исходные {default:.2f} м, лучшие {best.value:.2f} м")
            spent = sum(t.user_attrs.get("seconds", 0.0) for t in done) / 60
            lines.append(f"{label:16s} {len(done):7d} {default:9.2f} {best.value:7.2f} {best.value / default - 1:+7.1%} "
                         f"{spent:10.1f}  " + ", ".join(f"{k}={round(v, 5) if isinstance(v, float) else v}"
                                                        for k, v in best.params.items()))
            curves[label] = [t.value for t in done]
            best_path.write_text(json.dumps(best_all, ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("Подбор %s: лучшее %.2f м (исходные %.2f м), %.1f мин", label, best.value, default,
                     (time.time() - t0) / 60)
        summary.check("Бюджет подбора одинаков для всех моделей", len(set(counts)) == 1 and counts[0] >= n_trials,
                      f"завершённых попыток: {counts} (бюджет {n_trials})")
        summary.section("Шаг 2: подбор гиперпараметров (ADE на val, м)", lines + [
            f"окна: {list(config.TUNE_WINDOWS)}, эпох в попытке не больше {epochs}, ранняя остановка {patience}"])
        plot_convergence(curves, OUT_DIR / "convergence_correctors.png", "Генетический подбор корректоров")
    return True


# ============================================================================
# Шаг 3: итог
# ============================================================================

def phase_final(sets: dict, experiments: list, seeds: tuple, epochs: int, patience: int, device, smoke: bool,
                summary: StageSummary, log) -> bool:
    imm = imm_values()
    best_path = CACHE / "best_params.json"
    if imm is None or not best_path.exists():
        summary.check("Результаты подбора найдены", False, "сначала --phase imm и --phase tune")
        return False
    best_all = json.loads(best_path.read_text(encoding="utf-8"))
    missing = [label for label, _, _ in experiments if label not in best_all]
    if missing:
        summary.check("Подобраны параметры всех моделей", False, f"нет: {missing}")
        return False

    splits = sets["_splits"]
    used = {name: {tr.log_file for tr in sets[name]} for name in ("train", "val", "test", "holdout")}
    leak = (used["test"] | used["holdout"]) & (used["train"] | used["val"])
    summary.check("test и holdout не пересекаются с train и val", not leak and used["test"] and used["holdout"],
                  f"логов: test {len(used['test'])}, holdout {len(used['holdout'])}, пересечений {len(leak)}")

    windows = sorted({best_all[label]["params"]["window"] for label, _, _ in experiments})
    w_max = windows[-1]
    eval_sets = ("val", "test", "holdout")
    with search.config_override(imm):
        train_ws = {w: dataset.load_or_build(f"train{'_smoke' if smoke else ''}_w{w}", sets["train"] + sets["synth"],
                                             CACHE, window=w) for w in windows}
        built = {(name, w): dataset.build(sets[name], window=w) for name in eval_sets for w in windows}

    # Общие обзоры оценки - по самому длинному окну
    common = {}
    for name in eval_sets:
        ws = built[(name, w_max)][0]
        common[name] = [np.array(sorted(ws.k[ws.track == i]), dtype=int) for i in range(len(sets[name]))]

    forecasts = {name: {} for name in eval_sets}
    for name in eval_sets:
        _, rts, runs_tuned = built[(name, w_max)]
        kf = IMM(("CV",), mu0=(1.0,))
        with search.config_override(search.default_imm_params()):
            imm_default = IMM()
            runs_default = [imm_default.run(rt) for rt in rts]
        forecasts[name]["CV"] = [mask_forecast(cv_extrapolate(rt), ks) for rt, ks in zip(rts, common[name])]
        forecasts[name]["KF"] = [mask_forecast(kf.run(rt).forecast, ks) for rt, ks in zip(rts, common[name])]
        forecasts[name]["IMM (исходный)"] = [mask_forecast(r.forecast, ks) for r, ks in zip(runs_default, common[name])]
        forecasts[name]["IMM"] = [mask_forecast(r.forecast, ks) for r, ks in zip(runs_tuned, common[name])]

    per_seed = {name: {} for name in eval_sets}   # {набор: {метод: [отчёт по seed]}}
    histories = {}
    with search.config_override(imm):
        for label, corrector, mode in experiments:
            window, hp, train_params = search.split_params(best_all[label]["params"])
            params = {**train_params, "epochs": epochs, "patience": patience}
            ws_tr = train_ws[window]
            ws_val = built[("val", window)][0]
            for s_i, seed in enumerate(seeds):
                key = f"{label}, seed {seed}"
                res = run_experiment(key, corrector, mode, ws_tr, ws_val, params, device, summary, log, smoke,
                                     OUT_DIR, ART, window=window, seed=seed, model_kwargs=hp, overfit=s_i == 0,
                                     extra_sets={n: built[(n, window)][0] for n in ("test", "holdout")})
                histories[key] = res["history"]
                preds = {"val": res["pred"], **res["extra"]}
                for name in eval_sets:
                    ws, rts, runs = built[(name, window)]
                    fcs = []
                    for i, (rt, run) in enumerate(zip(rts, runs)):
                        sel = ws.track == i
                        ks, pred = ws.k[sel], preds[name][sel]
                        fc = (apply_correction(run.forecast, ks, ws.psi[sel], pred) if mode == "hybrid"
                              else apply_e2e(rt, ks, ws.psi_e2e[sel], pred))
                        fcs.append(mask_forecast(fc, common[name][i]))
                    rep = summarize_method(built[(name, w_max)][1], fcs)
                    per_seed[name].setdefault(label, []).append(rep)
                    if s_i == 0:
                        forecasts[name][label] = fcs
                if s_i == 0:
                    ga = best_all[label]["ade_val"]
                    got = per_seed["val"][label][0]["ade"]
                    summary.check(f"{label}: лучшая конфигурация воспроизводится", abs(got / ga - 1) <= 0.15,
                                  f"ADE val при подборе {ga:.2f} м, при итоговом обучении (seed {seed}, до "
                                  f"{epochs} эпох) {got:.2f} м")

    for name in eval_sets:
        rts = built[(name, w_max)][1]
        compare(rts, forecasts[name], summary, f"{name} (seed {seeds[0]})", OUT_DIR, name)
        lines = [f"{'метод':16s} {'ADE':>13s} {'FDE':>13s} {'R2':>13s} {'прямая':>13s} {'вход':>13s} {'маневр':>13s}"]
        base = summarize_method(rts, forecasts[name]["IMM"])
        lines.append(f"{'IMM':16s} {base['ade']:13.1f} {base['fde']:13.1f} {base['r2']:13.3f} {base['straight']:13.1f} "
                     f"{base['entry']:13.1f} {base['turn']:13.1f}")
        for label, reps in per_seed[name].items():
            row = f"{label:16s}"
            for key in ("ade", "fde", "r2", "straight", "entry", "turn"):
                vals = np.array([r[key] for r in reps])
                fmt = "{:6.3f}±{:<5.3f}" if key == "r2" else "{:6.1f}±{:<5.1f}"
                row += " " + fmt.format(vals.mean(), vals.std())
            lines.append(row)
        summary.section(f"{name}: среднее ± СКО по {len(seeds)} seed (м){' - СМОУК' if smoke else ''}", lines)

    (OUT_DIR / "final_metrics.json").write_text(json.dumps(per_seed, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "final_histories.json").write_text(json.dumps(histories, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def summarize_method(rts: list, fcs: list) -> dict:
    pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)])
    s = metrics.summarize(pairs)
    out = {"ade": s["ade"], "fde": s["fde"], "r2": s["r2"], "rmse": s["rmse"], "n": s["n"]}
    for p in ("straight", "entry", "turn", "exit"):
        out[p] = metrics.summarize(metrics.select(pairs, pairs["phase"] == PHASE_CODE[p]))["ade"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 4: генетический подбор и итог")
    parser.add_argument("--phase", choices=("imm", "tune", "final", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--imm-trials", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging(f"{STAGE}_{args.phase}" + ("_smoke" if args.smoke else ""))
    summary = StageSummary(f"Этап 4, шаг {args.phase}" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke and args.phase != "imm":
        log.error("GPU нет. Подбор корректоров и итог - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    experiments = [e for e in EXPERIMENTS if not only or e[1] in only]
    sets = load_data(args.smoke)
    imm_trials = args.imm_trials or (SMOKE["imm_trials"] if args.smoke else config.TUNE_IMM_TRIALS)
    trials = args.trials or (SMOKE["trials"] if args.smoke else config.TUNE_TRIALS)
    epochs_tune = SMOKE["epochs"] if args.smoke else config.TUNE_EPOCHS
    patience_tune = SMOKE["patience"] if args.smoke else config.TUNE_PATIENCE
    epochs_final = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience_final = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    if args.smoke:
        global CACHE
        CACHE = config.CACHE_DIR / "stage4_smoke"

    ok = True
    t0 = time.time()
    if args.phase in ("imm", "all"):
        ok &= phase_imm(sets, imm_trials, summary, log)
    if ok and args.phase in ("tune", "all"):
        ok &= phase_tune(sets, experiments, trials, epochs_tune, patience_tune, device, args.smoke, summary, log)
    if ok and args.phase in ("final", "all"):
        ok &= phase_final(sets, experiments, seeds, epochs_final, patience_final, device, args.smoke, summary, log)
    summary.section("Время", [f"шаг {args.phase}: {(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary_{args.phase}{'_smoke' if args.smoke else ''}.txt")
    return 0 if (ok and summary.all_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
