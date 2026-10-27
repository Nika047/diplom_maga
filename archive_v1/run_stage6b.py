"""
Этап 6б: строб - область, в которой окажется цель через h обзоров.

Новый компонент - область прогноза с заданным покрытием GATE_COVERAGE. Четыре варианта
на одних и тех же обзорах:
  1. IMM, эллипс хи-квадрат - классический строб: ковариация прогноза IMM
     (IMM.run(with_cov=True)), порог - квантиль хи-квадрат с 2 степенями свободы;
  2. IMM, эллипс, конформный - та же форма, радиус калибруется на val;
  3. гибрид, круг, конформный - круг вокруг прогноза IMM+GRU, радиус по val;
  4. гибрид, эллипс IMM, конформный - центр - прогноз гибрида, форма - ковариация IMM
     (подстраивается под маневр), радиус по val.
Калибровка (split conformal, eval/metrics.conformal_quantile) - по каждому горизонту
на val; оценка на test и holdout: покрытие в целом, по участкам и горизонтам,
эквивалентный радиус. IMM и IMM+GRU - подобранные на этапе 4, сеть - 3 seed.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage6b.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage6b.py --smoke

Проверки: конформный квантиль на подставных данных, ковариация не меняет прогноз IMM,
ковариации положительно определены, на калибровочном наборе покрытие не ниже
заданного, гибрид повторяет этап 4, проверки обучения из training/experiment.py.
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
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import mask_forecast  # noqa: E402
from filters.imm import IMM  # noqa: E402
from run_stage4 import load_data  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from tuning import search  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage6b"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
REGIONS = ("IMM, эллипс хи-квадрат", "IMM, эллипс, конформный", f"{HYB}, круг, конформный",
           f"{HYB}, эллипс IMM, конформный")
COLORS = dict(zip(REGIONS, ("#d32f2f", "#f48fb1", "#9ccc65", "#2e7d32")))


def check_quantile(summary: StageSummary) -> None:
    """Подставные данные: 99 значений 1..99, покрытие 0,9 - значение с номером ceil(100 * 0,9) = 90."""
    got = metrics.conformal_quantile(np.arange(99, 0, -1), 0.9)
    tiny = metrics.conformal_quantile(np.arange(1, 10), 0.95)   # ceil(10 * 0,95) = 10 > 9 - бесконечность
    summary.check("Конформный квантиль на подставных данных", got == 90.0 and tiny == float("inf"),
                  f"квантиль 0,9 из 1..99 = {got:g} (ожидалось 90), при малом n - {tiny}")


def gate_pairs(rts: list, fcs: list, runs_cov: list) -> dict:
    """Пары (k, h) всех трасс: ошибка по горизонтали, ковариация IMM, h, участок, трасса, обзор."""
    parts = {"err": [], "cov": [], "h": [], "phase": [], "track": [], "k": []}
    for i, (rt, fc, run) in enumerate(zip(rts, fcs, runs_cov)):
        pr = metrics.forecast_pairs(rt, fc)
        parts["err"].append(pr["err"][:, :2])
        parts["cov"].append(run.forecast_cov[pr["k"], pr["h"] - 1])
        parts["h"].append(pr["h"])
        parts["phase"].append(pr["phase"])
        parts["track"].append(np.full(len(pr["k"]), i))
        parts["k"].append(pr["k"])
    return {key: np.concatenate(val) for key, val in parts.items()}


def region_params(cal_imm: dict, cal_hyb: dict, coverage: float) -> dict:
    """Пороги по горизонтам, откалиброванные на val: {область: {h: порог}}."""
    out = {name: {} for name in REGIONS}
    q_chi2 = -2.0 * np.log(1.0 - coverage)   # квантиль хи-квадрат с 2 степенями свободы
    for h in range(1, config.HORIZON_SCANS + 1):
        si, sh = cal_imm["h"] == h, cal_hyb["h"] == h
        out[REGIONS[0]][h] = q_chi2
        out[REGIONS[1]][h] = metrics.conformal_quantile(
            metrics.mahalanobis2(cal_imm["err"][si], cal_imm["cov"][si]), coverage)
        out[REGIONS[2]][h] = metrics.conformal_quantile(np.linalg.norm(cal_hyb["err"][sh], axis=1), coverage)
        out[REGIONS[3]][h] = metrics.conformal_quantile(
            metrics.mahalanobis2(cal_hyb["err"][sh], cal_hyb["cov"][sh]), coverage)
    return out


def apply_region(name: str, thr: dict, pairs: dict) -> tuple:
    """(внутри области (N,), площадь (N,)) для пар pairs."""
    q = np.array([thr[name][int(h)] for h in pairs["h"]])
    if name == REGIONS[2]:
        return np.linalg.norm(pairs["err"], axis=1) <= q, np.pi * q ** 2
    det = np.linalg.det(pairs["cov"])
    return metrics.mahalanobis2(pairs["err"], pairs["cov"]) <= q, np.pi * q * np.sqrt(det)


def plot_by_h(reports: dict, key: str, path: Path, title: str, ylabel: str, target: float = None) -> None:
    hs = np.arange(1, config.HORIZON_SCANS + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, reps in reports.items():
        vals = np.array([[r[f"{key}_h{h}"] for h in hs] for r in reps])
        ax.errorbar(hs * config.SCAN_PERIOD, vals.mean(axis=0), yerr=vals.std(axis=0), fmt="-o", ms=4, capsize=3,
                    color=COLORS[name], label=name)
    if target is not None:
        ax.axhline(target, color="k", lw=0.8, ls="--", label=f"заданное {target:.0%}")
    ax.set_xlabel("горизонт прогноза, с")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def _ellipse(center, cov, q, color, label=None):
    vals, vecs = np.linalg.eigh(cov * q)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return Ellipse(center, 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]), angle=angle, fill=False, color=color,
                   lw=1.2, label=label)


def plot_example(rt, run_cov, fc_imm, fc_hyb, thr: dict, path: Path) -> None:
    """Стробы на горизонте GATE_EXAMPLE_H из обзоров o - 4, o, o + 4 вокруг одного начала маневра o."""
    h = config.GATE_EXAMPLE_H
    offsets = np.array([-4, 0, 4])
    ok = [o for o in rt.onsets if o - 6 >= 0 and o + 6 + h < len(rt.t)
          and not np.isnan(fc_hyb[o + offsets, h - 1]).any()]
    if not ok:
        return
    o = ok[len(ok) // 2]
    origin = rt.truth[o, :2]
    fig, ax = plt.subplots(figsize=(8, 7))
    seg = slice(max(o - 15, 0), min(o + 20 + h, len(rt.t)))
    ax.plot(*(rt.truth[seg, :2] - origin).T, "-", color="0.4", lw=1, label="истина")
    ax.plot(*(rt.meas[seg, :2][rt.detected[seg]] - origin).T, ".", color="0.6", ms=3, label="отметки")
    for j, k in enumerate(o + offsets):
        first = j == 0
        c = run_cov.forecast_cov[k, h - 1]
        ax.add_patch(_ellipse(fc_imm[k, h - 1, :2] - origin, c, thr[REGIONS[0]][h], COLORS[REGIONS[0]],
                              REGIONS[0] if first else None))
        ax.add_patch(_ellipse(fc_hyb[k, h - 1, :2] - origin, c, thr[REGIONS[3]][h], COLORS[REGIONS[3]],
                              REGIONS[3] if first else None))
        ax.plot(*(fc_hyb[k, h - 1, :2] - origin), "x", color=COLORS[REGIONS[3]], ms=6)
        ax.plot(*(rt.truth[k + h, :2] - origin), "*", color="k", ms=8, label="истина через h" if first else None)
        ax.plot(*(rt.truth[k, :2] - origin), "o", color="k", ms=3)
    ax.set_aspect("equal")
    ax.set_xlabel("восток, м")
    ax.set_ylabel("север, м")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"Стробы через {h * config.SCAN_PERIOD:.0f} с из обзоров вокруг начала маневра ({rt.flight_id})",
                 fontsize=9)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 6б: строб")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 6б: строб" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 6б - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    coverage = config.GATE_COVERAGE
    t0 = time.time()

    check_quantile(summary)
    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window, model_kwargs, train_params = search.split_params(best_all[HYB]["params"])
    params = {**train_params, "epochs": epochs, "patience": patience}

    sets = load_data(args.smoke)
    with search.config_override(imm):
        ws_tr = dataset.load_or_build(f"train{tag}_w{window}", sets["train"] + sets["synth"], cache4, window=window)
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
        imm_cov = IMM()
        runs_cov = {name: [imm_cov.run(rt, with_cov=True) for rt in built[name][1]] for name in EVAL_SETS}

    same = max(float(np.nanmax(np.abs(a.forecast - b.forecast), initial=0.0))
               for name in EVAL_SETS for a, b in zip(built[name][2], runs_cov[name]))
    summary.check("Ковариация не меняет прогноз IMM", same == 0.0, f"наибольшее расхождение прогноза {same:.3g} м")
    ks = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
          for name, b in built.items()}
    fc_imm = {name: [mask_forecast(r.forecast, k) for r, k in zip(built[name][2], ks[name])] for name in EVAL_SETS}
    pairs_imm = {name: gate_pairs(built[name][1], fc_imm[name], runs_cov[name]) for name in EVAL_SETS}
    eig = min(float(np.linalg.eigvalsh(p["cov"]).min()) for p in pairs_imm.values())
    summary.check("Ковариации прогноза IMM положительно определены", eig > 0.0, f"наименьшее собственное {eig:.3g} м^2")

    reports = {name: {r: [] for r in REGIONS} for name in EVAL_SETS}
    thresholds, fc_hyb0 = [], None
    with search.config_override(imm):
        for s_i, seed in enumerate(seeds):
            res = run_experiment(f"{HYB}, seed {seed}", "GRU", "hybrid", ws_tr, built["val"][0], params, device,
                                 summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                 model_kwargs=model_kwargs, overfit=s_i == 0,
                                 extra_sets={n: built[n][0] for n in config.EVAL_SETS[1:]})
            preds = {"val": res["pred"], **res["extra"]}
            pairs_hyb = {}
            for name in EVAL_SETS:
                ws, rts, runs = built[name]
                fcs = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                      preds[name][ws.track == i]), ks[name][i])
                       for i, run in enumerate(runs)]
                pairs_hyb[name] = gate_pairs(rts, fcs, runs_cov[name])
                if s_i == 0 and name == "test":
                    fc_hyb0 = fcs
            if s_i == 0:
                ade = float(np.linalg.norm(pairs_hyb["val"]["err"], axis=1).mean())
                ga = best_all[HYB]["ade_val"]
                summary.check(f"{HYB}: повторяет этап 4", abs(ade / ga - 1) <= 0.05,
                              f"ADE val на этапе 4 {ga:.2f} м, здесь (seed {seed}) {ade:.2f} м")
            thr = region_params(pairs_imm["val"], pairs_hyb["val"], coverage)
            thresholds.append(thr)
            for name in EVAL_SETS:
                for region in REGIONS:
                    p = pairs_imm[name] if region.startswith("IMM,") else pairs_hyb[name]
                    inside, area = apply_region(region, thr, p)
                    reports[name][region].append(metrics.gate_report(inside, area, p["phase"], p["h"]))
            if s_i == 0:
                low = min(reports["val"][r][0]["coverage"] for r in REGIONS[1:])
                summary.check("На калибровочном наборе (val) покрытие не ниже заданного", low >= coverage,
                              f"наименьшее покрытие конформных стробов {low:.4f} при заданном {coverage:.2f}")

    def cell(reps, key, fmt):
        vals = np.array([r[key] for r in reps], dtype=float)
        s = format(vals.mean(), fmt)
        return s if len(vals) == 1 or np.allclose(vals, vals[0]) else f"{s}±{format(vals.std(), fmt)}"

    for name in EVAL_SETS:
        lines = [f"{'строб':34s} {'покрытие':>12s} {'прямая':>12s} {'вход':>12s} {'маневр':>12s} {'выход':>12s} "
                 f"{'радиус, м':>12s}"]
        for region in REGIONS:
            reps = reports[name][region]
            lines.append(f"{region:34s}" + "".join(
                f" {cell(reps, k, '.3f'):>12s}" for k in ("coverage", "coverage_straight", "coverage_entry",
                                                           "coverage_turn", "coverage_exit"))
                         + f" {cell(reps, 'radius_median', '.1f'):>12s}")
        summary.section(f"{name}: покрытие и размер строба (заданное покрытие {coverage:.0%}, все горизонты)", lines + [
            "радиус - медиана эквивалентного радиуса sqrt(площадь / pi); гибрид - среднее ± СКО по seed"])
        hs = range(1, config.HORIZON_SCANS + 1)
        lines = [f"{'строб':34s}" + "".join(f" {'h=' + str(h) + ' покр.':>11s} {'радиус':>8s}" for h in hs)]
        for region in REGIONS:
            reps = reports[name][region]
            lines.append(f"{region:34s}" + "".join(
                f" {np.mean([r[f'coverage_h{h}'] for r in reps]):11.3f} {np.mean([r[f'radius_h{h}'] for r in reps]):8.1f}"
                for h in hs))
        summary.section(f"{name}: по горизонтам (h обзоров = 2h с)", lines)
        plot_by_h(reports[name], "coverage", OUT_DIR / f"coverage_by_h_{name}{tag}.png",
                  f"Покрытие стробов, {name}", "покрытие", target=coverage)
        plot_by_h(reports[name], "radius", OUT_DIR / f"radius_by_h_{name}{tag}.png",
                  f"Размер стробов (медиана эквивалентного радиуса), {name}", "радиус, м")

    lines = [f"{'строб':34s}" + "".join(f" {'h=' + str(h):>8s}" for h in range(1, config.HORIZON_SCANS + 1))]
    for region in REGIONS:
        lines.append(f"{region:34s}" + "".join(f" {thresholds[0][region][h]:8.2f}"
                                               for h in range(1, config.HORIZON_SCANS + 1)))
    summary.section(f"Пороги (seed {seeds[0]}): квантиль e^T C^-1 e для эллипсов, радиус в м для круга", lines)

    rts = built["test"][1]
    i = int(np.argmax([len(rt.onsets) for rt in rts]))
    plot_example(rts[i], runs_cov["test"][i], fc_imm["test"][i], fc_hyb0[i], thresholds[0],
                 OUT_DIR / f"gate_example{tag}.png")
    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
