"""
Этап 6а: пропуски отметок (вариант C).

РЛС обнаруживает цель в обзоре с вероятностью P_d (config.PD_LEVELS). При пропуске
IMM делает только прогноз, линейная экстраполяция берёт последние полученные
отметки, а в окнах сетей удерживается последняя полученная отметка (data/windows.held);
пропуск сеть видит по нулевой невязке и NIS.

Новый компонент - условие наблюдения с пропусками; модели и гиперпараметры - этапа 4
(IMM подобранный, IMM+GRU и GRU без IMM - лучшие параметры генетического подбора).
Для каждого P_d на одних и тех же обзорах:
  - CV, KF, IMM;
  - сети, обученные без пропусков (P_d = 1): устойчивость без переобучения;
  - сети, переобученные при этом P_d.
Сравнение на val, test и holdout, ADE и вход в маневр; сети - среднее ± СКО по seed.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage6a.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage6a.py --smoke

Проверки: признаки окна не зависят от отметок в обзорах с пропуском (утечки нет),
доля пропусков соответствует P_d, при P_d = 1 гибрид повторяет этап 4, проверки
обучения из training/experiment.py.
"""

import argparse
import copy
import json
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
from data.windows import apply_correction, apply_e2e, build_windows, build_windows_e2e  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from filters.extrapolation import cv_extrapolate  # noqa: E402
from filters.imm import IMM  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from tuning import search  # noqa: E402
from viz.filter_plots import method_color  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43), "levels": (1.0, 0.8)}

STAGE = "stage6a"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
NETS = (("IMM+GRU", "GRU", "hybrid"), ("GRU (без IMM)", "GRU", "e2e"))
CLASSIC = ("CV", "KF", "IMM")


def pd_tag(pd: float) -> str:
    return f"pd{round(pd * 100):03d}"


def check_no_leak(tracks: list, window: int, summary: StageSummary) -> None:
    """Отметки в обзорах с пропуском заменяются мусором - признаки окон не должны измениться."""
    worst, n_missed = 0.0, 0
    imm = IMM()
    for tr in tracks:
        rt = observe(tr)
        missed = ~rt.detected
        if not missed.any():
            continue
        n_missed += int(missed.sum())
        run = imm.run(rt)
        w = build_windows(rt, run, window=window)
        e = build_windows_e2e(rt, w["k"], window=window)
        bad = copy.deepcopy(rt)
        bad.meas[missed] += 1e4
        bad.polar[missed] += 1e3
        w2 = build_windows(bad, imm.run(bad), window=window)
        e2 = build_windows_e2e(bad, w2["k"], window=window)
        if len(w["x"]) != len(w2["x"]):
            worst = float("inf")
            break
        worst = max(worst, float(np.abs(w["x"] - w2["x"]).max(initial=0.0)),
                    float(np.abs(e["x"] - e2["x"]).max(initial=0.0)))
    summary.check("Признаки не зависят от отметок в обзорах с пропуском", n_missed > 0 and worst == 0.0,
                  f"пропусков в проверке {n_missed}, наибольшее изменение признаков {worst:.3g}")


def classic_forecasts(rts: list, runs: list, ks: list) -> dict:
    kf = IMM(("CV",), mu0=(1.0,))
    return {"CV": [mask_forecast(cv_extrapolate(rt), k) for rt, k in zip(rts, ks)],
            "KF": [mask_forecast(kf.run(rt).forecast, k) for rt, k in zip(rts, ks)],
            "IMM": [mask_forecast(r.forecast, k) for r, k in zip(runs, ks)]}


def net_forecasts(built_set: tuple, pred: np.ndarray, mode: str, ks: list) -> list:
    ws, rts, runs = built_set
    out = []
    for i, (rt, run) in enumerate(zip(rts, runs)):
        sel = ws.track == i
        fc = (apply_correction(run.forecast, ws.k[sel], ws.psi[sel], pred[sel]) if mode == "hybrid"
              else apply_e2e(rt, ws.k[sel], ws.psi_e2e[sel], pred[sel]))
        out.append(mask_forecast(fc, ks[i]))
    return out


def plot_levels(table: dict, levels: tuple, key: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, by_pd in table.items():
        vals = np.array([[r[key] for r in by_pd[pd]] for pd in levels])   # (уровни, seed)
        base = method.rsplit(" (обучен", 1)[0]
        retrained = "этом P_d" in method
        ax.errorbar(levels, vals.mean(axis=1), yerr=vals.std(axis=1), fmt="s" if retrained else "o", ms=5,
                    capsize=3, color=method_color(base), ls=":" if retrained else "-",
                    mfc="white" if retrained else None, label=method)
    ax.set_xlabel("вероятность обнаружения P_d")
    ax.set_ylabel("ADE, м" if key == "ade" else f"ADE ({key}), м")
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 6а: пропуски отметок")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 6а: пропуски отметок" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 6а - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    levels = SMOKE["levels"] if args.smoke else config.PD_LEVELS
    t0 = time.time()

    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or any(label not in best_all for label, _, _ in NETS):
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {[n[0] for n in NETS]} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    hp = {label: search.split_params(best_all[label]["params"]) for label, _, _ in NETS}
    windows = {w for w, _, _ in hp.values()}
    if len(windows) != 1:
        summary.check("Одно окно у всех сетей", False, f"окна {windows}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    window = windows.pop()
    sets = load_data(args.smoke)
    tracks_tr = sets["train"] + sets["synth"]

    # Окна оценки для всех уровней P_d и общие обзоры
    built, ks = {}, {}
    for pd in levels:
        with search.config_override({**imm, "RADAR_PD": pd}):
            if pd < 1.0 and pd == min(levels):
                check_no_leak(sets["test"][:4], window, summary)
            for name in EVAL_SETS:
                built[(name, pd)] = dataset.build(sets[name], window=window)
                ws = built[(name, pd)][0]
                ks[(name, pd)] = [np.array(sorted(ws.k[ws.track == i]), dtype=int) for i in range(len(sets[name]))]
        rts = built[("test", pd)][1]
        miss = 1.0 - float(np.mean(np.concatenate([rt.detected for rt in rts])))
        summary.check(f"P_d = {pd:.2f}: доля пропусков на test", abs(miss - (1.0 - pd)) <= config.PD_MISS_TOL,
                      f"{miss:.3f} при ожидаемой {1.0 - pd:.3f}")

    # {набор: {метод: {P_d: [отчёт по seed]}}}
    table = {name: {} for name in EVAL_SETS}
    forecasts = {(name, pd): {} for name in EVAL_SETS for pd in levels}
    for pd in levels:
        for name in EVAL_SETS:
            fcs = classic_forecasts(built[(name, pd)][1], built[(name, pd)][2], ks[(name, pd)])
            for method, fc in fcs.items():
                table[name].setdefault(method, {})[pd] = [summarize_method(built[(name, pd)][1], fc)]
            forecasts[(name, pd)].update(fcs)

    # Сети: обучение при каждом P_d; модель с P_d = 1 оценивается и на всех остальных
    for pd in levels:
        with search.config_override({**imm, "RADAR_PD": pd}):
            ws_tr = dataset.load_or_build(f"train{tag}_w{window}_{pd_tag(pd)}", tracks_tr, cache, window=window)
        ws_val = built[("val", pd)][0]
        eval_pds = levels if pd == 1.0 else (pd,)
        for label, corrector, mode in NETS:
            win, model_kwargs, train_params = hp[label]
            params = {**train_params, "epochs": epochs, "patience": patience}
            name_tr = f"{label} (обучен при P_d = 1)" if pd == 1.0 else f"{label} (обучен при этом P_d)"
            for s_i, seed in enumerate(seeds):
                extra = {f"{n}|{e}": built[(n, e)][0] for n in EVAL_SETS for e in eval_pds if (n, e) != ("val", pd)}
                res = run_experiment(f"{label}, P_d = {pd:.2f}, seed {seed}", corrector, mode, ws_tr, ws_val, params,
                                     device, summary, log, args.smoke, OUT_DIR, ART, window=win, seed=seed,
                                     model_kwargs=model_kwargs, overfit=(s_i == 0 and pd == 1.0), extra_sets=extra)
                preds = {**res["extra"], f"val|{pd}": res["pred"]}
                for e in eval_pds:
                    method = name_tr
                    for n in EVAL_SETS:
                        fc = net_forecasts(built[(n, e)], preds[f"{n}|{e}"], mode, ks[(n, e)])
                        table[n].setdefault(method, {}).setdefault(e, []).append(
                            summarize_method(built[(n, e)][1], fc))
                        if s_i == 0:
                            forecasts[(n, e)][method] = fc
                if pd == 1.0 and s_i == 0 and label == "IMM+GRU":
                    ga, got = best_all[label]["ade_val"], table["val"][name_tr][1.0][0]["ade"]
                    summary.check("При P_d = 1 гибрид повторяет этап 4", abs(got / ga - 1) <= 0.05,
                                  f"ADE val на этапе 4 {ga:.2f} м, здесь (seed {seed}) {got:.2f} м")
        del ws_tr

    # При P_d = 1 «обучен при этом P_d» - та же модель, что «обучен при P_d = 1»
    for name in EVAL_SETS:
        for label, _, _ in NETS:
            same = table[name].get(f"{label} (обучен при этом P_d)")
            if same is not None:
                same[1.0] = table[name][f"{label} (обучен при P_d = 1)"][1.0]

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    for name in EVAL_SETS:
        for key, what in (("ade", "ADE"), ("entry", "ADE на входе в маневр"), ("turn", "ADE на маневре")):
            lines = [f"{'метод':36s}" + "".join(f" {'P_d = ' + format(pd, '.1f'):>12s}" for pd in levels)]
            for method, by_pd in table[name].items():
                lines.append(f"{method:36s}" + "".join(
                    f" {cell(by_pd[pd], key) if pd in by_pd else '-':>12s}" for pd in levels))
            summary.section(f"{name}: {what} (м), сети - среднее ± СКО по {len(seeds)} seed", lines)
        lines = [f"{'метод':36s}" + "".join(f" {'P_d = ' + format(pd, '.1f'):>12s}" for pd in levels)]
        for method, by_pd in table[name].items():
            base = np.mean([r["ade"] for r in by_pd[1.0]])
            lines.append(f"{method:36s}" + "".join(
                f" {np.mean([r['ade'] for r in by_pd[pd]]) / base - 1:>+12.1%}" if pd in by_pd else f" {'-':>12s}"
                for pd in levels))
        summary.section(f"{name}: рост ADE относительно P_d = 1 того же метода", lines)
        for pd in levels:
            source_section(summary, f"{name}, P_d = {pd:.1f}",
                           {m: by_pd[pd] for m, by_pd in table[name].items() if pd in by_pd})
        plot_levels(table[name], levels, "ade", OUT_DIR / f"ade_vs_pd_{name}{tag}.png",
                    f"ADE при пропусках отметок, {name}")
        plot_levels(table[name], levels, "entry", OUT_DIR / f"entry_vs_pd_{name}{tag}.png",
                    f"ADE на входе в маневр при пропусках, {name}")

    low = min(levels)
    compare(built[("test", low)][1], forecasts[("test", low)], summary, f"test, P_d = {low:.2f} (seed {seeds[0]})",
            OUT_DIR, f"test_{pd_tag(low)}{tag}")
    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(
        {name: {m: {str(pd): reps for pd, reps in by_pd.items()} for m, by_pd in t.items()}
         for name, t in table.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
