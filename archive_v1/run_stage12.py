"""
Этап 12: доплер - радиальная скорость в модели РЛС (эксперимент 5 из PLAN.md, «Этапы 8+»).

Новый компонент - измерение радиальной скорости (radar/model.py, СКО config.RADAR_SIGMA_VR) и
его использование: в IMM и фильтре Калмана - дообновление каждой модели по радиальной
скорости (filters/imm.py, config.DOPPLER_IN_IMM); в сети - два признака обзора: радиальная
скорость и её невязка к прогнозу IMM (data/windows.py, config.DOPPLER_IN_NET). Варианты:
  без доплера (веса этапов 4 и 9, без обучения), доплер только в IMM, только в сети, в IMM и в
  сети. IMM+GRU с гиперпараметрами этапа 4 на двух наборах обучения (решение пользователя
  24.10.2026): набор этапа 4 и «+ все расширения» этапа 9; 3 seed.
Вопрос - ускоряет ли радиальная скорость реакцию на начало маневра (вход в маневр - главная
ошибка, этапы 10-11): смотрятся ошибка на входе, динамическая ошибка и вероятность модели CT
в первые обзоры маневра.

Истинная радиальная скорость считается по истине; у реальных полётов истина - GPS с
погрешностью в метры, поэтому их доплер шумнее заданного (оговорка, THESIS_NOTES.md 3.14).

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage12.py
Смоук (рабочий ПК, нужны смоуки этапов 4 и 9): .venv\\Scripts\\python.exe run_stage12.py --smoke

Проверки: известный ответ радиальной скорости; СКО шума доплера; отметки положения от
доплера не меняются; доплер только в сети не меняет прогноз IMM; на синтетике доплер в IMM
уточняет оценку скорости; окна с доплером причинные; веса этапов 4 и 9 повторяют свои
результаты; проверки обучения из training/experiment.py.
"""

import argparse
import dataclasses
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
from data import augment, generator  # noqa: E402
from data.windows import apply_correction, build_windows  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from filters.imm import IMM  # noqa: E402
from filters.models import VX, VY  # noqa: E402
from radar.model import observe, true_radial_velocity  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage6a import classic_forecasts  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage9 import KINDS  # noqa: E402
from run_stage11 import reference_ade  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment, slug  # noqa: E402
from training.trainer import pick_device, predict  # noqa: E402
from tuning import search  # noqa: E402
from viz.filter_plots import method_color  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage12"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
TRAIN_SETS = {"stage4": "этап 4", "aug": "+ все расширения"}
# вариант: (DOPPLER_IN_IMM, DOPPLER_IN_NET)
VARIANTS = {"без доплера": (False, False), "доплер в IMM": (True, False), "доплер в сети": (False, True),
            "доплер в IMM и в сети": (True, True)}
BASE = "без доплера"
# веса без доплера: (каталог, имя эксперимента без seed, этап для проверки повторения)
BASE_WEIGHTS = {"stage4": ("stage4", HYB, "stage4"), "aug": ("stage9", "+ все расширения", "stage9")}
ONSET_SCANS = 6              # вероятность CT в обзорах 0..5 от начала маневра


def vtag(variant: str) -> str:
    a, b = VARIANTS[variant]
    return f"d{int(a)}{int(b)}"


def doppler_override(variant: str) -> dict:
    a, b = VARIANTS[variant]
    return {"DOPPLER_IN_IMM": a, "DOPPLER_IN_NET": b}


def net_name(set_key: str, variant: str) -> str:
    return f"{HYB}, {TRAIN_SETS[set_key]}, {variant}"


def check_doppler(sets: dict, summary: StageSummary) -> None:
    # Известный ответ: прямолинейное движение - формула назад по трём точкам точна
    t = np.arange(30) * config.SCAN_PERIOD
    radial = np.column_stack([1000.0 + 20.0 * t, np.zeros_like(t), np.zeros_like(t)])
    across = np.column_stack([-300.0 + 20.0 * t, np.full_like(t, 1000.0), np.zeros_like(t)])
    exact = 20.0 * across[:, 0] / np.linalg.norm(across, axis=1)
    e1 = float(np.abs(true_radial_velocity(radial, np.zeros(3))[2:] - 20.0).max())
    e2 = float(np.abs(true_radial_velocity(across, np.zeros(3))[2:] - exact[2:]).max())
    summary.check("Доплер: известный ответ (движение от РЛС и мимо неё)", e1 < 1e-9 and e2 < 1e-9,
                  f"от РЛС 20 м/с: ошибка {e1:.1e}; мимо РЛС: ошибка {e2:.1e} м/с")

    tracks = sets["test"][:4]
    with search.config_override(doppler_override("доплер в IMM и в сети")):
        on = [observe(tr) for tr in tracks]
    off = [observe(tr) for tr in tracks]
    noise = np.concatenate([rt.vr - true_radial_velocity(rt.truth, rt.radar_pos) for rt in on])
    ratio = float(np.nanstd(noise) / config.RADAR_SIGMA_VR)
    summary.check("Доплер: СКО шума соответствует заданному", abs(ratio - 1.0) <= 0.05,
                  f"{ratio:.3f} заданного ({config.RADAR_SIGMA_VR} м/с), отсчётов {np.isfinite(noise).sum()}")
    same = all(np.array_equal(a.meas, b.meas) and np.array_equal(a.polar, b.polar) for a, b in zip(on, off))
    summary.check("Доплер не меняет отметки положения", same and all(rt.vr is None for rt in off),
                  f"трасс {len(tracks)}; без флагов радиальной скорости нет")

    # На синтетике (гладкая истина) доплер в IMM уточняет оценку скорости; эталон - центральная разность
    def vel_error(rts):
        out = []
        for rt in rts:
            run = IMM().run(rt)
            vt = (rt.truth[2:, :2] - rt.truth[:-2, :2]) / (2.0 * config.SCAN_PERIOD)
            e = np.linalg.norm(run.est[1:-1, [VX, VY]] - vt, axis=1)
            out.append(e[config.WARMUP_SCANS:])
        return float(np.nanmean(np.concatenate(out)))

    synth = sets["synth"][:20]
    with search.config_override(doppler_override("доплер в IMM")):
        with_d = vel_error([observe(tr) for tr in synth])
    without = vel_error([observe(tr) for tr in synth])
    summary.check("Синтетика: доплер в IMM уточняет оценку скорости", with_d < without,
                  f"средняя ошибка скорости IMM: без доплера {without:.2f}, с доплером {with_d:.2f} м/с (трасс {len(synth)})")

    # Причинность окон с доплером: признаки по началу трассы = по полной
    with search.config_override(doppler_override("доплер в IMM и в сети")):
        tr = max(sets["val"][:20], key=len)
        rt = observe(tr)
        imm = IMM()
        full = build_windows(rt, imm.run(rt), with_target=False)
        k0 = int(full["k"][len(full["k"]) // 2])
        cut = dataclasses.replace(rt, t=rt.t[:k0 + 1], truth=rt.truth[:k0 + 1], polar=rt.polar[:k0 + 1],
                                  meas=rt.meas[:k0 + 1], meas_cov=rt.meas_cov[:k0 + 1],
                                  detected=rt.detected[:k0 + 1], phase=rt.phase[:k0 + 1],
                                  onsets=[k for k in rt.onsets if k <= k0], vr=rt.vr[:k0 + 1])
        part = build_windows(cut, imm.run(cut), with_target=False)
    common = np.intersect1d(part["k"], full["k"])
    same = np.array_equal(part["x"][np.isin(part["k"], common)], full["x"][np.isin(full["k"], common)])
    summary.check("Окна с доплером причинные: признаки по началу трассы = по полной",
                  same and len(common) > 0 and full["x"].shape[-1] == part["x"].shape[-1],
                  f"трасса {tr.flight_id}, обрезка на обзоре {k0}, общих окон {len(common)}, признаков {full['x'].shape[-1]}")


def ct_response(rts: list, runs: list) -> np.ndarray:
    """Средняя вероятность модели CT в обзорах 0..ONSET_SCANS-1 от начала маневра."""
    j = config.IMM_MODELS.index("CT")
    vals = [[] for _ in range(ONSET_SCANS)]
    for rt, run in zip(rts, runs):
        for o in rt.onsets:
            for m in range(ONSET_SCANS):
                if o + m < len(rt.t) and not np.isnan(run.mu[o + m, j]):
                    vals[m].append(run.mu[o + m, j])
    return np.array([np.mean(v) if v else np.nan for v in vals])


def plot_bars(table: dict, path: Path, title: str) -> None:
    order = list(table)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, (key, name, color) in enumerate((("ade", "ADE", "#455a64"), ("entry", "вход в маневр", "#bdbdbd"))):
        vals = np.array([np.mean([r[key] for r in table[m]]) for m in order])
        errs = np.array([np.std([r[key] for r in table[m]]) for m in order])
        ax.bar(x + (j - 0.5) * 0.4, vals, 0.4, yerr=errs, capsize=3, label=name, color=color, edgecolor="black",
               lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("ошибка, м")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_ct(curves: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, curve in curves.items():
        ax.plot(np.arange(ONSET_SCANS), curve, "-o", ms=4, color=method_color(name), label=name)
    ax.set_xlabel("обзоров от начала маневра")
    ax.set_ylabel("средняя вероятность модели CT")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 12: доплер")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 12: доплер" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 12 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
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
    window, hp, train_params = search.split_params(best_all[HYB]["params"])
    params = {**train_params, "epochs": epochs, "patience": patience}

    sets = load_data(args.smoke)
    real, synth = sets["train"], sets["synth"]
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    copies = [c for kind in KINDS for c in augment.augment(real, kind, limits)[0]]
    with search.config_override(imm):
        check_doppler(sets, summary)

    # Наборы оценки во всех вариантах; обзоры оценки - без доплера (одни и те же)
    built = {}
    for variant in VARIANTS:
        with search.config_override({**imm, **doppler_override(variant)}):
            for name in EVAL_SETS:
                built[(name, variant)] = dataset.build(sets[name], window=window)
    ks = {name: [np.array(sorted(built[(name, BASE)][0].k[built[(name, BASE)][0].track == i]), dtype=int)
                 for i in range(len(sets[name]))] for name in EVAL_SETS}
    same_imm = all(np.array_equal(a.forecast, b.forecast, equal_nan=True)
                   for a, b in zip(built[("test", BASE)][2], built[("test", "доплер в сети")][2]))
    summary.check("Доплер только в сети не меняет прогноз IMM", same_imm, "test, все трассы")

    table = {name: {} for name in EVAL_SETS}
    forecasts = {name: {} for name in EVAL_SETS}
    for name in EVAL_SETS:
        rts = built[(name, BASE)][1]
        fcs = classic_forecasts(rts, built[(name, BASE)][2], ks[name])
        # KF - с параметрами по умолчанию, как «KF» без доплера; IMM - из прогонов с подобранными параметрами
        with search.config_override(doppler_override("доплер в IMM")):
            dop = classic_forecasts(built[(name, "доплер в IMM")][1], built[(name, "доплер в IMM")][2], ks[name])
        fcs.update({"KF с доплером": dop["KF"], "IMM с доплером": dop["IMM"]})
        for method, fc in fcs.items():
            table[name][method] = [summarize_method(rts, fc)]
            forecasts[name][method] = fc

    def net_forecasts(name, variant, pred):
        ws, _, runs = built[(name, variant)]
        return [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                               pred[ws.track == i]), ks[name][i]) for i, run in enumerate(runs)]

    # Без доплера - сохранённые веса этапов 4 и 9
    for set_key, (art, name_exp, ref_stage) in BASE_WEIGHTS.items():
        method = net_name(set_key, BASE)
        ps = [config.ARTIFACTS_DIR / art / f"{slug(f'{name_exp}, seed {s}')}{tag}.pt" for s in seeds]
        if not all(p.exists() for p in ps):
            summary.check(f"{method}: веса этапа {ref_stage[5:]} найдены", False, f"нет {[str(p) for p in ps if not p.exists()]}")
            continue
        for s_i, path in enumerate(ps):
            model, norm, _ = load_model(path, device)
            for name in EVAL_SETS:
                pred = norm.y_inv(predict(model, norm.x(built[(name, BASE)][0].x), device))
                fc = net_forecasts(name, BASE, pred)
                table[name].setdefault(method, []).append(summarize_method(built[(name, BASE)][1], fc))
                if s_i == 0:
                    forecasts[name][method] = fc
        ref, got = reference_ade(ref_stage, name_exp, tag, best_all), table["val"][method][0]["ade"]
        summary.check(f"{method}: веса повторяют свой результат на val (этап {ref_stage[5:]})",
                      ref is not None and abs(got / ref - 1) <= 0.005,
                      f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь (seed {seeds[0]}) {got:.2f} м")

    # С доплером - обучение заново
    for variant in [v for v in VARIANTS if v != BASE]:
        for set_key in TRAIN_SETS:
            method = net_name(set_key, variant)
            with search.config_override({**imm, **doppler_override(variant)}):
                base = dataset.load_or_build(f"train{tag}_w{window}_{vtag(variant)}", real + synth, cache, window=window)
                ws_tr = base if set_key == "stage4" else dataset.concat(
                    [base, dataset.load_or_build(f"aug_all{tag}_w{window}_{vtag(variant)}", copies, cache, window=window)])
            log.info("%s: %d окон, признаков %d", method, len(ws_tr), ws_tr.x.shape[-1])
            for s_i, seed in enumerate(seeds):
                res = run_experiment(f"{method}, seed {seed}", "GRU", "hybrid", ws_tr, built[("val", variant)][0],
                                     params, device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                     model_kwargs=hp, overfit=s_i == 0,
                                     extra_sets={n: built[(n, variant)][0] for n in EVAL_SETS[1:]})
                preds = {"val": res["pred"], **res["extra"]}
                for name in EVAL_SETS:
                    fc = net_forecasts(name, variant, preds[name])
                    table[name].setdefault(method, []).append(summarize_method(built[(name, BASE)][1], fc))
                    if s_i == 0:
                        forecasts[name][method] = fc
            del ws_tr

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    for name in EVAL_SETS:
        width = max(len(m) for m in table[name]) + 1
        lines = [f"{'метод':{width}s}" + "".join(f" {c:>11s}" for c in ("ADE", "FDE", "прямая", "вход", "маневр",
                                                                         "выход"))]
        for method, reps in table[name].items():
            lines.append(f"{method:{width}s}" + "".join(f" {cell(reps, k):>11s}"
                                                         for k in ("ade", "fde", "straight", "entry", "turn", "exit")))
        summary.section(f"{name}: ADE (м), сети - среднее ± СКО по {len(seeds)} seed", lines)
        source_section(summary, name, table[name])
        main_methods = ["IMM", "IMM с доплером"] + [net_name(s, v) for s in TRAIN_SETS
                                                     for v in (BASE, "доплер в IMM и в сети")]
        compare(built[(name, BASE)][1], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")
        plot_bars(table[name], OUT_DIR / f"ade_by_method_{name}{tag}.png",
                  f"Доплер: {name} (сети - среднее по {len(seeds)} seed)")
        curves = {m: ct_response(built[(name, v)][1], built[(name, v)][2])
                  for m, v in (("IMM", BASE), ("IMM с доплером", "доплер в IMM"))}
        summary.section(f"{name}: средняя вероятность модели CT в обзорах 0..{ONSET_SCANS - 1} от начала маневра",
                        [f"{m:16s} " + " ".join(f"{v:5.2f}" for v in c) for m, c in curves.items()])
        plot_ct(curves, OUT_DIR / f"ct_response_{name}{tag}.png", f"Реакция IMM на начало маневра: {name}")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
