"""
Этап 10: вклад шума отметок РЛС (эксперимент 3 из PLAN.md, «Этапы 8+»).

Новый компонент - условие наблюдения с уменьшенным шумом: СКО дальности, азимута и угла места
умножаются на config.NOISE_SCALES (1 - как везде, 0,01 - почти без шума). Случайные числа РЛС
те же (radar/model.observe), меняется только масштаб, поэтому уровни сравнимы обзор в обзор.
Для каждого уровня на одних и тех же обзорах:
  - CV, KF, IMM (параметры IMM подобраны на этапе 4 при штатном шуме; ковариация отметки
    пересчитывается по уменьшенному шуму);
  - IMM+GRU с гиперпараметрами этапа 4 на двух наборах обучения (решение пользователя
    24.10.2026): набор этапа 4 и «+ все расширения» этапа 9; обученные при штатном шуме
    (устойчивость без переобучения) и переобученные при этом уровне.
Разница ошибок при штатном шуме и почти без шума - вклад шума отметок; остаток - ошибка от
непредсказуемости движения и от самих методов.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage10.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage10.py --smoke

Проверки: шум отметок масштабируется точно (ошибка в полярных координатах при множителе s
равна s ошибки при штатном шуме, истина и положение РЛС те же), эмпирическое СКО дальности
соответствует уровню; при штатном шуме гибрид повторяет этап 4 (набор этапа 4) и этап 9
(«+ все расширения»); проверки обучения из training/experiment.py.
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
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from data import augment, generator  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from radar.model import observe, to_polar  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage6a import classic_forecasts  # noqa: E402
from run_stage9 import KINDS  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from tuning import search  # noqa: E402
from viz.filter_plots import METHOD_COLORS  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43), "levels": (1.0, 0.01)}

STAGE = "stage10"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
TRAIN_SETS = {"stage4": "этап 4", "aug": "+ все расширения"}
NET_COLORS = {"stage4": METHOD_COLORS.get(HYB, "tab:green"), "aug": "tab:purple"}
CLASSIC = ("CV", "KF", "IMM")


def noise_tag(s: float) -> str:
    return f"n{round(s * 100):03d}"


def noise_override(s: float) -> dict:
    return {"RADAR_SIGMA_R": config.RADAR_SIGMA_R * s, "RADAR_SIGMA_AZ_DEG": config.RADAR_SIGMA_AZ_DEG * s,
            "RADAR_SIGMA_EL_DEG": config.RADAR_SIGMA_EL_DEG * s}


def net_name(set_key: str, nominal: bool) -> str:
    return f"{HYB}, {TRAIN_SETS[set_key]} " + ("(обучен при штатном шуме)" if nominal else "(обучен при этом шуме)")


def check_noise(tracks: list, levels: tuple, summary: StageSummary) -> None:
    """Ошибка отметки в полярных координатах при множителе s = s * ошибка при штатном шуме."""
    worst, ratios = 0.0, {}
    base = []
    for tr in tracks:
        rt = observe(tr)
        base.append((rt, rt.polar - to_polar(rt.truth - rt.radar_pos)))
    for s in levels:
        errs = []
        with search.config_override(noise_override(s)):
            for tr, (rt0, e0) in zip(tracks, base):
                rt = observe(tr)
                e = rt.polar - to_polar(rt.truth - rt.radar_pos)
                worst = max(worst, float(np.abs(e - s * e0).max()), float(np.abs(rt.truth - rt0.truth).max()),
                            float(np.abs(rt.radar_pos - rt0.radar_pos).max()))
                errs.append(e[:, 0])
        ratios[s] = float(np.std(np.concatenate(errs)) / (config.RADAR_SIGMA_R * s))
    summary.check("Шум отметок масштабируется точно: ошибка при s = s * ошибка при штатном шуме", worst < 1e-6,
                  f"наибольшее расхождение {worst:.1e} (м и рад)")
    summary.check("Эмпирическое СКО дальности соответствует уровню", all(abs(r - 1) <= 0.05 for r in ratios.values()),
                  ", ".join(f"s = {s:g}: {r:.3f} заданного" for s, r in ratios.items()))


def plot_levels(table: dict, levels: tuple, key: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method, by_level in table.items():
        vals = np.array([[r[key] for r in by_level[s]] for s in levels])     # (уровни, seed)
        set_key = next((k for k, v in TRAIN_SETS.items() if f", {v} " in method), None)
        retrained = "этом шуме" in method
        color = NET_COLORS[set_key] if set_key else METHOD_COLORS.get(method)
        ax.errorbar(levels, vals.mean(axis=1), yerr=vals.std(axis=1), fmt="s" if retrained else "o", ms=5,
                    capsize=3, color=color, ls=":" if retrained else "-", mfc="white" if retrained else None,
                    label=method)
    ax.set_xscale("log")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{s:g}" for s in levels])
    ax.invert_xaxis()
    ax.set_xlabel("множитель СКО шума отметок (1 - штатный шум)")
    ax.set_ylabel("ADE, м" if key == "ade" else f"ADE ({key}), м")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 10: вклад шума отметок")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 10: вклад шума отметок" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 10 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache9 = config.CACHE_DIR / f"stage9{tag}"
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    levels = SMOKE["levels"] if args.smoke else config.NOISE_SCALES
    t0 = time.time()

    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all or levels[0] != 1.0:
        summary.check("Результаты этапа 4 найдены, первый уровень - штатный шум", False,
                      f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window, hp, train_params = search.split_params(best_all[HYB]["params"])
    params = {**train_params, "epochs": epochs, "patience": patience}

    sets = load_data(args.smoke)
    real, synth = sets["train"], sets["synth"]
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    copies = {kind: augment.augment(real, kind, limits)[0] for kind in KINDS}
    check_noise(sets["test"][:4], levels, summary)

    # Окна оценки на всех уровнях и общие обзоры
    built, ks = {}, {}
    for s in levels:
        with search.config_override({**imm, **noise_override(s)}):
            for name in EVAL_SETS:
                built[(name, s)] = dataset.build(sets[name], window=window)
                ws = built[(name, s)][0]
                ks[(name, s)] = [np.array(sorted(ws.k[ws.track == i]), dtype=int) for i in range(len(sets[name]))]

    # {набор: {метод: {уровень: [отчёт по seed]}}}
    table = {name: {} for name in EVAL_SETS}
    forecasts = {(name, s): {} for name in EVAL_SETS for s in levels}
    for s in levels:
        for name in EVAL_SETS:
            fcs = classic_forecasts(built[(name, s)][1], built[(name, s)][2], ks[(name, s)])
            for method, fc in fcs.items():
                table[name].setdefault(method, {})[s] = [summarize_method(built[(name, s)][1], fc)]
            forecasts[(name, s)].update(fcs)

    def train_windows(s: float, set_key: str):
        """Окна обучения при уровне s; при штатном шуме - из кэшей этапов 4 и 9 (тот же набор, тот же порядок)."""
        with search.config_override({**imm, **noise_override(s)}):
            if s == 1.0:
                base = dataset.load_or_build(f"train{tag}_w{window}", real + synth, cache4, window=window)
                if set_key == "stage4":
                    return base
                aug = [dataset.load_or_build(f"aug_{kind}{tag}_w{window}", copies[kind], cache9, window=window)
                       for kind in KINDS]
                return dataset.concat([base, *aug])
            base = dataset.load_or_build(f"train{tag}_w{window}_{noise_tag(s)}", real + synth, cache, window=window)
            if set_key == "stage4":
                return base
            aug = dataset.load_or_build(f"aug_all{tag}_w{window}_{noise_tag(s)}",
                                        [c for kind in KINDS for c in copies[kind]], cache, window=window)
            return dataset.concat([base, aug])

    def reference_ade(set_key: str):
        """ADE val (seed первый) того же набора при штатном шуме на этапе 4 или 9."""
        if set_key == "stage4":
            return best_all[HYB]["ade_val"], "этапе 4"
        path = config.RESULTS_DIR / "stage9" / f"metrics{tag}.json"
        if not path.exists():
            return None, "этапе 9"
        return json.loads(path.read_text(encoding="utf-8"))[TRAIN_SETS["aug"]][0]["val"]["ade"], "этапе 9"

    # Сети: обучение на каждом уровне; модель штатного шума оценивается на всех уровнях
    for s in levels:
        ws_val = built[("val", s)][0]
        eval_levels = levels if s == 1.0 else (s,)
        for set_key in TRAIN_SETS:
            ws_tr = train_windows(s, set_key)
            log.info("%s, шум x%g: %d окон", TRAIN_SETS[set_key], s, len(ws_tr))
            method = net_name(set_key, s == 1.0)
            for s_i, seed in enumerate(seeds):
                extra = {f"{n}|{e}": built[(n, e)][0] for n in EVAL_SETS for e in eval_levels if (n, e) != ("val", s)}
                with search.config_override(imm):
                    res = run_experiment(f"{HYB}, {TRAIN_SETS[set_key]}, шум x{s:g}, seed {seed}", "GRU", "hybrid",
                                         ws_tr, ws_val, params, device, summary, log, args.smoke, OUT_DIR, ART,
                                         window=window, seed=seed, model_kwargs=hp, overfit=(s_i == 0 and s == 1.0),
                                         extra_sets=extra)
                preds = {**res["extra"], f"val|{s}": res["pred"]}
                for e in eval_levels:
                    for n in EVAL_SETS:
                        ws, rts, runs = built[(n, e)]
                        fc = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                             preds[f"{n}|{e}"][ws.track == i]), ks[(n, e)][i])
                              for i, run in enumerate(runs)]
                        table[n].setdefault(method, {}).setdefault(e, []).append(summarize_method(rts, fc))
                        if s_i == 0:
                            forecasts[(n, e)][method] = fc
                if s == 1.0 and s_i == 0:
                    ref, where = reference_ade(set_key)
                    got = table["val"][method][1.0][0]["ade"]
                    summary.check(f"{TRAIN_SETS[set_key]}: при штатном шуме гибрид повторяет результат на {where}",
                                  ref is not None and abs(got / ref - 1) <= 0.05,
                                  f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь (seed {seed}) {got:.2f} м")
            del ws_tr

    # При штатном шуме «обучен при этом шуме» - та же модель, что «обучен при штатном шуме»
    for name in EVAL_SETS:
        for set_key in TRAIN_SETS:
            same = table[name].get(net_name(set_key, False))
            if same is not None:
                same[1.0] = table[name][net_name(set_key, True)][1.0]

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    width = max(len(m) for m in table[EVAL_SETS[0]]) + 1
    low = min(levels)
    for name in EVAL_SETS:
        for key, what in (("ade", "ADE"), ("entry", "ADE на входе в маневр"), ("turn", "ADE на маневре")):
            lines = [f"{'метод':{width}s}" + "".join(f" {'шум x' + format(s, 'g'):>12s}" for s in levels)]
            for method, by_level in table[name].items():
                lines.append(f"{method:{width}s}" + "".join(
                    f" {cell(by_level[s], key) if s in by_level else '-':>12s}" for s in levels))
            summary.section(f"{name}: {what} (м), сети - среднее ± СКО по {len(seeds)} seed", lines)
        lines = [f"{'метод':{width}s} {'ADE':>8s} {'вход':>8s} {'маневр':>8s}"]
        for method, by_level in table[name].items():
            if low not in by_level:
                continue
            share = [1.0 - np.mean([r[k] for r in by_level[low]]) / np.mean([r[k] for r in by_level[1.0]])
                     for k in ("ade", "entry", "turn")]
            lines.append(f"{method:{width}s}" + "".join(f" {v:>8.1%}" for v in share))
        summary.section(f"{name}: доля ошибки от шума отметок - 1 - ADE(шум x{low:g}) / ADE(штатный шум)", lines)
        for s in levels:
            source_section(summary, f"{name}, шум x{s:g}",
                           {m: by_level[s] for m, by_level in table[name].items() if s in by_level})
        plot_levels(table[name], levels, "ade", OUT_DIR / f"ade_vs_noise_{name}{tag}.png", f"ADE и шум отметок, {name}")
        plot_levels(table[name], levels, "entry", OUT_DIR / f"entry_vs_noise_{name}{tag}.png",
                    f"ADE на входе в маневр и шум отметок, {name}")

    compare(built[("test", low)][1], forecasts[("test", low)], summary, f"test, шум x{low:g} (seed {seeds[0]})",
            OUT_DIR, f"test_{noise_tag(low)}{tag}")
    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(
        {name: {m: {str(s): reps for s, reps in by_level.items()} for m, by_level in t.items()}
         for name, t in table.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
