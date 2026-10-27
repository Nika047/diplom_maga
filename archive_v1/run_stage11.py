"""
Этап 11: оракул - сколько ошибки остаётся от непредсказуемости будущего движения (эксперимент 4
из PLAN.md, «Этапы 8+»).

Новый компонент - прогноз по точному текущему состоянию цели (filters/oracle.py): оракул CV
(точная скорость, движение прямо), оракул CT (точные скорость и скорость разворота, дуга) и
задним числом лучший из них в каждом обзоре (идеальный выбор модели). Рядом на тех же обзорах:
  - CV, KF, IMM при штатном шуме; IMM почти без шума (config.NOISE_SCALES, наименьший);
  - IMM+GRU при штатном шуме на двух наборах обучения (решение пользователя 24.10.2026):
    веса этапа 4 и «+ все расширения» этапа 9;
  - IMM+GRU почти без шума: веса этапа 10, переобученные при этом шуме.
Сети не обучаются: берутся сохранённые веса (3 seed), поэтому нужны результаты этапов 4, 9, 10.
Разложение ошибки гибрида: шум РЛС (штатный шум -> почти без шума), остаток метода (почти без
шума -> лучший оракул) и непредсказуемость (лучший оракул). Оракул - не строгая нижняя граница:
сеть, которая по истории предугадывает маневр, может оказаться точнее постоянных моделей.

Запуск (домашний ПК, веса - только там): .venv\\Scripts\\python.exe run_stage11.py
Смоук (рабочий ПК, нужны смоуки этапов 4, 9, 10): .venv\\Scripts\\python.exe run_stage11.py --smoke

Проверки: на прямой оба оракула точны, на окружности точен оракул CT; прогноз оракула в
обзоре k не зависит от истины после k; лучший из оракулов не хуже каждого из них; веса этапов
4, 9, 10 найдены и повторяют свои результаты на val.
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
from data.windows import apply_correction  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from filters.oracle import best_of, oracle_forecast  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage6a import classic_forecasts  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage10 import noise_override  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import slug  # noqa: E402
from training.trainer import pick_device, predict  # noqa: E402
from tuning import search  # noqa: E402
from viz.plots import _save  # noqa: E402

SMOKE = {"seeds": (42, 43)}

STAGE = "stage11"
OUT_DIR = config.RESULTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
LOW = min(config.NOISE_SCALES)
ORACLES = ("оракул CV", "оракул CT", "оракул: лучший из CV и CT (задним числом)")
# метка: (каталог весов, имя эксперимента без seed, уровень шума, откуда ADE val для проверки)
NETS = {
    f"{HYB}, этап 4": ("stage4", HYB, 1.0, "stage4"),
    f"{HYB}, + все расширения": ("stage9", "+ все расширения", 1.0, "stage9"),
    f"{HYB}, этап 4, почти без шума": ("stage10", f"{HYB}, этап 4, шум x{LOW:g}", LOW, "stage10"),
    f"{HYB}, + все расширения, почти без шума": ("stage10", f"{HYB}, + все расширения, шум x{LOW:g}", LOW,
                                                 "stage10"),
}
LOW_IMM = "IMM, почти без шума"


def circle_track(t: np.ndarray) -> np.ndarray:
    w, v, psi0 = np.radians(10.0), 20.0, 0.3
    return np.column_stack([v / w * (np.cos(psi0) - np.cos(psi0 + w * t)), v / w * (np.sin(psi0 + w * t) - np.sin(psi0)),
                            100.0 + 0.5 * t])


def check_oracle(summary: StageSummary) -> None:
    t = np.arange(60) * config.SCAN_PERIOD
    circ, line = circle_track(t), np.column_stack([3.0 * t, 15.0 * t, np.full(len(t), 100.0)])

    def worst(fc, tr):
        h = config.HORIZON_SCANS
        return max(float(np.abs(fc[2:len(tr) - j, j - 1] - tr[2 + j:]).max()) for j in range(1, h + 1))

    e = {"CT, окружность": worst(oracle_forecast(circ, True), circ),
         "CV, прямая": worst(oracle_forecast(line, False), line), "CT, прямая": worst(oracle_forecast(line, True), line)}
    cv_circ = worst(oracle_forecast(circ, False), circ)
    summary.check("Оракул: известный ответ (прямая - оба точны, окружность - точен CT)",
                  max(e.values()) < 1e-6 and cv_circ > 1.0,
                  ", ".join(f"{k} {v:.1e} м" for k, v in e.items()) + f"; CV на окружности {cv_circ:.0f} м")
    changed = circ.copy()
    changed[30:] += 1000.0
    diff = np.nanmax(np.abs(oracle_forecast(circ, True)[:30] - oracle_forecast(changed, True)[:30]))
    summary.check("Оракул причинный: прогноз в k не зависит от истины после k", diff == 0.0,
                  f"наибольшее изменение прогноза до обзора 30 при сдвиге истины после него: {diff:.1e} м")


def reference_ade(stage: str, name: str, tag: str, best_all: dict):
    """ADE val (seed первый) тех же весов на своём этапе."""
    if stage == "stage4":
        return best_all.get(HYB, {}).get("ade_val")
    path = config.RESULTS_DIR / stage / f"metrics{tag}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if stage == "stage9":
        return data[name][0]["val"]["ade"]
    method = name.replace(f", шум x{LOW:g}", "") + " (обучен при этом шуме)"
    return data["val"][method][f"{LOW}"][0]["ade"]


def plot_decomposition(table: dict, path: Path, title: str) -> None:
    """ADE и ADE на входе в маневр по методам - от IMM к оракулу."""
    order = ["IMM", f"{HYB}, этап 4", f"{HYB}, + все расширения", LOW_IMM, f"{HYB}, этап 4, почти без шума",
             f"{HYB}, + все расширения, почти без шума", *ORACLES]
    order = [m for m in order if m in table]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (key, name) in enumerate((("ade", "ADE"), ("entry", "вход в маневр"))):
        vals = np.array([np.mean([r[key] for r in table[m]]) for m in order])
        errs = np.array([np.std([r[key] for r in table[m]]) for m in order])
        ax.bar(x + (j - 0.5) * 0.4, vals, 0.4, yerr=errs, capsize=3, label=name,
               color="#455a64" if key == "ade" else "#bdbdbd", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("ошибка, м")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 11: оракул")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 11: оракул" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    t0 = time.time()

    check_oracle(summary)
    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window = search.split_params(best_all[HYB]["params"])[0]
    paths = {label: [config.ARTIFACTS_DIR / art / f"{slug(f'{name}, seed {s}')}{tag}.pt" for s in seeds]
             for label, (art, name, _, _) in NETS.items()}
    found = {label: ps for label, ps in paths.items() if all(p.exists() for p in ps)}
    missing = sorted(set(paths) - set(found))
    summary.check("Веса этапов 4, 9, 10 найдены", not missing,
                  f"найдены {len(found)} из {len(paths)}" + (f", нет: {missing}" if missing else ""))

    # Окна и прогоны IMM при штатном шуме и почти без шума; обзоры оценки - как в итоге этапа 4
    sets = load_data(args.smoke)
    built = {}
    for s in (1.0, LOW):
        with search.config_override({**imm, **noise_override(s)}):
            for name in EVAL_SETS:
                built[(name, s)] = dataset.build(sets[name], window=window)
    ks = {name: [np.array(sorted(built[(name, 1.0)][0].k[built[(name, 1.0)][0].track == i]), dtype=int)
                 for i in range(len(sets[name]))] for name in EVAL_SETS}

    table = {name: {} for name in EVAL_SETS}          # {набор: {метод: [отчёт по seed]}}
    forecasts = {name: {} for name in EVAL_SETS}
    for name in EVAL_SETS:
        rts = built[(name, 1.0)][1]
        fcs = classic_forecasts(rts, built[(name, 1.0)][2], ks[name])
        fcs[LOW_IMM] = [mask_forecast(r.forecast, k) for r, k in zip(built[(name, LOW)][2], ks[name])]
        cv_o = [oracle_forecast(rt.truth, False) for rt in rts]
        ct_o = [oracle_forecast(rt.truth, True) for rt in rts]
        best = [best_of(rt.truth, [a, b]) for rt, a, b in zip(rts, cv_o, ct_o)]
        for method, fc in zip(ORACLES, (cv_o, ct_o, best)):
            fcs[method] = [mask_forecast(f, k) for f, k in zip(fc, ks[name])]
        for method, fc in fcs.items():
            table[name][method] = [summarize_method(rts, fc)]
            forecasts[name][method] = fc
    a, b, c = (table["test"][m][0]["ade"] for m in ORACLES)
    summary.check("Лучший из оракулов не хуже каждого из них (test)", c <= min(a, b) + 1e-9,
                  f"ADE: оракул CV {a:.2f}, CT {b:.2f}, лучший {c:.2f} м")

    # Сети: сохранённые веса, прогноз на своём уровне шума
    for label, ps in found.items():
        art, name_exp, s, ref_stage = NETS[label]
        for s_i, path in enumerate(ps):
            model, norm, mode = load_model(path, device)
            for name in EVAL_SETS:
                ws, rts, runs = built[(name, s)]
                pred = norm.y_inv(predict(model, norm.x(ws.x), device))
                fc = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                     pred[ws.track == i]), ks[name][i]) for i, run in enumerate(runs)]
                table[name].setdefault(label, []).append(summarize_method(built[(name, 1.0)][1], fc))
                if s_i == 0:
                    forecasts[name][label] = fc
        ref, got = reference_ade(ref_stage, name_exp, tag, best_all), table["val"][label][0]["ade"]
        summary.check(f"{label}: веса повторяют свой результат на val (этап {ref_stage[5:]})",
                      ref is not None and abs(got / ref - 1) <= 0.005,
                      f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь (seed {seeds[0]}) {got:.2f} м")

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    def mean(name, method, key):
        return float(np.mean([r[key] for r in table[name][method]]))

    for name in EVAL_SETS:
        width = max(len(m) for m in table[name]) + 1
        lines = [f"{'метод':{width}s}" + "".join(f" {c:>11s}" for c in ("ADE", "FDE", "прямая", "вход", "маневр",
                                                                         "выход"))]
        for method, reps in table[name].items():
            lines.append(f"{method:{width}s}" + "".join(f" {cell(reps, k):>11s}"
                                                         for k in ("ade", "fde", "straight", "entry", "turn", "exit")))
        summary.section(f"{name}: ADE (м), сети - среднее ± СКО по {len(seeds)} seed", lines)
        source_section(summary, name, table[name])
        # Разложение ошибки гибрида: шум РЛС, остаток метода, непредсказуемость
        lines = [f"{'набор обучения':24s} {'участок':>8s} {'штатный':>8s} {'без шума':>9s} {'оракул':>8s} "
                 f"{'шум':>7s} {'метод':>7s} {'непредск.':>10s}"]
        for set_name in ("этап 4", "+ все расширения"):
            nom, low = f"{HYB}, {set_name}", f"{HYB}, {set_name}, почти без шума"
            if nom not in table[name] or low not in table[name]:
                continue
            for key, part in (("ade", "все"), ("entry", "вход"), ("turn", "маневр")):
                a, b, c = mean(name, nom, key), mean(name, low, key), mean(name, ORACLES[2], key)
                lines.append(f"{set_name:24s} {part:>8s} {a:8.1f} {b:9.1f} {c:8.1f} {(a - b) / a:>7.0%} "
                             f"{(b - c) / a:>7.0%} {c / a:>10.0%}")
        summary.section(f"{name}: разложение ошибки IMM+GRU (м и доли от ошибки при штатном шуме; оракул - лучший "
                        f"из CV и CT; «метод» < 0 - сеть почти без шума точнее оракула)", lines)
        compare(built[(name, 1.0)][1], forecasts[name], summary, f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")
        plot_decomposition(table[name], OUT_DIR / f"decomposition_{name}{tag}.png",
                           f"От IMM к оракулу: {name} (сети - среднее по {len(seeds)} seed)")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
