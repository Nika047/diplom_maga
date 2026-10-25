"""
Этап 15: ансамбль - среднее прогнозов нескольких обученных сетей (эксперимент 8 из PLAN.md,
«Этапы 8+»).

Новый компонент - ансамбль корректоров: поправки к прогнозу IMM от нескольких сетей
усредняются (все гибриды строятся на одних окнах IMM, поправки - в одной системе курса). Сети
не обучаются: берутся сохранённые веса.
  3 seed IMM+GRU, набор этапа 4 (веса этапа 4);
  3 seed IMM+GRU, «+ все расширения» (веса этапа 9);
  6 сетей GRU - оба набора вместе;
  4 архитектуры гибрида (LSTM, GRU, Transformer, Mamba), seed первый (веса этапа 4);
  4 архитектуры x 3 seed - 12 сетей.
Рядом - каждая сеть отдельно (среднее ± СКО по участникам). Разброс прогнозов участников -
оценка неопределённости: смотрится, растёт ли ошибка вместе с разбросом.

Запуск (домашний ПК, веса - только там): .venv\\Scripts\\python.exe run_stage15.py
Смоук (рабочий ПК, нужны смоуки этапов 4 и 9): .venv\\Scripts\\python.exe run_stage15.py --smoke

Проверки: ансамбль из одной сети = эта сеть; ADE ансамбля не больше средней ADE участников
(неравенство треугольника для каждой пары); веса повторяют свои результаты на val (этапы 4, 9).
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
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import slug  # noqa: E402
from training.trainer import pick_device, predict  # noqa: E402
from tuning import search  # noqa: E402
from viz.plots import _save  # noqa: E402

SMOKE = {"seeds": (42, 43)}

STAGE = "stage15"
OUT_DIR = config.RESULTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
ARCHS = ("IMM+LSTM", "IMM+GRU", "IMM+Transformer", "IMM+Mamba")
AUG = "IMM+GRU, + все расширения"
N_SPREAD_BINS = 4            # разброс ансамбля - по квартилям


def ensembles(seeds: tuple) -> dict:
    """Имя ансамбля -> участники (метка сети, seed)."""
    n = len(seeds)
    return {
        f"ансамбль {n} seed: {HYB}, этап 4": [(HYB, s) for s in seeds],
        f"ансамбль {n} seed: {AUG}": [(AUG, s) for s in seeds],
        f"ансамбль {2 * n} GRU: этап 4 + все расширения": [(m, s) for m in (HYB, AUG) for s in seeds],
        "ансамбль 4 архитектур (seed первый)": [(a, seeds[0]) for a in ARCHS],
        f"ансамбль 4 архитектуры x {n} seed": [(a, s) for a in ARCHS for s in seeds],
    }


def reference_val(label: str, s_i: int, tag: str):
    """ADE val той же сети на своём этапе."""
    if label == AUG:
        path = config.RESULTS_DIR / "stage9" / f"metrics{tag}.json"
        return json.loads(path.read_text(encoding="utf-8"))["+ все расширения"][s_i]["val"]["ade"] if path.exists() else None
    path = config.RESULTS_DIR / "stage4" / "final_metrics.json"
    if not path.exists():
        return None
    reps = json.loads(path.read_text(encoding="utf-8"))["val"].get(label, [])
    return reps[s_i]["ade"] if s_i < len(reps) else None


def ranks(x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x))
    out[np.argsort(x, kind="stable")] = np.arange(len(x))
    return out


def plot_bars(table: dict, path: Path, title: str) -> None:
    order = list(table)
    x = np.arange(len(order))
    vals = np.array([np.mean([r["ade"] for r in table[m]]) for m in order])
    errs = np.array([np.std([r["ade"] for r in table[m]]) for m in order])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#455a64" if m.startswith("ансамбль") else "#bdbdbd" for m in order]
    ax.bar(x, vals, 0.6, yerr=errs, capsize=3, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(vals.min() * 0.9, None)
    ax.set_ylabel("ADE, м (серые - отдельные сети, среднее ± СКО по seed)")
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_spread(rows: dict, path: Path, title: str) -> None:
    """rows: {ансамбль: (средний разброс по квартилям, средняя ошибка по квартилям)}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (sp, err) in rows.items():
        ax.plot(sp, err, "o-", ms=5, label=name)
    ax.set_xlabel(f"разброс прогнозов участников на {config.HORIZON_SCANS * config.SCAN_PERIOD:.0f} с, м (квартили)")
    ax.set_ylabel("ошибка ансамбля на том же горизонте, м")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 15: ансамбль")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 15: ансамбль" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    t0 = time.time()

    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window = search.split_params(best_all[HYB]["params"])[0]

    paths = {label: [config.ARTIFACTS_DIR / "stage4" / f"{slug(f'{label}, seed {s}')}{tag}.pt" for s in seeds]
             for label in ARCHS}
    paths[AUG] = [config.ARTIFACTS_DIR / "stage9" / f"{slug(f'+ все расширения, seed {s}')}{tag}.pt" for s in seeds]
    found = {label: ps for label, ps in paths.items() if all(p.exists() for p in ps)}
    missing = sorted(set(paths) - set(found))
    summary.check("Веса участников найдены (этапы 4 и 9)", HYB in found and (args.smoke or not missing),
                  f"найдены {sorted(found)}" + (f", нет {missing}" if missing else ""))

    sets = load_data(args.smoke)
    with search.config_override(imm):
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
    ks = {name: [np.array(sorted(built[name][0].k[built[name][0].track == i]), dtype=int)
                 for i in range(len(sets[name]))] for name in EVAL_SETS}

    def forecasts_of(name, corr):
        ws, _, runs = built[name]
        return [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                               corr[ws.track == i]), ks[name][i]) for i, run in enumerate(runs)]

    # Поправки всех участников, м
    corr = {}                                       # (метка, seed) -> {набор: (N, H, 2)}
    table = {name: {} for name in EVAL_SETS}
    for label, ps in found.items():
        for s_i, (seed, path) in enumerate(zip(seeds, ps)):
            model, norm, mode = load_model(path, device)
            corr[(label, seed)] = {name: norm.y_inv(predict(model, norm.x(built[name][0].x), device))
                                   for name in EVAL_SETS}
            for name in EVAL_SETS:
                table[name].setdefault(label, []).append(summarize_method(built[name][1],
                                                                          forecasts_of(name, corr[(label, seed)][name])))
            ref, got = reference_val(label, s_i, tag), table["val"][label][-1]["ade"]
            summary.check(f"{label}, seed {seed}: веса повторяют свой результат на val",
                          ref is not None and abs(got / ref - 1) <= 0.005,
                          f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь {got:.2f} м")

    # Ансамбль из одной сети = эта сеть
    one = (HYB, seeds[0])
    single = np.mean([corr[one]["test"]], axis=0)
    summary.check("Ансамбль из одной сети = эта сеть", np.array_equal(single, corr[one]["test"]), f"{HYB}, seed {seeds[0]}")

    # Ансамбли
    groups = {g: m for g, m in ensembles(seeds).items() if all(k in corr for k in m)}
    skipped = sorted(set(ensembles(seeds)) - set(groups))
    jensen, forecasts, spread_rows = [], {name: {} for name in EVAL_SETS}, {name: {} for name in EVAL_SETS}
    for name in EVAL_SETS:
        forecasts[name]["IMM"] = [mask_forecast(r.forecast, k) for r, k in zip(built[name][2], ks[name])]
        table[name]["IMM"] = [summarize_method(built[name][1], forecasts[name]["IMM"])]
        for label in (HYB, AUG):
            if (label, seeds[0]) in corr:
                forecasts[name][label] = forecasts_of(name, corr[(label, seeds[0])][name])
    for group, members in groups.items():
        for name in EVAL_SETS:
            stack = np.stack([corr[m][name] for m in members])               # (M, N, H, 2)
            mean = stack.mean(axis=0)
            fc = forecasts_of(name, mean)
            rep = summarize_method(built[name][1], fc)
            table[name][group] = [rep]
            forecasts[name][group] = fc
            member_mean = np.mean([summarize_method(built[name][1], forecasts_of(name, corr[m][name]))["ade"]
                                   for m in members])
            jensen.append(rep["ade"] <= member_mean + 1e-9)
            # Разброс участников на последнем горизонте и ошибка ансамбля там же
            y = built[name][0].y[:, -1]
            spread = np.linalg.norm(stack[:, :, -1] - mean[None, :, -1], axis=-1).mean(axis=0)
            err = np.linalg.norm(mean[:, -1] - y, axis=-1)
            rho = float(np.corrcoef(ranks(spread), ranks(err))[0, 1]) if len(err) > 2 else float("nan")
            edges = np.quantile(spread, np.linspace(0, 1, N_SPREAD_BINS + 1))
            bins = np.clip(np.searchsorted(edges, spread, side="right") - 1, 0, N_SPREAD_BINS - 1)
            sp = [float(spread[bins == b].mean()) for b in range(N_SPREAD_BINS)]
            er = [float(err[bins == b].mean()) for b in range(N_SPREAD_BINS)]
            spread_rows[name][group] = (sp, er, rho)
    summary.check("ADE ансамбля не больше средней ADE участников (все ансамбли и наборы)", bool(jensen) and all(jensen),
                  f"сравнений {len(jensen)}" + (f"; пропущены ансамбли без весов: {skipped}" if skipped else ""))

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
        summary.section(f"{name}: ADE (м); отдельные сети - среднее ± СКО по {len(seeds)} seed", lines)
        source_section(summary, name, table[name])
        summary.section(f"{name}: разброс участников и ошибка ансамбля на горизонте {config.HORIZON_SCANS} обзоров "
                        f"(квартили разброса, м; rho - ранговая корреляция)", [
                            f"{g:44s} rho {rho:5.2f}; разброс " + " / ".join(f"{v:5.1f}" for v in sp)
                            + "; ошибка " + " / ".join(f"{v:6.1f}" for v in er)
                            for g, (sp, er, rho) in spread_rows[name].items()])
        main_methods = ["IMM", HYB, AUG] + [g for g in groups if "seed:" in g or " x " in g]
        compare(built[name][1], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (отдельные сети - seed {seeds[0]})", OUT_DIR, f"{name}{tag}")
        plot_bars(table[name], OUT_DIR / f"ade_ensembles_{name}{tag}.png", f"Ансамбли и отдельные сети: {name}")
        plot_spread({g: v[:2] for g, v in spread_rows[name].items()}, OUT_DIR / f"spread_{name}{tag}.png",
                    f"Разброс ансамбля как оценка ошибки: {name}")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
