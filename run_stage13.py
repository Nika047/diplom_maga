"""
Этап 13: вес ошибкам на входе в маневр при обучении (эксперимент 6 из PLAN.md, «Этапы 8+»).

Новый компонент - взвешенная функция потерь корректора (training/trainer.weighted_mse, вес
примера передаётся через training/experiment.run_experiment(weights=...)). Важные примеры
получают вес из config.ENTRY_WEIGHTS, остальные - 1:
  «вход»                 - участок entry в момент прогноза k (как в метрике «вход»);
  «около начала маневра» - метка этапа 5 (data/windows.onset_labels): маневр начнётся в
                           пределах ONSET_AHEAD обзоров или начался не больше ONSET_BEHIND назад.
Метки берутся из разметки истины - только для весов обучения, во входы сети не идут. На val
веса те же (ранняя остановка - по взвешенному loss). Вес 1 - сохранённые веса этапов 4 и 9.
IMM+GRU с гиперпараметрами этапа 4 на двух наборах обучения (решение пользователя 24.10.2026):
набор этапа 4 и «+ все расширения» этапа 9; 3 seed. Ожидается размен: вход в маневр точнее,
прямая и средняя ошибка - не лучше.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage13.py
Смоук (рабочий ПК, нужны смоуки этапов 4 и 9): .venv\\Scripts\\python.exe run_stage13.py --smoke

Проверки: взвешенный loss - известный ответ (веса 1 = MSE, вес 2 = пример дважды); доля важных
примеров в train и val не нулевая; веса этапов 4 и 9 повторяют свои результаты; проверки
обучения из training/experiment.py.
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
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction, onset_labels  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage6a import classic_forecasts  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage9 import KINDS  # noqa: E402
from run_stage11 import reference_ade  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment, slug  # noqa: E402
from training.trainer import _mse_loss, pick_device, predict, weighted_mse  # noqa: E402
from tuning import search  # noqa: E402
from viz.filter_plots import method_color  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage13"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
TRAIN_SETS = {"stage4": "этап 4", "aug": "+ все расширения"}
TARGETS = {"entry": "вход", "onset": "около начала маневра"}
BASE_WEIGHTS = {"stage4": ("stage4", HYB, "stage4"), "aug": ("stage9", "+ все расширения", "stage9")}


def net_name(set_key: str, target: str = None, w: float = 1.0) -> str:
    return f"{HYB}, {TRAIN_SETS[set_key]}, " + ("без веса" if target is None else f"вес {TARGETS[target]} x{w:g}")


def important(ws, rts: dict, target: str) -> np.ndarray:
    """Маска важных окон (N,): rts - {flight_id: RadarTrack} с разметкой истины."""
    out = np.zeros(len(ws), dtype=bool)
    for i, fid in enumerate(ws.flight_ids):
        sel = ws.track == i
        if not sel.any():
            continue
        rt, ks = rts[fid], ws.k[sel]
        out[sel] = rt.phase[ks] == PHASE_CODE["entry"] if target == "entry" else onset_labels(rt, ks) > 0
    return out


def check_loss(summary: StageSummary) -> None:
    g = torch.Generator().manual_seed(0)
    p, y = torch.randn(6, 5, 2, generator=g), torch.randn(6, 5, 2, generator=g)
    ones = torch.cat([y.reshape(6, -1), torch.ones(6, 1)], 1)
    w = torch.ones(6, 1)
    w[2] = 2.0
    a, b = float(weighted_mse(p, ones)), float(_mse_loss(p, y))
    c = float(weighted_mse(p, torch.cat([y.reshape(6, -1), w], 1)))
    d = float(_mse_loss(torch.cat([p, p[2:3]]), torch.cat([y, y[2:3]])))
    summary.check("Взвешенный loss: известный ответ (веса 1 = MSE, вес 2 = пример дважды)",
                  abs(a - b) < 1e-5 and abs(c - d) < 1e-5, f"{a:.6f} = {b:.6f}; {c:.6f} = {d:.6f}")


def plot_tradeoff(table: dict, name: str, path: Path) -> None:
    """ADE против ошибки на входе в маневр: каждая точка - метод, среднее по seed."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, reps in table.items():
        x, y = np.mean([r["ade"] for r in reps]), np.mean([r["entry"] for r in reps])
        ax.plot(x, y, "o", ms=6, color=method_color(method))
        ax.annotate(method.replace(f"{HYB}, ", ""), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"ADE {name}, м")
    ax.set_ylabel(f"ADE на входе в маневр, {name}, м")
    ax.set_title(f"Размен средней ошибки и ошибки на входе в маневр: {name}", fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 13: вес ошибкам на входе в маневр")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 13: вес ошибкам на входе в маневр" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 13 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache9 = config.CACHE_DIR / f"stage9{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    t0 = time.time()

    check_loss(summary)
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
    copies = {kind: augment.augment(real, kind, limits)[0] for kind in KINDS}
    with search.config_override(imm):
        ws_base = dataset.load_or_build(f"train{tag}_w{window}", real + synth, cache4, window=window)
        ws_aug = dataset.concat([ws_base] + [dataset.load_or_build(f"aug_{kind}{tag}_w{window}", copies[kind], cache9,
                                                                   window=window) for kind in KINDS])
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
    train_ws = {"stage4": ws_base, "aug": ws_aug}
    # Разметка истины для весов: observe не зависит от параметров IMM
    rts = {tr.flight_id: observe(tr) for tr in real + synth + [c for cs in copies.values() for c in cs]}
    rts.update({rt.flight_id: rt for name in EVAL_SETS for rt in built[name][1]})
    masks = {(key, target): important(ws, rts, target) for key, ws in train_ws.items() for target in TARGETS}
    val_masks = {target: important(built["val"][0], rts, target) for target in TARGETS}
    shares = [f"{TRAIN_SETS[k]}, {TARGETS[t]}: {m.mean():.1%} окон train" for (k, t), m in masks.items()]
    shares += [f"val, {TARGETS[t]}: {m.mean():.1%} окон" for t, m in val_masks.items()]
    summary.check("Важные примеры есть в train и val", all(m.any() for m in masks.values())
                  and all(m.any() for m in val_masks.values()), "; ".join(shares))
    ks = {name: [np.array(sorted(built[name][0].k[built[name][0].track == i]), dtype=int)
                 for i in range(len(sets[name]))] for name in EVAL_SETS}

    table = {name: {} for name in EVAL_SETS}
    forecasts = {name: {} for name in EVAL_SETS}
    for name in EVAL_SETS:
        for method, fc in classic_forecasts(built[name][1], built[name][2], ks[name]).items():
            if method == "IMM":
                table[name][method] = [summarize_method(built[name][1], fc)]
                forecasts[name][method] = fc

    def net_forecasts(name, pred):
        ws, _, runs = built[name]
        return [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                               pred[ws.track == i]), ks[name][i]) for i, run in enumerate(runs)]

    def record(method, s_i, preds):
        for name in EVAL_SETS:
            fc = net_forecasts(name, preds[name])
            table[name].setdefault(method, []).append(summarize_method(built[name][1], fc))
            if s_i == 0:
                forecasts[name][method] = fc

    # Вес 1 - сохранённые веса этапов 4 и 9
    for set_key, (art, name_exp, ref_stage) in BASE_WEIGHTS.items():
        method = net_name(set_key)
        ps = [config.ARTIFACTS_DIR / art / f"{slug(f'{name_exp}, seed {s}')}{tag}.pt" for s in seeds]
        if not all(p.exists() for p in ps):
            summary.check(f"{method}: веса этапа {ref_stage[5:]} найдены", False,
                          f"нет {[str(p) for p in ps if not p.exists()]}")
            continue
        for s_i, path in enumerate(ps):
            model, norm, _ = load_model(path, device)
            record(method, s_i, {name: norm.y_inv(predict(model, norm.x(built[name][0].x), device))
                                 for name in EVAL_SETS})
        ref, got = reference_ade(ref_stage, name_exp, tag, best_all), table["val"][method][0]["ade"]
        summary.check(f"{method}: веса повторяют свой результат на val (этап {ref_stage[5:]})",
                      ref is not None and abs(got / ref - 1) <= 0.005,
                      f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь (seed {seeds[0]}) {got:.2f} м")

    # Взвешенное обучение
    for target in TARGETS:
        for w in config.ENTRY_WEIGHTS:
            for set_key, ws_tr in train_ws.items():
                method = net_name(set_key, target, w)
                w_tr = np.where(masks[(set_key, target)], w, 1.0)
                w_val = np.where(val_masks[target], w, 1.0)
                for s_i, seed in enumerate(seeds):
                    res = run_experiment(f"{method}, seed {seed}", "GRU", "hybrid", ws_tr, built["val"][0], params,
                                         device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                         model_kwargs=hp, overfit=s_i == 0,
                                         extra_sets={n: built[n][0] for n in EVAL_SETS[1:]}, weights=(w_tr, w_val))
                    record(method, s_i, {"val": res["pred"], **res["extra"]})

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
        main_methods = ["IMM"] + [net_name("stage4")] + [net_name("stage4", t, max(config.ENTRY_WEIGHTS))
                                                         for t in TARGETS]
        compare(built[name][1], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")
        plot_tradeoff(table[name], name, OUT_DIR / f"tradeoff_{name}{tag}.png")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
