"""
Этап 16: модульная схема - распознаватель маневра и два эксперта (эксперимент 9 из PLAN.md,
«Этапы 8+», решение пользователя 23.10.2026; смесь экспертов, как у Fraser [2]).

Новый компонент - схема из трёх отдельно обученных сетей с гиперпараметрами IMM+GRU этапа 4:
  распознаватель - GRU-классификатор (models/hybrids.ModeGRU): будет ли в горизонте прогноза
                   поворот (data/windows.turn_class > 0, то есть «влево» или «вправо»);
  эксперт «прямая» и эксперт «поворот» - корректоры GRU, каждый обучен только на своих окнах
                   (класс - по истине, только для разбиения обучения).
Прогноз:
  жёсткое переключение - эксперт по вероятности распознавателя (порог MODULAR_THRESHOLD);
  мягкая смесь         - поправки экспертов с весами p и 1 - p;
  верный эксперт задним числом - эксперт по истинному классу: предел схемы при идеальном
                   распознавателе (не прогноз).
Рядом - один корректор (веса этапов 4 и 9). Два набора обучения (решение пользователя
24.10.2026): этап 4 и «+ все расширения» этапа 9; 3 seed. Ожидание (PLAN.md): схема упрётся в
то же запаздывание распознавания на входе в маневр (этапы 11, 14).

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage16.py
Смоук (рабочий ПК, нужны смоуки этапов 4 и 9): .venv\\Scripts\\python.exe run_stage16.py --smoke

Проверки: окна экспертов не пересекаются и покрывают весь train; распознаватель до обучения
даёт 0,5; смесь при p = 0 и 1 совпадает с экспертами; веса этапов 4 и 9 повторяют свои
результаты; проверки обучения (overfit, «обучился», train loss падает).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import augment, generator  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from models.hybrids import ModeGRU, bce_loss  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage9 import KINDS  # noqa: E402
from run_stage11 import reference_ade  # noqa: E402
from run_stage14 import classes  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment, slug  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from tuning import search  # noqa: E402
from viz import train_plots  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage16"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
TRAIN_SETS = {"stage4": "этап 4", "aug": "+ все расширения"}
OUTPUTS = {"hard": "жёсткое переключение", "soft": "мягкая смесь", "oracle": "верный эксперт задним числом"}
EXPERTS = {"straight": "прямая", "turn": "поворот"}
BASE_WEIGHTS = {"stage4": ("stage4", HYB, "stage4"), "aug": ("stage9", "+ все расширения", "stage9")}


def name_of(set_key: str, output: str = None) -> str:
    if output is None:
        return f"{HYB}, {TRAIN_SETS[set_key]}, один корректор"
    return f"модульная, {TRAIN_SETS[set_key]}: {OUTPUTS[output]}"


def mix(p: np.ndarray, straight: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """Мягкая смесь поправок экспертов (N, H, 2) с вероятностью поворота p (N,)."""
    return p[:, None, None] * turn + (1.0 - p[:, None, None]) * straight


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 16: модульная схема")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 16: модульная схема" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 16 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache9 = config.CACHE_DIR / f"stage9{tag}"
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

    a = np.random.default_rng(0).normal(size=(5, config.HORIZON_SCANS, 2))
    b = np.random.default_rng(1).normal(size=(5, config.HORIZON_SCANS, 2))
    summary.check("Смесь при p = 0 и p = 1 совпадает с экспертами",
                  np.array_equal(mix(np.zeros(5), a, b), a) and np.array_equal(mix(np.ones(5), a, b), b),
                  "мягкая смесь поправок «прямая» / «поворот»")

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
    rts = {tr.flight_id: observe(tr) for tr in real + synth + [c for cs in copies.values() for c in cs]}
    rts.update({rt.flight_id: rt for name in EVAL_SETS for rt in built[name][1]})
    turn_tr = {key: classes(ws, rts) > 0 for key, ws in train_ws.items()}
    turn_ev = {name: classes(built[name][0], rts) > 0 for name in EVAL_SETS}
    parts = {key: (dataset.subset(ws, ~turn_tr[key]), dataset.subset(ws, turn_tr[key])) for key, ws in train_ws.items()}
    ok = all(len(s) + len(t) == len(train_ws[k]) and len(s) > 0 and len(t) > 0 for k, (s, t) in parts.items())
    summary.check("Окна экспертов не пересекаются и покрывают весь train", ok, "; ".join(
        f"{TRAIN_SETS[k]}: прямая {len(s)}, поворот {len(t)} ({len(t) / len(train_ws[k]):.1%})"
        for k, (s, t) in parts.items()))
    ks = {name: [np.array(sorted(built[name][0].k[built[name][0].track == i]), dtype=int)
                 for i in range(len(sets[name]))] for name in EVAL_SETS}

    table = {name: {} for name in EVAL_SETS}
    forecasts = {name: {} for name in EVAL_SETS}
    for name in EVAL_SETS:
        fc = [mask_forecast(r.forecast, k) for r, k in zip(built[name][2], ks[name])]
        table[name]["IMM"] = [summarize_method(built[name][1], fc)]
        forecasts[name]["IMM"] = fc

    def record(method, s_i, corr_by_set):
        for name in EVAL_SETS:
            ws, rts_ev, runs = built[name]
            fc = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                 corr_by_set[name][ws.track == i]), ks[name][i])
                  for i, run in enumerate(runs)]
            table[name].setdefault(method, []).append(summarize_method(rts_ev, fc))
            if s_i == 0:
                forecasts[name][method] = fc

    # Один корректор - сохранённые веса этапов 4 и 9
    for set_key, (art, name_exp, ref_stage) in BASE_WEIGHTS.items():
        method = name_of(set_key)
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

    # Модульная схема
    rec_lines = []
    ws_val = built["val"][0]
    val_parts = (dataset.subset(ws_val, ~turn_ev["val"]), dataset.subset(ws_val, turn_ev["val"]))
    extra = {f"{n}_all": built[n][0] for n in EVAL_SETS}
    for set_key, ws_tr in train_ws.items():
        norm = dataset.Normalizer.fit(ws_tr)
        x_tr, lab_tr = norm.x(ws_tr.x), turn_tr[set_key].astype(np.float32)[:, None]
        x_ev = {name: norm.x(built[name][0].x) for name in EVAL_SETS}
        lab_val = turn_ev["val"].astype(np.float32)[:, None]
        for s_i, seed in enumerate(seeds):
            key = f"распознаватель, {TRAIN_SETS[set_key]}, seed {seed}"
            rng = make_rng("training", seed)
            if s_i == 0:
                idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
                torch.manual_seed(seed)
                first, last = overfit_one_batch(ModeGRU(x_tr.shape[-1], **hp), x_tr[idx], lab_tr[idx],
                                                config.OVERFIT_STEPS, 3e-3, device, loss_fn=bce_loss)
                summary.check(f"распознаватель, {TRAIN_SETS[set_key]}: overfit one batch",
                              last < config.OVERFIT_TARGET * first, f"loss {first:.4f} -> {last:.4f}")
                rng = make_rng("training", seed)
            torch.manual_seed(seed)
            clf = ModeGRU(x_tr.shape[-1], **hp)
            if s_i == 0:
                p0 = 1.0 / (1.0 + np.exp(-predict(clf, x_ev["val"][:256], device)[:, 0]))
                summary.check("Распознаватель до обучения даёт вероятность 0,5", np.abs(p0 - 0.5).max() < 1e-6,
                              f"max |p - 0,5| {np.abs(p0 - 0.5).max():.1e}")
            hist, _ = train(clf, x_tr, lab_tr, x_ev["val"], lab_val, params, device, rng, loss_fn=bce_loss)
            summary.check(f"{key}: классификатор обучился", hist["best_val_loss"] < hist["val_loss_initial"],
                          f"val BCE {hist['val_loss_initial']:.4f} -> {hist['best_val_loss']:.4f}")
            train_plots.plot_history(hist, OUT_DIR / f"loss_recognizer_{set_key}_{seed}{tag}.png", f"Обучение: {key}")
            torch.save({"state_dict": clf.state_dict(), "normalizer": norm.as_dict(), "seed": seed, "model_kwargs": hp},
                       ART / f"recognizer_{set_key}_seed_{seed}{tag}.pt")
            prob = {name: 1.0 / (1.0 + np.exp(-predict(clf, x_ev[name], device)[:, 0])) for name in EVAL_SETS}
            for name in EVAL_SETS:
                rts_ev = built[name][1]
                ws = built[name][0]
                entry = np.array([rts_ev[t].phase[k] == PHASE_CODE["entry"] for t, k in zip(ws.track, ws.k)], dtype=bool)
                hit = (prob[name] >= config.MODULAR_THRESHOLD) == turn_ev[name]
                rec_lines.append(
                    f"{name:8s} {TRAIN_SETS[set_key]:18s} seed {seed}: AP {metrics.average_precision(prob[name], turn_ev[name]):.3f}, "
                    f"верно {hit.mean():.1%}, на входе в маневр {hit[entry].mean() if entry.any() else float('nan'):.1%}, "
                    f"доля поворота {turn_ev[name].mean():.1%}")

            corr = {}
            for j, ex in enumerate(EXPERTS):                 # j: 0 - прямая, 1 - поворот (как в parts и val_parts)
                res = run_experiment(f"эксперт «{EXPERTS[ex]}», {TRAIN_SETS[set_key]}, seed {seed}", "GRU", "hybrid",
                                     parts[set_key][j], val_parts[j], params, device, summary, log, args.smoke, OUT_DIR, ART,
                                     window=window, seed=seed, model_kwargs=hp, overfit=s_i == 0, extra_sets=extra)
                corr[ex] = {name: res["extra"][f"{name}_all"] for name in EVAL_SETS}
            outs = {o: {} for o in OUTPUTS}
            for name in EVAL_SETS:
                s, t = corr["straight"][name], corr["turn"][name]
                outs["hard"][name] = np.where((prob[name] >= config.MODULAR_THRESHOLD)[:, None, None], t, s)
                outs["soft"][name] = mix(prob[name], s, t)
                outs["oracle"][name] = np.where(turn_ev[name][:, None, None], t, s)
            for o in OUTPUTS:
                record(name_of(set_key, o), s_i, outs[o])

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
        summary.section(f"{name}: ADE (м), сети - среднее ± СКО по {len(seeds)} seed; «верный эксперт задним "
                        f"числом» - предел схемы при идеальном распознавателе, не прогноз", lines)
        source_section(summary, name, table[name])
        main_methods = ["IMM", name_of("stage4")] + [name_of("stage4", o) for o in OUTPUTS]
        compare(built[name][1], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")
    summary.section(f"Распознаватель «в горизонте - поворот»: средняя точность (AP) и доля верных ответов при пороге "
                    f"{config.MODULAR_THRESHOLD}", rec_lines)

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
