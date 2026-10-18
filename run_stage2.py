"""
Этап 2: конвейер корректора на GRU (гибрид IMM + GRU).

1. Проверки с известным ответом: поворот в систему курса, причинность окон,
   нулевая поправка = IMM, смоук модели (формы, нулевой выход, градиенты).
2. Окна train (реальные train + синтетика train) и val (реальные val), кэш cache/stage2/.
3. Проверка разбиения, overfit one batch.
4. Обучение с ранней остановкой; веса в artifacts/stage2/, история в results/stage2/.
5. Сравнение IMM и IMM+GRU на val: метрики, участки, динамическая ошибка (научный
   результат, не блокирует); рисунки и сводка.

Запуск:
  домашний ПК (GPU):  .venv\\Scripts\\python.exe run_stage2.py
  рабочий ПК (смоук): .venv\\Scripts\\python.exe run_stage2.py --smoke
--smoke: подвыборка трасс и 3 эпохи - проверка конвейера, не результат.
--rebuild: пересобрать окна, даже если кэш подходит.
"""

import argparse
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import splits as splits_mod  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction, build_windows, n_features, to_global, to_local  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from filters.imm import IMM  # noqa: E402
from models import build_corrector  # noqa: E402
from training import dataset  # noqa: E402
from training.trainer import mse, overfit_one_batch, pick_device, predict, train  # noqa: E402
from viz import filter_plots, train_plots  # noqa: E402

# Настройки, зависящие от устройства
FULL = {"batch": 512, "epochs": 80, "lr": 1e-3, "patience": 10, "weight_decay": 1e-4}
SMOKE = {"batch": 256, "epochs": 3, "lr": 1e-3, "patience": 3, "weight_decay": 1e-4}
SMOKE_REAL_STEP = 8      # смоук: каждый 8-й реальный кусок train
SMOKE_SYNTH_STEP = 25    # и каждая 25-я синтетическая траектория

STAGE = "stage2"
CORRECTOR = "GRU"
HYBRID = f"IMM+{CORRECTOR}"
OUT_DIR = config.RESULTS_DIR / STAGE
CACHE = config.CACHE_DIR / STAGE
ART = config.PROJECT_DIR / "artifacts" / STAGE
CACHE0 = config.CACHE_DIR / "stage0"
TURN_CODES = (PHASE_CODE["entry"], PHASE_CODE["turn"])


def self_tests(summary: StageSummary, val_tracks: list) -> None:
    # Поворот: курс на восток, вектор на восток - это «вдоль курса»
    v = np.array([[1.0, 0.0]])
    loc = to_local(v, np.array([np.pi / 2]))
    back = to_global(loc, np.array([np.pi / 2]))
    summary.check("Система курса: известный ответ", np.allclose(loc, [[0.0, 1.0]]) and np.allclose(back, v),
                  f"восток при курсе 90 град -> {loc.round(6).tolist()}, обратно {back.round(6).tolist()}")

    # Причинность: признаки по началу трассы совпадают с признаками полной трассы
    from radar.model import observe
    tr = max(val_tracks, key=len)
    rt = observe(tr)
    imm = IMM()
    full = build_windows(rt, imm.run(rt), with_target=False)
    k0 = int(full["k"][len(full["k"]) // 2])
    cut = dataclasses.replace(rt, t=rt.t[:k0 + 1], truth=rt.truth[:k0 + 1], polar=rt.polar[:k0 + 1],
                              meas=rt.meas[:k0 + 1], meas_cov=rt.meas_cov[:k0 + 1],
                              detected=rt.detected[:k0 + 1], phase=rt.phase[:k0 + 1],
                              onsets=[k for k in rt.onsets if k <= k0])
    part = build_windows(cut, imm.run(cut), with_target=False)
    common = np.intersect1d(part["k"], full["k"])
    same = np.array_equal(part["x"][np.isin(part["k"], common)], full["x"][np.isin(full["k"], common)])
    summary.check("Окна причинные: признаки по началу трассы = по полной", same and len(common) > 0,
                  f"трасса {tr.flight_id}, обрезка на обзоре {k0}, общих окон {len(common)}")

    # Нулевая поправка - ровно прогноз IMM
    run = imm.run(rt)
    w = build_windows(rt, run)
    hyb = apply_correction(run.forecast, w["k"], w["psi"], np.zeros_like(w["y"]))
    a = metrics.summarize(metrics.forecast_pairs(rt, run.forecast))
    b = metrics.summarize(metrics.forecast_pairs(rt, hyb))
    summary.check("Нулевая поправка = IMM", np.array_equal(hyb, run.forecast, equal_nan=True) and a == b,
                  f"ADE IMM {a['ade']:.3f} м, гибрида с нулевой поправкой {b['ade']:.3f} м")

    # Смоук модели
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector(CORRECTOR, n_features(len(config.IMM_MODELS)), config.HORIZON_SCANS)
    x = torch.randn(8, config.WINDOW_SCANS, n_features(len(config.IMM_MODELS)))
    out = model(x)
    loss = ((out - torch.ones_like(out)) ** 2).mean()
    loss.backward()
    grads = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    n_params = sum(p.numel() for p in model.parameters())
    summary.check("Модель: формы, нулевой выход до обучения, градиенты",
                  tuple(out.shape) == (8, config.HORIZON_SCANS, 2) and float(out.abs().max()) == 0.0 and grads > 0,
                  f"выход {tuple(out.shape)}, max|выход| {float(out.abs().max())}, сумма |градиентов| {grads:.3f}, "
                  f"параметров {n_params}")


def evaluate(model, norm, val_tracks: list, device, summary: StageSummary) -> dict:
    ws, rts, runs = dataset.build(val_tracks)
    corr = norm.y_inv(predict(model, norm.x(ws.x), device))
    imm_fc, hyb_fc = [], []
    for i, (rt, run) in enumerate(zip(rts, runs)):
        sel = ws.track == i
        imm_fc.append(run.forecast)
        hyb_fc.append(apply_correction(run.forecast, ws.k[sel], ws.psi[sel], corr[sel]))
    report = {}
    lines = [f"{'метод':8s} {'MAE':>7s} {'RMSE':>7s} {'ADE':>7s} {'FDE':>7s} {'R2':>6s} {'пар':>7s}  "
             f"ADE по горизонтам, м"]
    phase_lines = [f"{'метод':8s} " + " ".join(f"{p:>9s}" for p in config.PHASES if p != "slow")]
    curves, levels = {}, {}
    dyn_lines = [f"{'метод':8s} {'h':>2s} {'прямая':>7s} {'пик':>7s} {'при m':>6s} {'площадь':>8s}"]
    for name, fcs in (("IMM", imm_fc), (HYBRID, hyb_fc)):
        pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)])
        s = metrics.summarize(pairs)
        by_h = metrics.ade_by_horizon(pairs)
        by_phase = {p: metrics.summarize(metrics.select(pairs, pairs["phase"] == c))["ade"]
                    for p, c in PHASE_CODE.items()}
        maneuver = metrics.summarize(metrics.select(pairs, np.isin(pairs["phase"], TURN_CODES)))
        report[name] = {"overall": s, "by_horizon": by_h.tolist(), "by_phase": by_phase, "maneuver": maneuver}
        lines.append(f"{name:8s} {s['mae']:7.1f} {s['rmse']:7.1f} {s['ade']:7.1f} {s['fde']:7.1f} "
                     f"{s['r2']:6.3f} {s['n']:7d}  " + " ".join(f"{v:6.1f}" for v in by_h))
        phase_lines.append(f"{name:8s} " + " ".join(f"{by_phase[p]:9.1f}" for p in config.PHASES if p != "slow"))
        for h in (1, config.HORIZON_SCANS):
            ms, curve, counts = metrics.dynamic_curve(rts, fcs, h)
            level = metrics.straight_level(pairs, h)
            st = metrics.dynamic_stats(ms, curve, level)
            curves[(name, h)], levels[(name, h)] = (ms, curve, counts), level
            report[name][f"dynamic_h{h}"] = {"level": level, **st}
            dyn_lines.append(f"{name:8s} {h:2d} {level:7.1f} {st['peak']:7.1f} {st['peak_m']:6.0f} {st['area']:8.1f}")
    summary.section("val: IMM против гибрида, по горизонтали (м)", lines)
    summary.section("val: ADE по участкам (м)", phase_lines)
    summary.section("val: динамическая ошибка (м)", dyn_lines)
    filter_plots.plot_dynamic_curves(curves, levels, OUT_DIR / "dynamic_error_val.png",
                                     f"Динамическая ошибка: IMM и {HYBRID}, val")
    filter_plots.plot_by_phase({m: report[m]["by_phase"] for m in report}, OUT_DIR / "ade_by_phase_val.png",
                               f"Ошибка по участкам: IMM и {HYBRID}, val")
    filter_plots.plot_by_horizon({m: report[m]["by_horizon"] for m in report}, OUT_DIR / "ade_by_horizon_val.png",
                                 f"Ошибка по горизонтам: IMM и {HYBRID}, val")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 2: корректор GRU")
    parser.add_argument("--smoke", action="store_true", help="подвыборка и 3 эпохи (проверка конвейера)")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать окна")
    args = parser.parse_args()

    log = setup_logging(STAGE + ("_smoke" if args.smoke else ""))
    summary = StageSummary("Этап 2: корректор GRU" + (" (СМОУК: подвыборка, не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Полное обучение - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    params = SMOKE if args.smoke else FULL
    torch.manual_seed(config.RANDOM_SEED)

    with open(CACHE0 / "tracks_real.pkl", "rb") as fh:
        real = pickle.load(fh)
    with open(CACHE0 / "tracks_synth_train.pkl", "rb") as fh:
        synth = pickle.load(fh)
    splits = splits_mod.load_splits(CACHE0 / "splits.json")
    real_train = [tr for tr in real if splits.get(tr.log_file) == "train"]
    val_tracks = [tr for tr in real if splits.get(tr.log_file) == "val"]
    if args.smoke:
        real_train, synth = real_train[::SMOKE_REAL_STEP], synth[::SMOKE_SYNTH_STEP]

    log.info("1. Проверки с известным ответом")
    self_tests(summary, val_tracks)

    log.info("2. Окна")
    t0 = time.time()
    train_name = "train_smoke" if args.smoke else "train"
    ws_tr = dataset.load_or_build(train_name, real_train + synth, CACHE, args.rebuild)
    ws_val = dataset.load_or_build("val", val_tracks, CACHE, args.rebuild)
    n_real = sum(1 for i in ws_tr.track if i < len(real_train))
    summary.section("Данные", [
        f"train: {len(real_train)} реальных кусков + {len(synth)} синтетических, {len(ws_tr)} окон "
        f"(из них реальных {n_real})",
        f"val: {len(val_tracks)} реальных кусков, {len(ws_val)} окон",
        f"окно {config.WINDOW_SCANS} обзоров, признаков {ws_tr.x.shape[-1]}, горизонт {config.HORIZON_SCANS}",
        f"подготовка окон: {time.time() - t0:.1f} с",
    ])
    train_logs = {tr.log_file for tr in real_train}
    val_logs = {tr.log_file for tr in val_tracks}
    summary.check("Разбиение: логи train и val не пересекаются", not (train_logs & val_logs) and len(ws_val) > 0,
                  f"общих логов {len(train_logs & val_logs)}, окон val {len(ws_val)}")

    norm = dataset.Normalizer.fit(ws_tr)
    x_tr, y_tr = norm.x(ws_tr.x), norm.y(ws_tr.y)
    x_val, y_val = norm.x(ws_val.x), norm.y(ws_val.y)

    log.info("3. Overfit one batch")
    rng = make_rng("splits")  # только для выбора батча и перемешивания, разбиение уже сделано
    batch_idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector(CORRECTOR, x_tr.shape[-1], config.HORIZON_SCANS)
    first, last = overfit_one_batch(model, x_tr[batch_idx], y_tr[batch_idx], config.OVERFIT_STEPS, 3e-3, device)
    summary.check("Overfit one batch", last < config.OVERFIT_TARGET * first,
                  f"loss {first:.4f} -> {last:.4f} за {config.OVERFIT_STEPS} шагов "
                  f"(нужно ниже {config.OVERFIT_TARGET:.0%} начального)")

    log.info("4. Обучение (%s)", "смоук" if args.smoke else "полное")
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector(CORRECTOR, x_tr.shape[-1], config.HORIZON_SCANS)
    t0 = time.time()
    history, _ = train(model, x_tr, y_tr, x_val, y_val, params, device, rng)
    history["train_seconds"] = time.time() - t0
    history["device"] = str(device)
    history["params"] = params
    epochs = history["epochs"]
    summary.check("Модель обучилась: val loss ниже, чем у нулевой поправки (IMM)",
                  history["best_val_loss"] < history["val_loss_initial"],
                  f"val loss {history['val_loss_initial']:.4f} -> лучший {history['best_val_loss']:.4f}")
    summary.check("Train loss падает", len(epochs) >= 2 and epochs[-1]["train_loss"] < epochs[0]["train_loss"],
                  f"эпоха 1: {epochs[0]['train_loss']:.4f}, эпоха {epochs[-1]['epoch']}: {epochs[-1]['train_loss']:.4f}")
    last_ep = epochs[-1]
    summary.section("Обучение", [
        f"устройство {device}, эпох {len(epochs)}, время {history['train_seconds']:.0f} с, параметры {params}",
        f"последняя эпоха: train {last_ep['train_loss']:.4f}, val {last_ep['val_loss']:.4f}; "
        f"разрыв val/train {last_ep['val_loss'] / max(last_ep['train_loss'], 1e-9):.2f}",
    ])
    ART.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "corrector": CORRECTOR,
                "signature": dataset.signature(ws_tr.flight_ids), "smoke": args.smoke},
               ART / f"{CORRECTOR.lower()}{'_smoke' if args.smoke else ''}.pt")
    (OUT_DIR / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    train_plots.plot_history(history, OUT_DIR / "loss.png", f"Обучение {HYBRID}" + (" (смоук)" if args.smoke else ""))

    log.info("5. Сравнение с IMM на val")
    report = evaluate(model, norm, val_tracks, device, summary)
    a, b = report["IMM"]["overall"]["ade"], report[HYBRID]["overall"]["ade"]
    am, bm = report["IMM"]["maneuver"]["ade"], report[HYBRID]["maneuver"]["ade"]
    summary.section("Научный результат (не блокирует этап)", [
        f"ADE val: IMM {a:.1f} м, {HYBRID} {b:.1f} м ({(b - a) / a:+.1%})",
        f"ADE val на входе и маневре: IMM {am:.1f} м, {HYBRID} {bm:.1f} м ({(bm - am) / am:+.1%})",
    ] + (["СМОУК: числа не для диплома"] if args.smoke else []))
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.write(OUT_DIR / ("summary_smoke.txt" if args.smoke else "summary.txt"))
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
