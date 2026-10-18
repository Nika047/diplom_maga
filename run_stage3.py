"""
Этап 3: сравнение архитектур корректора и сквозных моделей без IMM.

Гибриды IMM + корректор: LSTM, GRU, Transformer, Mamba. Сквозные модели (только
отметки РЛС, без IMM): GRU и Mamba - показывают, помогает ли IMM нейросети.
У всех одинаковые данные, порядок батчей, параметры обучения и пары (k, h) оценки.

1. Проверки с известным ответом: причинность окон без IMM, нулевое смещение =
   «последняя отметка», причинность Mamba, смоук всех моделей.
2. Окна train и val (кэш cache/stage2/ общий с этапом 2).
3. Для каждой модели: overfit one batch, обучение, проверка «модель обучилась».
4. Сравнение на val (научный результат, не блокирует), время, число параметров,
   рисунки, сводка.

Запуск:
  домашний ПК (GPU):  .venv\\Scripts\\python.exe run_stage3.py
  рабочий ПК (смоук): .venv\\Scripts\\python.exe run_stage3.py --smoke
--only GRU,Mamba - обучить только перечисленные эксперименты (по имени корректора).
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
from data.windows import N_FEATURES_E2E, apply_correction, apply_e2e, build_windows, build_windows_e2e  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from filters.imm import IMM  # noqa: E402
from models import CORRECTORS, build_corrector  # noqa: E402
from radar.model import observe  # noqa: E402
from training import dataset  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from viz import train_plots  # noqa: E402

# Настройки, зависящие от устройства (одинаковые для всех моделей)
FULL = {"batch": 512, "epochs": 80, "lr": 1e-3, "patience": 10, "weight_decay": 1e-4}
SMOKE = {"batch": 256, "epochs": 2, "lr": 1e-3, "patience": 2, "weight_decay": 1e-4}
SMOKE_REAL_STEP = 8
SMOKE_SYNTH_STEP = 25

# (имя метода, корректор, режим окон)
EXPERIMENTS = [("IMM+LSTM", "LSTM", "hybrid"), ("IMM+GRU", "GRU", "hybrid"),
               ("IMM+Transformer", "Transformer", "hybrid"), ("IMM+Mamba", "Mamba", "hybrid"),
               ("GRU (без IMM)", "GRU", "e2e"), ("Mamba (без IMM)", "Mamba", "e2e")]

STAGE = "stage3"
OUT_DIR = config.RESULTS_DIR / STAGE
CACHE = config.CACHE_DIR / "stage2"
ART = config.PROJECT_DIR / "artifacts" / STAGE
CACHE0 = config.CACHE_DIR / "stage0"


def self_tests(summary: StageSummary, val_tracks: list) -> None:
    tr = max(val_tracks, key=len)
    rt = observe(tr)
    imm = IMM()
    run = imm.run(rt)
    w = build_windows(rt, run)
    e = build_windows_e2e(rt, w["k"])

    # Нулевое смещение - прогноз «последняя отметка»
    fc = apply_e2e(rt, e["k"], e["psi"], np.zeros_like(e["y"]))
    expect = np.broadcast_to(rt.meas[e["k"]][:, None, :], fc[e["k"]].shape)
    summary.check("Без IMM: нулевое смещение = последняя отметка", np.allclose(fc[e["k"]], expect),
                  f"обзоров {len(e['k'])}, макс. отличие {np.abs(fc[e['k']] - expect).max():.2e} м")

    # Причинность окон без IMM
    k0 = int(e["k"][len(e["k"]) // 2])
    cut = dataclasses.replace(rt, t=rt.t[:k0 + 1], truth=rt.truth[:k0 + 1], polar=rt.polar[:k0 + 1],
                              meas=rt.meas[:k0 + 1], meas_cov=rt.meas_cov[:k0 + 1],
                              detected=rt.detected[:k0 + 1], phase=rt.phase[:k0 + 1],
                              onsets=[k for k in rt.onsets if k <= k0])
    ks = e["k"][e["k"] <= k0]
    part = build_windows_e2e(cut, ks, with_target=False)
    full = build_windows_e2e(rt, ks, with_target=False)
    summary.check("Без IMM: окна причинные", np.array_equal(part["x"], full["x"]) and len(ks) > 0,
                  f"обрезка на обзоре {k0}, окон {len(ks)}")

    # Причинность Mamba: изменение входа после t не меняет выход до t
    torch.manual_seed(config.RANDOM_SEED)
    m = build_corrector("Mamba", 8, config.HORIZON_SCANS).eval()
    x = torch.randn(4, config.WINDOW_SCANS, 8)
    x2 = x.clone()
    t_cut = config.WINDOW_SCANS // 2
    x2[:, t_cut:] += 5.0
    with torch.no_grad():
        a, b = m.sequence(x), m.sequence(x2)
    before = float((a[:, :t_cut] - b[:, :t_cut]).abs().max())
    after = float((a[:, t_cut:] - b[:, t_cut:]).abs().max())
    summary.check("Mamba: выход в момент t не зависит от входов после t", before == 0.0 and after > 0.0,
                  f"отличие до t: {before}, после t: {after:.3f}")

    # Смоук всех моделей в обоих режимах
    lines = []
    ok = True
    for name in CORRECTORS:
        for n_feat in (w["x"].shape[-1], N_FEATURES_E2E):
            torch.manual_seed(config.RANDOM_SEED)
            model = build_corrector(name, n_feat, config.HORIZON_SCANS)
            xb = torch.randn(8, config.WINDOW_SCANS, n_feat)
            out = model(xb)
            ((out - 1.0) ** 2).mean().backward()
            grads = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
            ok &= tuple(out.shape) == (8, config.HORIZON_SCANS, 2) and float(out.abs().max()) == 0.0 and grads > 0
        lines.append(f"{name}: {sum(p.numel() for p in model.parameters())} параметров (вход без IMM)")
    summary.check("Модели: формы, нулевой выход до обучения, градиенты", ok, "; ".join(lines))


def run_experiment(label, corrector, mode, ws_tr, ws_val, params, device, summary, log, smoke) -> dict:
    tr, va = ws_tr.view(mode), ws_val.view(mode)
    norm = dataset.Normalizer.fit(tr)
    x_tr, y_tr, x_val, y_val = norm.x(tr.x), norm.y(tr.y), norm.x(va.x), norm.y(va.y)

    rng = make_rng("training")
    batch_idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector(corrector, x_tr.shape[-1], config.HORIZON_SCANS)
    first, last = overfit_one_batch(model, x_tr[batch_idx], y_tr[batch_idx], config.OVERFIT_STEPS, 3e-3, device)
    summary.check(f"{label}: overfit one batch", last < config.OVERFIT_TARGET * first,
                  f"loss {first:.4f} -> {last:.4f}")

    log.info("Обучение %s", label)
    rng = make_rng("training")
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector(corrector, x_tr.shape[-1], config.HORIZON_SCANS)
    t0 = time.time()
    history, _ = train(model, x_tr, y_tr, x_val, y_val, params, device, rng)
    train_s = time.time() - t0
    ep = history["epochs"]
    summary.check(f"{label}: модель обучилась (val ниже начального)",
                  history["best_val_loss"] < history["val_loss_initial"],
                  f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
    summary.check(f"{label}: train loss падает", len(ep) >= 2 and ep[-1]["train_loss"] < ep[0]["train_loss"],
                  f"{ep[0]['train_loss']:.4f} -> {ep[-1]['train_loss']:.4f} за {len(ep)} эпох")

    t0 = time.time()
    out = norm.y_inv(predict(model, x_val, device))
    infer_us = (time.time() - t0) / max(len(x_val), 1) * 1e6
    slug = label.replace("+", "_").replace(" ", "_").replace("(", "").replace(")", "")
    ART.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "corrector": corrector,
                "mode": mode, "smoke": smoke}, ART / f"{slug}{'_smoke' if smoke else ''}.pt")
    history.update({"train_seconds": train_s, "infer_us_per_sample": infer_us, "device": str(device),
                    "params": params, "n_params": sum(p.numel() for p in model.parameters())})
    train_plots.plot_history(history, OUT_DIR / f"loss_{slug}.png", f"Обучение {label}")
    return {"pred": out, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 3: сравнение архитектур")
    parser.add_argument("--smoke", action="store_true", help="подвыборка и 2 эпохи (проверка конвейера)")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать окна")
    parser.add_argument("--only", default="", help="через запятую: корректоры, которые обучать")
    args = parser.parse_args()

    log = setup_logging(STAGE + ("_smoke" if args.smoke else ""))
    summary = StageSummary("Этап 3: сравнение архитектур" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Полное обучение - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    params = SMOKE if args.smoke else FULL
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    experiments = [e for e in EXPERIMENTS if not only or e[1] in only]

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
    ws_tr = dataset.load_or_build("train_smoke" if args.smoke else "train", real_train + synth, CACHE, args.rebuild)
    ws_val, rts, runs = dataset.build(val_tracks)
    summary.section("Данные", [
        f"train: {len(real_train)} реальных + {len(synth)} синтетических трасс, {len(ws_tr)} окон",
        f"val: {len(val_tracks)} трасс, {len(ws_val)} окон; оценка - только по обзорам с окном",
        f"окно {config.WINDOW_SCANS} обзоров; признаков: гибрид {ws_tr.x.shape[-1]}, без IMM {ws_tr.x_e2e.shape[-1]}",
        f"подготовка окон: {time.time() - t0:.1f} с; параметры обучения (для всех моделей) {params}",
    ])

    log.info("3. Модели")
    results = {}
    for label, corrector, mode in experiments:
        results[label] = run_experiment(label, corrector, mode, ws_tr, ws_val, params, device, summary, log,
                                        args.smoke)

    log.info("4. Сравнение на val")
    forecasts = {"IMM": []}
    for label in results:
        forecasts[label] = []
    for i, (rt, run) in enumerate(zip(rts, runs)):
        sel = ws_val.track == i
        ks = ws_val.k[sel]
        forecasts["IMM"].append(mask_forecast(run.forecast, ks))
        for label, corrector, mode in experiments:
            pred = results[label]["pred"][sel]
            if mode == "hybrid":
                fc = apply_correction(run.forecast, ks, ws_val.psi[sel], pred)
            else:
                fc = apply_e2e(rt, ks, ws_val.psi_e2e[sel], pred)
            forecasts[label].append(mask_forecast(fc, ks))
    report = compare(rts, forecasts, summary, "val", OUT_DIR, "val")

    base = report["IMM"]
    lines = [f"{'метод':16s} {'параметров':>10s} {'эпох':>5s} {'обуч., с':>9s} {'прогноз, мкс':>13s} "
             f"{'ADE':>6s} {'к IMM':>7s} {'вход':>6s} {'маневр':>7s}"]
    for label in results:
        h = results[label]["history"]
        r = report[label]
        lines.append(f"{label:16s} {h['n_params']:10d} {len(h['epochs']):5d} {h['train_seconds']:9.0f} "
                     f"{h['infer_us_per_sample']:13.1f} {r['overall']['ade']:6.1f} "
                     f"{(r['overall']['ade'] / base['overall']['ade'] - 1):+7.1%} {r['by_phase']['entry']:6.1f} "
                     f"{r['by_phase']['turn']:7.1f}")
    lines.append(f"{'IMM':16s} {'-':>10s} {'-':>5s} {'-':>9s} {'-':>13s} {base['overall']['ade']:6.1f} "
                 f"{'':7s} {base['by_phase']['entry']:6.1f} {base['by_phase']['turn']:7.1f}")
    summary.section("Итог сравнения (научный результат, не блокирует)" + (" - СМОУК, не для диплома" if args.smoke else ""),
                    lines)
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "histories.json").write_text(json.dumps({k: v["history"] for k, v in results.items()},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    summary.write(OUT_DIR / ("summary_smoke.txt" if args.smoke else "summary.txt"))
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
