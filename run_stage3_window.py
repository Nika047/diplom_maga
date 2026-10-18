"""
Этап 3, дополнение: влияние длины окна наблюдения на корректоры.

Вопрос: проявится ли преимущество Mamba (линейная сложность по длине
последовательности) на длинном окне. Те же 6 моделей, что на этапе 3, обучаются на
окнах из WINDOWS обзоров с одинаковыми параметрами.

Честное сравнение окон: длинное окно появляется в трассе позже, поэтому все
варианты оцениваются только на обзорах, где есть окно самой большой длины (общий
набор пар (k, h)). Обучающие выборки у окон разные: чем длиннее окно, тем меньше
примеров из начала каждой трассы.

Запуск:
  домашний ПК (GPU):  .venv\\Scripts\\python.exe run_stage3_window.py
  рабочий ПК (смоук): .venv\\Scripts\\python.exe run_stage3_window.py --smoke --windows 20,30 --only GRU,Transformer
--windows 20,60,100 - длины окон; --only - корректоры (по имени); --rebuild - пересобрать окна.
"""

import argparse
import json
import pickle
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
from data import splits as splits_mod  # noqa: E402
from data.windows import apply_correction, apply_e2e  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from viz.filter_plots import METHOD_COLORS  # noqa: E402
from viz.plots import _save  # noqa: E402

FULL = {"batch": 512, "epochs": 80, "lr": 1e-3, "patience": 10, "weight_decay": 1e-4}
SMOKE = {"batch": 256, "epochs": 2, "lr": 1e-3, "patience": 2, "weight_decay": 1e-4}
SMOKE_REAL_STEP = 8
SMOKE_SYNTH_STEP = 25
WINDOWS = (20, 60, 100)
EXPERIMENTS = [("IMM+LSTM", "LSTM", "hybrid"), ("IMM+GRU", "GRU", "hybrid"),
               ("IMM+Transformer", "Transformer", "hybrid"), ("IMM+Mamba", "Mamba", "hybrid"),
               ("GRU (без IMM)", "GRU", "e2e"), ("Mamba (без IMM)", "Mamba", "e2e")]

STAGE = "stage3_window"
OUT_DIR = config.RESULTS_DIR / STAGE
CACHE = config.CACHE_DIR / "stage2"
ART = config.PROJECT_DIR / "artifacts" / STAGE
CACHE0 = config.CACHE_DIR / "stage0"


def plot_ade_vs_window(table: dict, windows: list, path: Path) -> None:
    """table: {метод: {окно: ADE}}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    secs = [w * config.SCAN_PERIOD for w in windows]
    for method, vals in table.items():
        ax.plot(secs, [vals.get(w, np.nan) for w in windows], "-o", color=METHOD_COLORS.get(method), label=method)
    ax.set_xlabel("длина окна наблюдения, с")
    ax.set_ylabel("ADE по горизонтали на val, м")
    ax.set_xticks(secs)
    ax.legend(fontsize=8)
    ax.set_title("Ошибка корректоров в зависимости от длины окна (общие обзоры)", fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 3: длина окна")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    log = setup_logging(STAGE + ("_smoke" if args.smoke else ""))
    summary = StageSummary("Этап 3, дополнение: длина окна" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Полное обучение - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    params = SMOKE if args.smoke else FULL
    windows = sorted(int(w) for w in args.windows.split(","))
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

    # Окна по всем длинам; общий набор обзоров val - обзоры самого длинного окна
    data = {}
    lines = []
    for w in windows:
        t0 = time.time()
        name = f"{'train_smoke' if args.smoke else 'train'}" + ("" if w == config.WINDOW_SCANS else f"_w{w}")
        ws_tr = dataset.load_or_build(name, real_train + synth, CACHE, args.rebuild, window=w)
        ws_val, rts, runs = dataset.build(val_tracks, window=w)
        data[w] = (ws_tr, ws_val, rts, runs)
        lines.append(f"окно {w} обзоров ({w * config.SCAN_PERIOD:.0f} с): train {len(ws_tr)} окон, val {len(ws_val)}, "
                     f"{time.time() - t0:.0f} с")
    w_max = windows[-1]
    ws_val_max = data[w_max][1]
    common = {i: set(ws_val_max.k[ws_val_max.track == i].tolist()) for i in range(len(val_tracks))}
    n_common = sum(len(v) for v in common.values())
    nested = all(common[i] <= set(data[w][1].k[data[w][1].track == i].tolist())
                 for w in windows for i in common)
    summary.check("Общие обзоры оценки есть у всех длин окна", nested and n_common > 0,
                  f"общих обзоров val {n_common} (по окну {w_max})")
    summary.section("Данные", lines + [f"параметры обучения (для всех) {params}"])

    log.info("Обучение")
    rts, runs = data[w_max][2], data[w_max][3]
    forecasts = {"IMM": [mask_forecast(run.forecast, np.array(sorted(common[i]), dtype=int))
                         for i, run in enumerate(runs)]}
    histories = {}
    for w in windows:
        ws_tr, ws_val, _, _ = data[w]
        for label, corrector, mode in experiments:
            key = f"{label}, окно {w}"
            res = run_experiment(key, corrector, mode, ws_tr, ws_val, params, device, summary, log, args.smoke,
                                 OUT_DIR, ART, window=w)
            histories[key] = res["history"]
            fcs = []
            for i, (rt, run) in enumerate(zip(rts, runs)):
                sel = ws_val.track == i
                ks = ws_val.k[sel]
                pred = res["pred"][sel]
                if mode == "hybrid":
                    fc = apply_correction(run.forecast, ks, ws_val.psi[sel], pred)
                else:
                    fc = apply_e2e(rt, ks, ws_val.psi_e2e[sel], pred)
                fcs.append(mask_forecast(fc, np.array(sorted(common[i]), dtype=int)))
            forecasts[key] = fcs

    report = compare(rts, forecasts, summary, "val, общие обзоры", None)
    table = {}
    lines = [f"{'метод':16s} " + " ".join(f"{'окно ' + str(w):>10s}" for w in windows)
             + "   (ADE, м; в скобках - время обучения, с)"]
    for label, _, _ in experiments:
        table[label] = {w: report[f"{label}, окно {w}"]["overall"]["ade"] for w in windows}
        lines.append(f"{label:16s} " + " ".join(
            f"{table[label][w]:5.1f} ({histories[f'{label}, окно {w}']['train_seconds']:3.0f})" for w in windows))
    lines.append(f"{'IMM':16s} {report['IMM']['overall']['ade']:5.1f}")
    summary.section("Итог: ADE в зависимости от длины окна (научный результат, не блокирует)"
                    + (" - СМОУК" if args.smoke else ""), lines)
    plot_ade_vs_window(table, windows, OUT_DIR / "ade_vs_window.png")
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "histories.json").write_text(json.dumps(histories, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.write(OUT_DIR / ("summary_smoke.txt" if args.smoke else "summary.txt"))
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
