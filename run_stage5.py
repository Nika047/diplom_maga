"""
Этап 5: прогноз начала маневра.

Новый компонент - корректор с двумя головами (models/onset.py): поправка к прогнозу
IMM и вероятность начала маневра. Метка (data/windows.onset_labels): маневр начнётся в
ближайшие ONSET_AHEAD обзоров (горизонт прогноза) или начался не больше ONSET_BEHIND
обзоров назад. Обучение совместное: MSE поправки + ONSET_LOSS_WEIGHT * BCE.

Два сравнения:
  1. Обнаружение начала маневра. Сеть против классических обнаружителей по прогону
     IMM: вероятность модели CT и сумма NIS модели CV за ONSET_NIS_SCANS обзоров.
     Сравнение при одинаковой вероятности ложной тревоги: порог каждого обнаружителя
     выбирается на val так, чтобы тревога горела в доле ONSET_P_FA обзоров прямого
     полёта вне интервалов метки. На val, test и holdout: средняя точность (без порога), точность
     и полнота по обзорам, полнота по событиям, опережение в обзорах, ложные тревоги.
  2. Прогноз. Тот же GRU с теми же гиперпараметрами (этап 4, IMM+GRU) и seed - без
     головы начала маневра и с ней: помогает ли вторая задача на входе в маневр.
IMM - подобранный на этапе 4 (cache/stage4/imm_best.json), окна train - из кэша этапа 4.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage5.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage5.py --smoke

Проверки: метки на подставной трассе, нулевая поправка до обучения, overfit one batch,
модель обучилась, однозадачный GRU повторяет этап 4, сеть обнаруживает лучше случайного
(средняя точность на val выше доли положительных обзоров).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data.windows import apply_correction, onset_labels  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from models.onset import OnsetGRU, multitask_loss  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from tuning import search  # noqa: E402
from viz import train_plots  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage5"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.PROJECT_DIR / "artifacts" / STAGE
EVAL_SETS = ("val", "test", "holdout")
NET = "сеть"
DETECTORS = ("IMM: вероятность CT", f"NIS модели CV за {config.ONSET_NIS_SCANS} обз.", NET)
SINGLE, MULTI = "IMM+GRU", "IMM+GRU+начало маневра"


def check_labels(summary: StageSummary) -> None:
    """Подставная трасса: начала маневров в обзорах 10 и 30, известный ответ."""
    ks = np.arange(40)
    got = onset_labels(SimpleNamespace(onsets=[10, 30]), ks, ahead=5, behind=2)
    want = (((ks >= 5) & (ks <= 12)) | ((ks >= 25) & (ks <= 32))).astype(np.float32)
    summary.check("Метки начала маневра на подставной трассе", np.array_equal(got, want),
                  f"положительных {int(got.sum())} из {len(ks)}, ожидалось {int(want.sum())}")


def labels_for(ws, rts: list) -> np.ndarray:
    out = np.zeros(len(ws), dtype=np.float32)
    for i, rt in enumerate(rts):
        sel = ws.track == i
        out[sel] = onset_labels(rt, ws.k[sel])
    return out


def classic_scores(ws, runs: list) -> dict:
    """Оценки классических обнаружителей в обзорах окон: {имя: (N,)}."""
    i_ct, i_cv = config.IMM_MODELS.index("CT"), config.IMM_MODELS.index("CV")
    mu, nis = np.zeros(len(ws)), np.zeros(len(ws))
    for i, run in enumerate(runs):
        sel = ws.track == i
        ks = ws.k[sel]
        mu[sel] = run.mu[ks, i_ct]
        csum = np.cumsum(np.r_[0.0, np.nan_to_num(run.nis[:, i_cv])])
        nis[sel] = csum[ks + 1] - csum[np.maximum(ks + 1 - config.ONSET_NIS_SCANS, 0)]  # причинная сумма
    return {DETECTORS[0]: mu, DETECTORS[1]: nis}


def per_track(ws, rts: list, score: np.ndarray) -> list:
    return [(rt, ws.k[ws.track == i], score[ws.track == i]) for i, rt in enumerate(rts)]


def detector_report(ws, rts: list, label: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """recall_scan - полнота по обзорам, recall - по событиям (начала маневров)."""
    b = metrics.binary_scores(score >= threshold, label)
    out = {"ap": metrics.average_precision(score, label), "precision": b["precision"], "recall_scan": b["recall"]}
    out.update(metrics.onset_events(per_track(ws, rts, score), threshold))
    return out


def fmt(vals: list, digits: int) -> str:
    """Среднее ± СКО по seed; одно значение - без СКО."""
    vals = np.array(vals, dtype=float)
    s = f"{vals.mean():.{digits}f}"
    return s if len(vals) == 1 else f"{s}±{vals.std():.{digits}f}"


def plot_response(curves: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, (ms, share) in curves.items():
        ax.plot(ms, share, "-o", ms=3, label=name)
    ax.axvspan(-config.ONSET_AHEAD, config.ONSET_BEHIND, color="0.9", zorder=0, label="интервал метки")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("обзор относительно начала маневра, m")
    ax.set_ylabel("доля тревог")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_example(rt, ks: np.ndarray, scores: dict, thresholds: dict, label: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(len(scores), 1, figsize=(9, 2.2 * len(scores)), sharex=True)
    for ax, (name, score) in zip(np.atleast_1d(axes), scores.items()):
        ax.fill_between(ks, 0, 1, where=label > 0.5, transform=ax.get_xaxis_transform(), color="0.88",
                        step="mid", label="метка")
        ax.plot(ks, score, "-", lw=1, label=name)
        ax.axhline(thresholds[name], color="r", lw=0.8, ls="--", label="порог")
        for o in rt.onsets:
            ax.axvline(o, color="k", lw=0.6)
        ax.legend(fontsize=7, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("обзор")
    fig.suptitle(f"Начало маневра: {rt.flight_id} (вертикальные линии - начала маневров по истине)", fontsize=9)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 5: прогноз начала маневра")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 5: прогноз начала маневра" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 5 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    t0 = time.time()

    check_labels(summary)
    imm_path, best_path = cache / "imm_best.json", cache / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or SINGLE not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {SINGLE} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window, hp, train_params = search.split_params(best_all[SINGLE]["params"])
    params = {**train_params, "epochs": epochs, "patience": patience}
    log.info("IMM этапа 4: %s; GRU этапа 4: окно %d, %s, %s", imm, window, hp, train_params)

    sets = load_data(args.smoke)
    tracks_tr = sets["train"] + sets["synth"]
    with search.config_override(imm):
        ws_tr = dataset.load_or_build(f"train{tag}_w{window}", tracks_tr, cache, window=window)
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
    lab_tr = labels_for(ws_tr, [observe(tr) for tr in tracks_tr])
    labels = {name: labels_for(b[0], b[1]) for name, b in built.items()}
    summary.section("Метки начала маневра", [
        f"интервал: от {config.ONSET_AHEAD} обзоров до начала маневра до {config.ONSET_BEHIND} после",
        f"train: {len(lab_tr)} окон, положительных {lab_tr.mean():.1%}"] + [
        f"{name}: {len(lab)} окон, положительных {lab.mean():.1%}, начал маневров "
        f"{sum(len(rt.onsets) for rt in built[name][1])}" for name, lab in labels.items()])

    norm = dataset.Normalizer.fit(ws_tr)
    x_tr = norm.x(ws_tr.x)
    n_out = config.HORIZON_SCANS * 2

    def mt_target(ws, lab):
        return np.concatenate([norm.y(ws.y).reshape(len(ws), n_out), lab[:, None]], axis=1).astype(np.float32)

    y_tr = mt_target(ws_tr, lab_tr)
    ws_val = built["val"][0]
    x_val, y_val = norm.x(ws_val.x), mt_target(ws_val, labels["val"])
    loss_fn = multitask_loss()

    ks_common = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
                 for name, b in built.items()}
    forecasts = {name: {"IMM": [mask_forecast(r.forecast, ks) for r, ks in zip(b[2], ks_common[name])]}
                 for name, b in built.items()}
    reports = {name: {} for name in EVAL_SETS}      # {набор: {метод: [отчёт по seed]}}
    det = {name: {} for name in EVAL_SETS}          # {набор: {обнаружитель: [отчёт по seed]}}
    net_scores, thresholds = {}, {}

    def to_forecasts(name, corr):
        ws, rts, runs = built[name]
        out = []
        for i, run in enumerate(runs):
            sel = ws.track == i
            fc = apply_correction(run.forecast, ws.k[sel], ws.psi[sel], corr[sel])
            out.append(mask_forecast(fc, ks_common[name][i]))
        return out

    with search.config_override(imm):
        for s_i, seed in enumerate(seeds):
            # Однозадачный GRU - тот же путь, что в итоге этапа 4
            res = run_experiment(f"{SINGLE}, seed {seed}", "GRU", "hybrid", ws_tr, ws_val, params, device, summary,
                                 log, args.smoke, OUT_DIR, ART, window=window, seed=seed, model_kwargs=hp,
                                 overfit=False, extra_sets={n: built[n][0] for n in ("test", "holdout")})
            preds = {"val": res["pred"], **res["extra"]}
            for name in EVAL_SETS:
                fcs = to_forecasts(name, preds[name])
                reports[name].setdefault(SINGLE, []).append(summarize_method(built[name][1], fcs))
                if s_i == 0:
                    forecasts[name][SINGLE] = fcs
            if s_i == 0:
                ga, got = best_all[SINGLE]["ade_val"], reports["val"][SINGLE][0]["ade"]
                summary.check(f"{SINGLE}: повторяет этап 4", abs(got / ga - 1) <= 0.05,
                              f"ADE val на этапе 4 {ga:.2f} м, здесь (seed {seed}) {got:.2f} м")

            # GRU с головой начала маневра
            key = f"{MULTI}, seed {seed}"
            if s_i == 0:
                rng = make_rng("training", seed)
                idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
                torch.manual_seed(seed)
                m = OnsetGRU(x_tr.shape[-1], config.HORIZON_SCANS, **hp)
                first, last = overfit_one_batch(m, x_tr[idx], y_tr[idx], config.OVERFIT_STEPS, 3e-3, device,
                                                loss_fn=loss_fn)
                summary.check(f"{MULTI}: overfit one batch", last < config.OVERFIT_TARGET * first,
                              f"loss {first:.4f} -> {last:.4f}")
            rng = make_rng("training", seed)
            torch.manual_seed(seed)
            model = OnsetGRU(x_tr.shape[-1], config.HORIZON_SCANS, **hp)
            if s_i == 0:
                zero = np.abs(predict(model, x_val[:256], device)[:, :n_out]).max()
                summary.check(f"{MULTI}: до обучения поправка нулевая (= IMM)", zero == 0.0,
                              f"max |поправка| {zero:.2e}")
            log.info("Обучение %s", key)
            history, _ = train(model, x_tr, y_tr, x_val, y_val, params, device, rng, loss_fn=loss_fn)
            summary.check(f"{key}: модель обучилась", history["best_val_loss"] < history["val_loss_initial"],
                          f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
            train_plots.plot_history(history, OUT_DIR / f"loss_onset_{seed}{tag}.png", f"Обучение: {key}")
            ART.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "model_kwargs": hp,
                        "window": window, "seed": seed, "imm": imm}, ART / f"onset_gru_{seed}{tag}.pt")

            for name in EVAL_SETS:
                raw = predict(model, norm.x(built[name][0].x), device)
                corr = norm.y_inv(raw[:, :n_out].reshape(-1, config.HORIZON_SCANS, 2))
                fcs = to_forecasts(name, corr)
                reports[name].setdefault(MULTI, []).append(summarize_method(built[name][1], fcs))
                net_scores[(name, seed)] = 1.0 / (1.0 + np.exp(-raw[:, -1].astype(float)))
                if s_i == 0:
                    forecasts[name][MULTI] = fcs
            thresholds[(NET, seed)] = metrics.threshold_at_false_alarm(
                per_track(ws_val, built["val"][1], net_scores[("val", seed)]))
            for name in EVAL_SETS:
                ws, rts, _ = built[name]
                det[name].setdefault(NET, []).append(
                    detector_report(ws, rts, labels[name], net_scores[(name, seed)], thresholds[(NET, seed)]))

    # Классические обнаружители: порог по val, оценка на всех наборах
    classic = {name: classic_scores(built[name][0], built[name][2]) for name in EVAL_SETS}
    for d in DETECTORS[:2]:
        thresholds[d] = metrics.threshold_at_false_alarm(per_track(ws_val, built["val"][1], classic["val"][d]))
        for name in EVAL_SETS:
            ws, rts, _ = built[name]
            det[name][d] = [detector_report(ws, rts, labels[name], classic[name][d], thresholds[d])]

    ap_val = det["val"][NET][0]["ap"]
    summary.check("Сеть обнаруживает начало маневра лучше случайного", ap_val > labels["val"].mean(),
                  f"средняя точность на val {ap_val:.3f}, доля положительных {labels['val'].mean():.3f}")

    for name in EVAL_SETS:
        lines = [f"{'обнаружитель':26s} {'AP':>11s} {'точн.':>11s} {'полнота':>11s} {'по событ.':>11s} "
                 f"{'опереж.':>11s} {'раньше':>11s} {'P лт':>11s} {'ложн./ч':>11s}"]
        for d in DETECTORS:
            reps = det[name][d]
            row = f"{d:26s}"
            for key, digits in (("ap", 3), ("precision", 3), ("recall_scan", 3), ("recall", 3), ("lead_mean", 2),
                                ("early_share", 2), ("p_fa", 3), ("false_per_hour", 1)):
                row += f" {fmt([r[key] for r in reps], digits):>11s}"
            lines.append(row)
        n_on = det[name][NET][0]["n_onsets"]
        summary.section(f"{name}: обнаружение начала маневра (сеть - по {len(seeds)} seed)", lines + [
            f"начал маневров с полным интервалом: {n_on}; доля положительных обзоров {labels[name].mean():.3f}",
            "AP - средняя точность по обзорам без порога; точность и полнота - по обзорам при пороге с val;",
            f"по событ. - доля начал маневров o, где эпизод тревоги начался в o - {config.ONSET_AHEAD} .. "
            f"o + {config.ONSET_DETECT_LATE};",
            "опереж. - среднее опережение начала эпизода, обзоров (> 0 - до начала маневра, < 0 - задержка);",
            "раньше - доля обнаруженных до начала маневра;",
            "P лт - доля обзоров прямой вне интервалов метки с тревогой (вероятность ложной тревоги);",
            "ложн./ч - эпизоды тревоги, начавшиеся в таких обзорах, на час такого полёта"])

        lines = [f"{'метод':24s} {'ADE':>11s} {'FDE':>11s} {'прямая':>11s} {'вход':>11s} {'маневр':>11s} {'выход':>11s}"]
        base = summarize_method(built[name][1], forecasts[name]["IMM"])
        lines.append(f"{'IMM':24s}" + "".join(f" {base[k]:11.1f}" for k in ("ade", "fde", "straight", "entry",
                                                                              "turn", "exit")))
        for method in (SINGLE, MULTI):
            reps = reports[name][method]
            lines.append(f"{method:24s}" + "".join(f" {fmt([r[k] for r in reps], 1):>11s}"
                                                   for k in ("ade", "fde", "straight", "entry", "turn", "exit")))
        summary.section(f"{name}: прогноз, ADE (м), среднее ± СКО по {len(seeds)} seed", lines)
        compare(built[name][1], forecasts[name], summary, f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")

        curves = {}
        for d in DETECTORS:
            score = net_scores[(name, seeds[0])] if d == NET else classic[name][d]
            thr = thresholds[(NET, seeds[0])] if d == NET else thresholds[d]
            ms, share, _ = metrics.alarm_response(per_track(built[name][0], built[name][1], score), thr)
            curves[d] = (ms, share)
        plot_response(curves, OUT_DIR / f"onset_response_{name}{tag}.png",
                      f"Тревоги относительно начала маневра, {name}")

    # Пример: трасса test с наибольшим числом начал маневров
    ws, rts, _ = built["test"]
    i = int(np.argmax([len(rt.onsets) for rt in rts]))
    sel = ws.track == i
    scores = {d: (net_scores[("test", seeds[0])] if d == NET else classic["test"][d])[sel] for d in DETECTORS}
    thr = {d: thresholds[(NET, seeds[0])] if d == NET else thresholds[d] for d in DETECTORS}
    plot_example(rts[i], ws.k[sel], scores, thr, labels["test"][sel], OUT_DIR / f"onset_example{tag}.png")

    summary.section(f"Пороги (на val вероятность ложной тревоги {config.ONSET_P_FA:g})", [f"{d}: {thresholds[d]:.4g}" for d in DETECTORS[:2]] + [
        f"{NET}, seed {s}: {thresholds[(NET, s)]:.4g}" for s in seeds])
    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps({"forecast": reports, "detectors": det}, ensure_ascii=False,
                                                           indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
