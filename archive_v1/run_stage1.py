"""
Этап 1: модель РЛС и классические методы.

1. Проверки на подставных данных с известным ответом (метрики, модели движения,
   линейная экстраполяция, пересчёт отметок).
2. Отметки РЛС для реальных полётов val и синтетики анализа (кэш этапа 0).
3. Линейная экстраполяция (CV), фильтр Калмана (одна CV), IMM (CV, CA, CT).
4. Метрики по горизонтам и участкам, кривые динамической ошибки.
5. Проверки PASS / FAIL, рисунки и сводка в results/stage1/, метрики в metrics.json.

Тест (выборки test и holdout) здесь не используется - он для итогового сравнения.

Запуск: .venv\\Scripts\\python.exe run_stage1.py (после run_stage0.py)
"""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from data import splits as splits_mod  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import ade_by_source, source_of, source_section  # noqa: E402
from filters.extrapolation import cv_extrapolate  # noqa: E402
from filters.imm import IMM  # noqa: E402
from filters.models import CA, CT, CV, VX, VY, W, X, Y  # noqa: E402
from radar.model import RadarTrack, converted_cov, observe, polar_to_enu, to_polar  # noqa: E402
from viz import filter_plots  # noqa: E402

STAGE = "stage1"
OUT_DIR = config.RESULTS_DIR / STAGE
CACHE0 = config.CACHE_DIR / "stage0"
METHODS = ("CV", "KF", "IMM")
VAL = "val (реальные)" if config.USE_HOLDOUT else "val"     # в merged val смешанный
TURN_CODES = (PHASE_CODE["entry"], PHASE_CODE["turn"])


# ============================================================================
# 1. Подставные данные с известным ответом
# ============================================================================

def _fake_radar_track(truth: np.ndarray) -> RadarTrack:
    k = len(truth)
    return RadarTrack("test", np.arange(k) * config.SCAN_PERIOD, truth, np.zeros(3), np.zeros((k, 3)),
                      truth.copy(), np.tile(np.eye(3), (k, 1, 1)), np.ones(k, bool),
                      np.zeros(k, np.int8), [])


def self_tests(summary: StageSummary) -> None:
    hz = config.HORIZON_SCANS
    k = 40
    truth = np.column_stack([np.arange(k) * 20.0, np.zeros(k), np.zeros(k)])
    rt = _fake_radar_track(truth)
    fc = np.zeros((k, hz, 3))
    for h in range(1, hz + 1):
        fc[: k - h, h - 1] = truth[h:] + np.array([3.0, 4.0, 0.0])
    s = metrics.summarize(metrics.forecast_pairs(rt, fc))
    summary.check("Метрики: известный ответ (ошибка 3, 4 м)",
                  abs(s["ade"] - 5) < 1e-9 and abs(s["rmse"] - 5) < 1e-9 and abs(s["mae"] - 3.5) < 1e-9
                  and abs(s["fde"] - 5) < 1e-9,
                  f"ADE {s['ade']:.3f}, RMSE {s['rmse']:.3f}, MAE {s['mae']:.3f}, FDE {s['fde']:.3f} "
                  f"(ожидалось 5, 5, 3.5, 5)")
    for h in range(1, hz + 1):
        fc[: k - h, h - 1] = truth[h:]
    s = metrics.summarize(metrics.forecast_pairs(rt, fc))
    summary.check("Метрики: точный прогноз", s["ade"] == 0 and abs(s["r2"] - 1) < 1e-12,
                  f"ADE {s['ade']}, R2 {s['r2']}")

    # Модели движения: CT на окружности, CV и CA на прямой - точный прогноз
    v, w, t = 20.0, 0.1, config.SCAN_PERIOD
    x = np.zeros(9)
    x[VX], x[W] = v, w
    model = CT()
    for _ in range(10):
        x = model.f(x, t)
    th = w * 10 * t
    exact = np.array([v / w * np.sin(th), v / w * (1 - np.cos(th))])
    err_ct = float(np.linalg.norm(x[[X, Y]] - exact))
    x = np.zeros(9)
    x[VX], x[VY] = 10.0, -5.0
    err_cv = float(np.linalg.norm(CV().f(x, 4.0)[[X, Y]] - [40.0, -20.0]))
    x = np.zeros(9)
    x[VX], x[6] = 10.0, 1.0
    err_ca = float(abs(CA().f(x, 4.0)[X] - (40.0 + 0.5 * 16.0)))
    summary.check("Модели движения: известный ответ", max(err_ct, err_cv, err_ca) < 1e-6,
                  f"ошибка CT на окружности {err_ct:.2e} м, CV {err_cv:.2e} м, CA {err_ca:.2e} м")

    # Линейная экстраполяция на прямой без шума
    rt = _fake_radar_track(truth)
    fc = cv_extrapolate(rt)
    s = metrics.summarize(metrics.forecast_pairs(rt, fc))
    summary.check("Линейная экстраполяция: прямая без шума", s["ade"] < 1e-6, f"ADE {s['ade']:.2e} м")

    # Пересчёт отметок: туда-обратно точно; ковариация согласована с шумом (NEES около 3)
    rng = np.random.default_rng(0)
    rel = np.array([[2000.0, 1500.0, 300.0]])
    polar = to_polar(rel)
    back = float(np.abs(polar_to_enu(polar) - rel).max())
    sig = np.array([config.RADAR_SIGMA_R, np.radians(config.RADAR_SIGMA_AZ_DEG),
                    np.radians(config.RADAR_SIGMA_EL_DEG)])
    noisy = polar + rng.standard_normal((20000, 3)) * sig
    d = polar_to_enu(noisy) - rel
    cov = converted_cov(noisy)
    nees = np.einsum("ni,nij,nj->n", d, np.linalg.inv(cov), d)
    summary.check("Пересчёт отметок: точность и NEES", back < 1e-6 and 2.85 <= nees.mean() <= 3.15,
                  f"ошибка туда-обратно {back:.1e} м, средний NEES {nees.mean():.3f} (ожидалось около 3)")


# ============================================================================
# 2-4. Данные, методы, метрики
# ============================================================================

def load_stage0(log):
    paths = [CACHE0 / name for name in ("tracks_real.pkl", "tracks_synth_analysis.pkl", "splits.json")]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        log.error("Нет кэша этапа 0: %s. Сначала запустить run_stage0.py", missing)
        return None
    with open(paths[0], "rb") as fh:
        real = pickle.load(fh)
    with open(paths[1], "rb") as fh:
        synth = pickle.load(fh)
    splits = splits_mod.load_splits(paths[2])
    return real, synth, splits


def check_radar(datasets: dict, summary: StageSummary) -> None:
    res, nees = [], []
    for rts in datasets.values():
        for rt in rts:
            true_polar = to_polar(rt.truth - rt.radar_pos)
            r = rt.polar - true_polar
            r[:, 1] = (r[:, 1] + np.pi) % (2 * np.pi) - np.pi
            res.append(r)
            d = rt.meas - rt.truth
            nees.append(np.einsum("ni,nij,nj->n", d, np.linalg.inv(rt.meas_cov), d))
    res = np.concatenate(res)
    nees = np.concatenate(nees)
    std = res.std(axis=0)
    want = np.array([config.RADAR_SIGMA_R, np.radians(config.RADAR_SIGMA_AZ_DEG),
                     np.radians(config.RADAR_SIGMA_EL_DEG)])
    rel = np.abs(std / want - 1)
    summary.check("РЛС: СКО шума равны заданным", (rel <= 0.05).all(),
                  f"дальность {std[0]:.2f} м (задано {want[0]}), азимут {np.degrees(std[1]):.3f} град "
                  f"({config.RADAR_SIGMA_AZ_DEG}), угол места {np.degrees(std[2]):.3f} град "
                  f"({config.RADAR_SIGMA_EL_DEG}); отметок {len(res)}")
    summary.check("РЛС: ковариация пересчитанных отметок согласована", 2.85 <= nees.mean() <= 3.15,
                  f"средний NEES {nees.mean():.3f} (ожидалось около 3)")


def run_methods(rts: list, log, label: str) -> dict:
    out = {}
    kf, imm = IMM(("CV",), mu0=(1.0,)), IMM()
    t0 = time.time()
    out["CV"] = {"forecast": [cv_extrapolate(rt) for rt in rts], "runs": None}
    t1 = time.time()
    kf_runs = [kf.run(rt) for rt in rts]
    t2 = time.time()
    imm_runs = [imm.run(rt) for rt in rts]
    t3 = time.time()
    out["KF"] = {"forecast": [r.forecast for r in kf_runs], "runs": kf_runs}
    out["IMM"] = {"forecast": [r.forecast for r in imm_runs], "runs": imm_runs}
    n_scans = sum(len(rt.t) for rt in rts)
    log.info("%s: %d трасс, %d обзоров; время CV %.1f с, KF %.1f с, IMM %.1f с", label, len(rts),
             n_scans, t1 - t0, t2 - t1, t3 - t2)
    out["_timing"] = {"scans": n_scans, "CV": t1 - t0, "KF": t2 - t1, "IMM": t3 - t2}
    return out


def evaluate(rts: list, res: dict, label: str, summary: StageSummary) -> dict:
    report = {}
    lines = [f"{'метод':5s} {'MAE':>7s} {'RMSE':>7s} {'ADE':>7s} {'FDE':>7s} {'R2':>6s} {'ADE 3D':>7s} "
             f"{'пар':>7s}  ADE по горизонтам 1..{config.HORIZON_SCANS}, м"]
    phase_lines = [f"{'метод':5s} " + " ".join(f"{p:>9s}" for p in config.PHASES if p != "slow")]
    for method in METHODS:
        pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, fc)
                                      for rt, fc in zip(rts, res[method]["forecast"])])
        s = metrics.summarize(pairs)
        by_h = metrics.ade_by_horizon(pairs)
        by_phase = {p: metrics.summarize(metrics.select(pairs, pairs["phase"] == code))["ade"]
                    for p, code in PHASE_CODE.items()}
        maneuver = metrics.summarize(metrics.select(pairs, np.isin(pairs["phase"], TURN_CODES)))
        report[method] = {"overall": s, "by_horizon": by_h.tolist(), "by_phase": by_phase,
                          "maneuver": maneuver, "pairs": pairs}
        lines.append(f"{method:5s} {s['mae']:7.1f} {s['rmse']:7.1f} {s['ade']:7.1f} {s['fde']:7.1f} "
                     f"{s['r2']:6.3f} {s['ade_3d']:7.1f} {s['n']:7d}  " + " ".join(f"{v:6.1f}" for v in by_h))
        phase_lines.append(f"{method:5s} " + " ".join(f"{by_phase[p]:9.1f}" for p in config.PHASES if p != "slow"))
    summary.section(f"{label}: метрики по горизонтали (м), все горизонты", lines)
    summary.section(f"{label}: ADE по участкам в момент прогноза (м)", phase_lines)

    dyn_lines = [f"{'метод':5s} {'h':>2s} {'прямая':>7s} {'пик':>7s} {'при m':>6s} {'площадь':>8s} "
                 f"{'восст., обз.':>12s} {'маневров':>9s}"]
    curves, levels = {}, {}
    for h in (1, config.HORIZON_SCANS):
        for method in METHODS:
            ms, curve, counts = metrics.dynamic_curve(rts, res[method]["forecast"], h)
            level = metrics.straight_level(report[method]["pairs"], h)
            st = metrics.dynamic_stats(ms, curve, level)
            curves[(method, h)] = (ms, curve, counts)
            levels[(method, h)] = level
            report[method][f"dynamic_h{h}"] = {"level": level, **st, "curve": np.nan_to_num(curve).tolist()}
            dyn_lines.append(f"{method:5s} {h:2d} {level:7.1f} {st['peak']:7.1f} {st['peak_m']:6.0f} "
                             f"{st['area']:8.1f} {st['recovery']:12.0f} {int(counts[ms == 0][0]):9d}")
    summary.section(f"{label}: динамическая ошибка (м; m - обзоров от начала маневра)", dyn_lines)
    source_section(summary, label, {m: [{"ade": report[m]["overall"]["ade"], **{
        f"ade_{s}": v for s, v in ade_by_source(rts, res[m]["forecast"]).items()}}] for m in METHODS})
    report["_curves"], report["_levels"] = curves, levels
    return report


def check_methods(datasets: dict, results: dict, reports: dict, summary: StageSummary) -> None:
    # Фильтр Калмана на прямых сглаживает шум отметок
    rts = [rt for rt in datasets["синтетика"] if rt.flight_id.startswith("synthA_straight")]
    runs = [r for rt, r in zip(datasets["синтетика"], results["синтетика"]["KF"]["runs"])
            if rt.flight_id.startswith("synthA_straight")]
    est_err, meas_err = [], []
    for rt, run in zip(rts, runs):
        k = np.arange(config.WARMUP_SCANS, len(rt.t))
        est_err.append(np.linalg.norm(run.est[k][:, [X, Y]] - rt.truth[k, :2], axis=1))
        meas_err.append(np.linalg.norm(rt.meas[k, :2] - rt.truth[k, :2], axis=1))
    e, m = np.concatenate(est_err), np.concatenate(meas_err)
    summary.check("Фильтр Калмана на прямой точнее отметок", e.mean() < m.mean(),
                  f"средняя горизонтальная ошибка оценки {e.mean():.1f} м против отметок {m.mean():.1f} м")

    # IMM на маневре переключается на CT
    i_ct = config.IMM_MODELS.index("CT")
    mu_turn, mu_straight = [], []
    for rt, run in zip(datasets["синтетика"], results["синтетика"]["IMM"]["runs"]):
        ok = ~np.isnan(run.mu[:, i_ct])
        mu_turn.append(run.mu[ok & (rt.phase == PHASE_CODE["turn"]), i_ct])
        mu_straight.append(run.mu[ok & (rt.phase == PHASE_CODE["straight"]), i_ct])
    a, b = np.concatenate(mu_turn).mean(), np.concatenate(mu_straight).mean()
    summary.check("IMM: на маневре растёт вероятность CT", a > b + 0.2,
                  f"средняя вероятность CT на маневре {a:.2f}, на прямой {b:.2f}")

    for label, report in reports.items():
        imm, kf = report["IMM"]["maneuver"]["ade"], report["KF"]["maneuver"]["ade"]
        summary.check(f"{label}: IMM на маневрах точнее фильтра Калмана", imm < kf,
                      f"ADE на входе и маневре: IMM {imm:.1f} м, KF {kf:.1f} м")

    # Кривая динамической ошибки ловит маневр: пик после начала заметно выше уровня прямой
    dyn = reports["синтетика"]["KF"][f"dynamic_h{config.HORIZON_SCANS}"]
    summary.check("Кривая динамической ошибки видит маневр",
                  dyn["peak"] > 1.5 * dyn["level"] and 0 <= dyn["peak_m"] <= 10,
                  f"KF, h={config.HORIZON_SCANS}: пик {dyn['peak']:.1f} м на m={dyn['peak_m']:.0f} "
                  f"при уровне прямой {dyn['level']:.1f} м")


def main() -> int:
    log = setup_logging(STAGE)
    summary = StageSummary("Этап 1: модель РЛС и классические методы")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("1. Проверки на подставных данных")
    self_tests(summary)

    log.info("2. Данные и отметки РЛС")
    loaded = load_stage0(log)
    if loaded is None:
        summary.check("Кэш этапа 0 найден", False, "запустить run_stage0.py")
        summary.write(OUT_DIR / "summary.txt")
        return 1
    real, synth, splits = loaded
    val = splits_mod.load_sets(CACHE0)["val"]     # в варианте merged - вместе со сгенерированными
    datasets = {VAL: [observe(tr) for tr in val],
                "синтетика": [observe(tr) for tr in synth]}
    summary.section("Данные", [f"{name}: {len(rts)} трасс, {sum(len(rt.t) for rt in rts)} обзоров, "
                               f"{sum(len(rt.onsets) for rt in rts)} маневров" for name, rts in datasets.items()]
                    + [f"РЛС: СКО дальности {config.RADAR_SIGMA_R} м, азимута {config.RADAR_SIGMA_AZ_DEG} град, "
                       f"угла места {config.RADAR_SIGMA_EL_DEG} град, период {config.SCAN_PERIOD} с, "
                       f"вероятность обнаружения {config.RADAR_PD}, расстояние {config.RADAR_DIST_RANGE} м"])
    check_radar(datasets, summary)

    log.info("3. Методы")
    results = {name: run_methods(rts, log, name) for name, rts in datasets.items()}
    summary.section("Время работы (рабочий ПК)", [
        f"{name}: {r['_timing']['scans']} обзоров; CV {r['_timing']['CV']:.1f} с, KF {r['_timing']['KF']:.1f} с, "
        f"IMM {r['_timing']['IMM']:.1f} с" for name, r in results.items()])

    log.info("4. Метрики")
    reports = {name: evaluate(datasets[name], results[name], name, summary) for name in datasets}
    check_methods(datasets, results, reports, summary)

    log.info("5. Рисунки")
    for name, tag in (("синтетика", "synth"), (VAL, "val")):
        rep = reports[name]
        filter_plots.plot_dynamic_curves(rep["_curves"], rep["_levels"], OUT_DIR / f"dynamic_error_{tag}.png",
                                         f"Динамическая ошибка: {name}")
        filter_plots.plot_by_horizon({m: rep[m]["by_horizon"] for m in METHODS},
                                     OUT_DIR / f"ade_by_horizon_{tag}.png", f"Ошибка по горизонтам: {name}")
        filter_plots.plot_by_phase({m: rep[m]["by_phase"] for m in METHODS},
                                   OUT_DIR / f"ade_by_phase_{tag}.png", f"Ошибка по участкам: {name}")
    rts_s = datasets["синтетика"]
    for scenario in ("zigzag", "loiter"):
        i = next(i for i, rt in enumerate(rts_s) if rt.flight_id.startswith(f"synthA_{scenario}"))
        filter_plots.plot_track_filters(rts_s[i], results["синтетика"]["IMM"]["runs"][i],
                                        OUT_DIR / f"imm_example_{scenario}.png",
                                        f"IMM на синтетике ({rts_s[i].flight_id})")
    rts_v = datasets[VAL]
    i = int(np.argmax([len(rt.onsets) if source_of(rt) == "real" else -1 for rt in rts_v]))
    rt = rts_v[i]
    k_mid = rt.onsets[len(rt.onsets) // 2]
    filter_plots.plot_track_filters(rt, results[VAL]["IMM"]["runs"][i],
                                    OUT_DIR / "imm_example_val.png",
                                    f"IMM на реальном полёте ({rt.flight_id}), 3 минуты вокруг маневра",
                                    k_range=(max(0, k_mid - 45), min(len(rt.t), k_mid + 45)))

    clean = {name: {m: {k: v for k, v in rep[m].items() if k != "pairs"} for m in METHODS}
             for name, rep in reports.items()}
    (OUT_DIR / "metrics.json").write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.write(OUT_DIR / "summary.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
