"""
Сравнение методов на одних и тех же трассах: таблицы для сводки, отчёт для
metrics.json и рисунки. Определения метрик - только из eval/metrics.py.

Сравнение честное, если у всех методов одинаковые пары (k, h): прогнозы вне
общего набора обзоров нужно заранее заменить на nan (mask_forecast).

Набор из реальных и сгенерированных трасс (вариант данных merged) дополнительно
разбирается по источнику - только в тексте сводки и в отчёте, рисунки строятся по
всему набору (решение пользователя 23.10.2026: в рисунках диплома этой разбивки нет).
"""

from pathlib import Path

import numpy as np

import config
from data.phases import PHASE_CODE
from eval import metrics
from viz import filter_plots

TURN_CODES = (PHASE_CODE["entry"], PHASE_CODE["turn"])
SOURCES = {"real": "реальные", "synth": "сгенерированные"}


def source_of(rt) -> str:
    """Источник трассы по имени: у сгенерированных flight_id начинается с synth (synth_, synthT_, synthX...)."""
    return "synth" if rt.flight_id.startswith("synth") else "real"


def ade_by_source(rts: list, fcs: list) -> dict:
    """ADE отдельно по реальным и сгенерированным трассам; {} - если набор не смешанный (вариант base)."""
    groups = {s: [i for i, rt in enumerate(rts) if source_of(rt) == s] for s in SOURCES}
    if not all(groups.values()):
        return {}
    return {s: metrics.summarize(metrics.concat_pairs([metrics.forecast_pairs(rts[i], fcs[i]) for i in idx]))["ade"]
            for s, idx in groups.items()}


def source_section(summary, label: str, table: dict) -> None:
    """
    Раздел сводки «ADE по источнику» по отчётам {метод: [отчёт по seed]} с ключами ade, ade_real,
    ade_synth (summarize_method этапа 4). Методы без разбивки пропускаются; в base раздела нет.
    """
    rows = {m: reps for m, reps in table.items() if reps and "ade_real" in reps[0]}
    if not rows:
        return
    def cell(vals):
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    width = max(len(m) for m in rows) + 1
    lines = [f"{'метод':{width}s} {'все':>12s}" + "".join(f" {v:>16s}" for v in SOURCES.values())]
    for m, reps in rows.items():
        cells = [cell(np.array([r[k] for r in reps])) for k in ("ade", *(f"ade_{s}" for s in SOURCES))]
        lines.append(f"{m:{width}s} {cells[0]:>12s}" + "".join(f" {c:>16s}" for c in cells[1:]))
    summary.section(f"{label}: ADE по источнику трасс (м; при нескольких seed - среднее ± СКО), только в тексте",
                    lines)


def mask_forecast(forecast: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Прогноз только в обзорах ks, остальное - nan."""
    out = np.full_like(forecast, np.nan)
    out[ks] = forecast[ks]
    return out


def compare(rts: list, forecasts: dict, summary, label: str, out_dir: Path = None, tag: str = "") -> dict:
    """forecasts: {метод: [прогноз (K, H, 3) по каждой трассе]}. Пишет разделы сводки, возвращает отчёт."""
    report, curves, levels = {}, {}, {}
    width = max(len(m) for m in forecasts) + 1
    lines = [f"{'метод':{width}s} {'MAE':>6s} {'RMSE':>6s} {'ADE':>6s} {'FDE':>6s} {'R2':>6s} {'пар':>6s}  "
             f"ADE по горизонтам, м"]
    phase_lines = [f"{'метод':{width}s} " + " ".join(f"{p:>9s}" for p in config.PHASES if p != "slow")]
    dyn_lines = [f"{'метод':{width}s} {'h':>2s} {'прямая':>7s} {'пик':>7s} {'при m':>6s} {'площадь':>8s}"]
    for name, fcs in forecasts.items():
        pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)])
        s = metrics.summarize(pairs)
        by_h = metrics.ade_by_horizon(pairs)
        by_phase = {p: metrics.summarize(metrics.select(pairs, pairs["phase"] == c))["ade"]
                    for p, c in PHASE_CODE.items()}
        maneuver = metrics.summarize(metrics.select(pairs, np.isin(pairs["phase"], TURN_CODES)))
        report[name] = {"overall": s, "by_horizon": by_h.tolist(), "by_phase": by_phase, "maneuver": maneuver}
        lines.append(f"{name:{width}s} {s['mae']:6.1f} {s['rmse']:6.1f} {s['ade']:6.1f} {s['fde']:6.1f} "
                     f"{s['r2']:6.3f} {s['n']:6d}  " + " ".join(f"{v:6.1f}" for v in by_h))
        phase_lines.append(f"{name:{width}s} " + " ".join(f"{by_phase[p]:9.1f}" for p in config.PHASES if p != "slow"))
        for h in (1, config.HORIZON_SCANS):
            ms, curve, counts = metrics.dynamic_curve(rts, fcs, h)
            level = metrics.straight_level(pairs, h)
            st = metrics.dynamic_stats(ms, curve, level)
            curves[(name, h)], levels[(name, h)] = (ms, curve, counts), level
            report[name][f"dynamic_h{h}"] = {"level": level, **st, "curve": np.nan_to_num(curve).tolist()}
            dyn_lines.append(f"{name:{width}s} {h:2d} {level:7.1f} {st['peak']:7.1f} {st['peak_m']:6.0f} "
                             f"{st['area']:8.1f}")
    summary.section(f"{label}: метрики по горизонтали (м)", lines)
    summary.section(f"{label}: ADE по участкам (м)", phase_lines)
    summary.section(f"{label}: динамическая ошибка (м)", dyn_lines)
    by_source = {name: ade_by_source(rts, fcs) for name, fcs in forecasts.items()}
    if all(by_source.values()):
        src_lines = [f"{'метод':{width}s} {'все':>8s}" + "".join(f" {v:>16s}" for v in SOURCES.values())]
        for name, src in by_source.items():
            report[name]["by_source"] = src
            src_lines.append(f"{name:{width}s} {report[name]['overall']['ade']:8.1f}"
                             + "".join(f" {src[s]:16.1f}" for s in SOURCES))
        summary.section(f"{label}: ADE по источнику трасс (м), только в тексте", src_lines)
    if out_dir is not None:
        filter_plots.plot_dynamic_curves(curves, levels, out_dir / f"dynamic_error_{tag}.png",
                                         f"Динамическая ошибка: {label}")
        filter_plots.plot_by_phase({m: report[m]["by_phase"] for m in report}, out_dir / f"ade_by_phase_{tag}.png",
                                   f"Ошибка по участкам: {label}")
        filter_plots.plot_by_horizon({m: report[m]["by_horizon"] for m in report},
                                     out_dir / f"ade_by_horizon_{tag}.png", f"Ошибка по горизонтам: {label}")
    return report
