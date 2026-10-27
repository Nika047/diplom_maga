"""
Оценка методов для этапов версии 2: прогнозы по трассам -> метрики, динамическая ошибка, ошибки по полётам.

Определения метрик - только в eval/metrics.py; здесь сборка отчёта метода на наборе оценки. Отчёт:
ade, fde, rmse, mae, r2, n; ade по участкам (straight, entry, turn, exit); ade по источнику трасс
(real / synth, если набор смешанный); динамическая ошибка на горизонте H (пик, площадь, восстановление -
время реакции на маневр, обзоров); "flights" - суммы и числа по полётам для бутстрепа (eval/bootstrap.py).
"""

import numpy as np

import config
from data.phases import PHASE_CODE
from data.windows import apply_correction, apply_e2e
from eval import bootstrap, metrics
from eval.compare import ade_by_source
from filters.extrapolation import cv_extrapolate
from filters.imm import IMM
from pipeline.casedata import CaseData
from pipeline.data import flight_key

PHASE_NAMES = ("straight", "entry", "turn", "exit")


def summarize_method(rts: list, fcs: list) -> dict:
    pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)])
    s = metrics.summarize(pairs)
    out = {"ade": s["ade"], "fde": s["fde"], "r2": s["r2"], "rmse": s["rmse"], "mae": s["mae"], "n": s["n"]}
    for p in PHASE_NAMES:
        out[p] = metrics.summarize(metrics.select(pairs, pairs["phase"] == PHASE_CODE[p]))["ade"]
    out.update({f"ade_{src}": v for src, v in ade_by_source(rts, fcs).items()})
    return out


def flights_of(tracks: list, rts: list, fcs: list) -> dict:
    keys, sums, counts = bootstrap.flight_sums([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)],
                                               [flight_key(tr) for tr in tracks])
    return {"keys": keys, "sums": sums.tolist(), "counts": counts.tolist()}


def dynamic_summary(rts: list, fcs: list) -> dict:
    h = config.HORIZON_SCANS
    pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)])
    ms, curve, _ = metrics.dynamic_curve(rts, fcs, h)
    st = metrics.dynamic_stats(ms, curve, metrics.straight_level(pairs, h))
    return {"dyn_peak": st["peak"], "dyn_area": st["area"], "dyn_recovery": st["recovery"]}


def report(cd: CaseData, name: str, fcs: list) -> dict:
    """Полный отчёт метода на наборе оценки name (fcs - прогнозы по трассам, уже по обзорам ks)."""
    _, rts, _ = cd.eval(name)
    out = summarize_method(rts, fcs)
    out.update(dynamic_summary(rts, fcs))
    out["flights"] = flights_of(cd.sets[name], rts, fcs)
    return out


def reports(cd: CaseData, fcs_by_set: dict) -> dict:
    return {name: report(cd, name, fcs) for name, fcs in fcs_by_set.items()}


def hybrid_forecasts(cd: CaseData, name: str, corr: np.ndarray) -> list:
    """Прогноз IMM плюс поправка сети (N, H, 2) на обзорах окон набора name."""
    ws, _, runs = cd.eval(name)
    fcs = [apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i], corr[ws.track == i])
           for i, run in enumerate(runs)]
    return cd.mask(name, fcs)


def e2e_forecasts(cd: CaseData, name: str, disp: np.ndarray) -> list:
    """Прогноз сети без IMM: отметка плюс смещение (N, H, 2)."""
    ws, rts, _ = cd.eval(name)
    fcs = [apply_e2e(rt, ws.k[ws.track == i], ws.psi_e2e[ws.track == i], disp[ws.track == i])
           for i, rt in enumerate(rts)]
    return cd.mask(name, fcs)


def imm_forecasts(cd: CaseData, name: str) -> list:
    _, _, runs = cd.eval(name)
    return cd.mask(name, [r.forecast for r in runs])


def classic_forecasts(cd: CaseData, name: str, imm_default: dict = None) -> dict:
    """CV (линейная экстраполяция), KF (одна модель CV), IMM случая; при imm_default - ещё IMM с исходными параметрами."""
    from common.cases import config_override

    _, rts, runs = cd.eval(name)
    out = {"CV": cd.mask(name, [cv_extrapolate(rt) for rt in rts])}
    with config_override(cd.override()):
        kf = IMM(("CV",), mu0=(1.0,))
        out["KF"] = cd.mask(name, [kf.run(rt).forecast for rt in rts])
    out["IMM"] = cd.mask(name, [r.forecast for r in runs])
    if imm_default is not None:
        with config_override({**cd.override(), **imm_default}):
            imm = IMM()
            out["IMM (исходный)"] = cd.mask(name, [imm.run(rt).forecast for rt in rts])
    return out


def cell(vals, fmt: str = "{:.1f}") -> str:
    vals = np.asarray(vals, float)
    if len(vals) == 1:
        return fmt.format(vals[0])
    return (fmt + "±" + fmt).format(vals.mean(), vals.std())
