"""
Метрики прогноза - единые для всех этапов. В скрипты этапов не копировать.

Прогноз из обзора k на h обзоров сравнивается с истиной в обзоре k + h. Пара (k, h)
учитывается, если k >= WARMUP_SCANS, k + h есть в трассе и прогноз существует.

Основные метрики - по горизонтали (x, y): маневры горизонтальные, а угол места РЛС
меряет грубо (ошибка высоты ~ дальность * СКО угла места - десятки метров), и в 3D
она заслоняла бы всё остальное. 3D-ошибка считается отдельно (ade_3d).

  MAE  - среднее |ошибки| по компонентам x и y;
  RMSE - корень из среднего квадрата горизонтальной ошибки;
  ADE  - средняя горизонтальная ошибка по всем горизонтам;
  FDE  - средняя горизонтальная ошибка на последнем горизонте H;
  R2   - 1 - сумма квадратов ошибок / сумма квадратов отклонений истинного смещения
         от среднего; смещение - от последней отметки (k) до истины в k + h.

Динамическая ошибка: для каждого начала маневра (обзор onset) берутся прогнозы из
обзоров onset + m, m в DYN_OFFSETS, на горизонте h; кривая - средняя ошибка по m.
  пик - максимум кривой при m >= 0;
  площадь - среднее за первые DYN_AREA_SCANS обзоров после начала маневра;
  восстановление - первый m после пика, где ошибка не выше уровня прямой,
  умноженного на DYN_RECOVERY_FACTOR (уровень прямой - средняя ошибка на этом
  горизонте из обзоров участка "straight").

Проверяется на этапе 1 на подставных данных с известным ответом.
"""

import numpy as np

import config
from data.phases import PHASE_CODE


def forecast_pairs(rt, forecast: np.ndarray, warmup: int = config.WARMUP_SCANS) -> dict:
    """Все пары (k, h) одной трассы: ошибка, смещение, участок в момент k."""
    k_total, horizon, _ = forecast.shape
    rows = {"err": [], "disp": [], "h": [], "k": [], "phase": []}
    for h in range(1, horizon + 1):
        k = np.arange(warmup, k_total - h)
        if len(k) == 0:
            continue
        pred = forecast[k, h - 1]
        ok = ~np.isnan(pred).any(axis=1)
        k = k[ok]
        rows["err"].append(forecast[k, h - 1] - rt.truth[k + h])
        rows["disp"].append(rt.truth[k + h] - rt.meas[k])
        rows["h"].append(np.full(len(k), h))
        rows["k"].append(k)
        rows["phase"].append(rt.phase[k])
    if not rows["err"]:
        return {"err": np.zeros((0, 3)), "disp": np.zeros((0, 3)), "h": np.zeros(0, int),
                "k": np.zeros(0, int), "phase": np.zeros(0, np.int8)}
    return {key: np.concatenate(val) for key, val in rows.items()}


def concat_pairs(parts: list) -> dict:
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def summarize(pairs: dict, horizon: int = config.HORIZON_SCANS) -> dict:
    """Метрики по набору пар (k, h)."""
    err, disp, h = pairs["err"], pairs["disp"], pairs["h"]
    if len(err) == 0:
        return {name: float("nan") for name in ("mae", "rmse", "ade", "fde", "r2", "ade_3d", "n")}
    e2 = err[:, :2]
    dist = np.linalg.norm(e2, axis=1)
    d2 = disp[:, :2]
    sst = float(((d2 - d2.mean(axis=0)) ** 2).sum())
    return {
        "mae": float(np.abs(e2).mean()),
        "rmse": float(np.sqrt((dist ** 2).mean())),
        "ade": float(dist.mean()),
        "fde": float(dist[h == horizon].mean()) if (h == horizon).any() else float("nan"),
        "r2": 1.0 - float((dist ** 2).sum()) / sst if sst > 0 else float("nan"),
        "ade_3d": float(np.linalg.norm(err, axis=1).mean()),
        "n": int(len(err)),
    }


def ade_by_horizon(pairs: dict, horizon: int = config.HORIZON_SCANS) -> np.ndarray:
    dist = np.linalg.norm(pairs["err"][:, :2], axis=1)
    return np.array([dist[pairs["h"] == h].mean() if (pairs["h"] == h).any() else np.nan
                     for h in range(1, horizon + 1)])


def select(pairs: dict, mask: np.ndarray) -> dict:
    return {key: val[mask] for key, val in pairs.items()}


def dynamic_curve(radar_tracks: list, forecasts: list, h: int,
                  offsets=config.DYN_OFFSETS) -> tuple:
    """Кривая динамической ошибки: (m, средняя ошибка, число маневров) по всем трассам."""
    ms = np.arange(offsets[0], offsets[1] + 1)
    sums = np.zeros(len(ms))
    counts = np.zeros(len(ms), dtype=int)
    for rt, fc in zip(radar_tracks, forecasts):
        k_total = len(rt.t)
        for onset in rt.onsets:
            for i, m in enumerate(ms):
                k = onset + m
                if k < config.WARMUP_SCANS or k + h >= k_total or np.isnan(fc[k, h - 1]).any():
                    continue
                sums[i] += np.linalg.norm(fc[k, h - 1, :2] - rt.truth[k + h, :2])
                counts[i] += 1
    with np.errstate(invalid="ignore"):
        return ms, sums / counts, counts


def straight_level(pairs: dict, h: int) -> float:
    sel = (pairs["h"] == h) & (pairs["phase"] == PHASE_CODE["straight"])
    return float(np.linalg.norm(pairs["err"][sel, :2], axis=1).mean()) if sel.any() else float("nan")


def dynamic_stats(ms: np.ndarray, curve: np.ndarray, level: float) -> dict:
    after = ms >= 0
    m_after, c_after = ms[after], curve[after]
    valid = ~np.isnan(c_after)
    if not valid.any():
        return {"peak": float("nan"), "peak_m": float("nan"), "area": float("nan"), "recovery": float("nan")}
    i_peak = int(np.nanargmax(c_after))
    area_sel = (m_after < config.DYN_AREA_SCANS) & valid
    recovery = float("nan")
    for m, c in zip(m_after[i_peak:], c_after[i_peak:]):
        if not np.isnan(c) and c <= level * config.DYN_RECOVERY_FACTOR:
            recovery = float(m)
            break
    return {"peak": float(c_after[i_peak]), "peak_m": float(m_after[i_peak]),
            "area": float(c_after[area_sel].mean()), "recovery": recovery}
