"""
Оракул (этап 11): прогноз по точному текущему состоянию цели - сколько ошибки остаётся от
непредсказуемости будущего движения, а не от шума РЛС и не от фильтра.

Состояние в обзоре k берётся по трём последним точкам истины (k-2, k-1, k) - только прошлое,
будущее не используется. Две хорды дают поворот курса за обзор и длину дуги; для движения с
постоянной скоростью и скоростью разворота (прямая и окружность) это состояние точное:
  w   = (курс хорды k-1 -> k  -  курс хорды k-2 -> k-1) / T;
  psi = курс хорды k-1 -> k + w T / 2  (хорда смотрит в середину дуги);
  V   = |хорда| / T * (w T / 2) / sin(w T / 2).
Прогноз:
  oracle_forecast(..., turn=False) - CV: точная текущая скорость, движение прямо;
  oracle_forecast(..., turn=True)  - CT: точные скорость и скорость разворота, дуга;
  best_of - задним числом лучший из прогнозов в каждом обзоре (по средней ошибке на
            горизонтах): идеальный выбор модели; это не прогноз, а нижняя граница для
            методов, переключающих CV и CT.
Высота - постоянная вертикальная скорость по двум последним точкам (метрики - по горизонтали).

На реальных полётах истина - GPS с собственной погрешностью, поэтому «точное состояние»
там тоже не идеально; на синтетике истина гладкая.

Проверяется на этапе 11: на прямой оба оракула точны, на окружности точен CT; прогноз в
обзоре k не зависит от истины после k.
"""

import warnings

import numpy as np

import config


def oracle_forecast(truth: np.ndarray, turn: bool, horizon: int = config.HORIZON_SCANS,
                    period: float = config.SCAN_PERIOD) -> np.ndarray:
    """Прогноз (K, H, 3) по точной истине truth (K, 3); nan в обзорах 0 и 1."""
    k_total = len(truth)
    out = np.full((k_total, horizon, 3), np.nan)
    if k_total < 3:
        return out
    p = truth[:, :2]
    c1, c2 = p[1:-1] - p[:-2], p[2:] - p[1:-1]                     # хорды (k-2 -> k-1), (k-1 -> k)
    psi1, psi2 = np.arctan2(c1[:, 0], c1[:, 1]), np.arctan2(c2[:, 0], c2[:, 1])
    w = np.angle(np.exp(1j * (psi2 - psi1))) / period if turn else np.zeros(len(c2))
    half = np.angle(np.exp(1j * (psi2 - psi1))) / 2                 # w T / 2 истинного движения
    chord = np.linalg.norm(c2, axis=1)
    safe = np.abs(half) > 1e-9
    speed = chord / period * np.where(safe, half / np.where(safe, np.sin(half), 1.0), 1.0)
    psi = psi2 + half
    vz = (truth[2:, 2] - truth[1:-1, 2]) / period
    small = np.abs(w) < 1e-9
    w_safe = np.where(small, 1.0, w)
    for h in range(1, horizon + 1):
        t = h * period
        dx = np.where(small, speed * t * np.sin(psi), speed / w_safe * (np.cos(psi) - np.cos(psi + w * t)))
        dy = np.where(small, speed * t * np.cos(psi), speed / w_safe * (np.sin(psi + w * t) - np.sin(psi)))
        out[2:, h - 1] = np.column_stack([p[2:, 0] + dx, p[2:, 1] + dy, truth[2:, 2] + vz * t])
    return out


def best_of(truth: np.ndarray, forecasts: list) -> np.ndarray:
    """Задним числом в каждом обзоре - прогноз с наименьшей средней горизонтальной ошибкой."""
    k_total, horizon, _ = forecasts[0].shape
    errs = []
    for fc in forecasts:
        e = np.full((k_total, horizon), np.nan)
        for h in range(1, horizon + 1):
            e[:k_total - h, h - 1] = np.linalg.norm(fc[:k_total - h, h - 1, :2] - truth[h:, :2], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            errs.append(np.nanmean(e, axis=1))       # нет ни одного горизонта - nan
    errs = np.stack(errs)                             # (методы, K)
    choice = np.argmin(np.where(np.isnan(errs), np.inf, errs), axis=0)
    out = np.stack(forecasts)[choice, np.arange(k_total)]
    out[np.isnan(errs).all(axis=0)] = np.nan
    return out
