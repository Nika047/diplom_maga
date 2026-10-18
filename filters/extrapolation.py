"""
Линейная экстраполяция (CV) - самый простой базовый метод.

В каждом обзоре k по последним CV_LS_POINTS обнаруженным отметкам методом
наименьших квадратов строится прямая положение(время) по каждой оси и
продолжается на 1..H обзоров вперёд. Фильтрации нет, только сглаживание по окну.

Проверяется на этапе 1: на прямой без шума прогноз точный.
"""

import numpy as np

import config
from radar.model import RadarTrack


def cv_extrapolate(rt: RadarTrack, n_points: int = config.CV_LS_POINTS,
                   horizon: int = config.HORIZON_SCANS) -> np.ndarray:
    """Прогноз (K, H, 3); nan, пока отметок меньше n_points."""
    k_total = len(rt.t)
    forecast = np.full((k_total, horizon, 3), np.nan)
    det = np.flatnonzero(rt.detected)
    steps = np.arange(1, horizon + 1) * config.SCAN_PERIOD
    for pos in range(n_points - 1, len(det)):
        idx = det[pos - n_points + 1: pos + 1]
        k = det[pos]
        tt = rt.t[idx] - rt.t[k]
        a = np.column_stack([np.ones_like(tt), tt])
        coef, *_ = np.linalg.lstsq(a, rt.meas[idx], rcond=None)  # (2, 3)
        forecast[k] = coef[0] + np.outer(steps, coef[1])
    return forecast
