"""
Модель измерений обзорной РЛС: из истинной траектории - отметки раз в обзор.

  - РЛС стоит отдельно для каждой трассы (config.RADAR_DIST_RANGE, RADAR_BELOW_MIN_Z),
    положение и шум - из потока "radar", своего для каждой трассы (make_track_rng);
  - моменты обзоров t_k = k * SCAN_PERIOD, истина в них - линейная интерполяция трека
    (для логов и синтетики с шагом 2 с совпадает с точками трека);
  - отметка - дальность, азимут (от севера по часовой), угол места с гауссовым шумом;
  - пропуски: каждая отметка обнаружена с вероятностью RADAR_PD (по умолчанию 1);
  - для фильтров отметка пересчитывается в декартовы координаты ENU трассы, ковариация -
    через якобиан пересчёта в измеренной точке;
  - радиальная скорость (доплер, этап 12) - только при config.DOPPLER_IN_IMM или
    DOPPLER_IN_NET: проекция истинной скорости на линию визирования плюс шум RADAR_SIGMA_VR.
    Истинная скорость - по трём последним точкам истины (формула назад второго порядка,
    точна при постоянном ускорении): будущее положение в измерение не попадает. Шум
    тянется после всех прежних случайных чисел - отметки положения от флага не меняются.
    При config.DOPPLER_TRUTH = "telemetry" (этап 18) у реальных полётов истинная скорость -
    скорость GPS-приёмника из лога (Track.vel_log), где она есть.

Проверяется на этапе 1: эмпирические СКО шума равны заданным, нормированная ошибка
пересчитанных отметок (NEES) в среднем около 3. Доплер - на этапе 12: известный ответ
радиальной скорости, СКО шума, отметки положения те же.
"""

from dataclasses import dataclass

import numpy as np

import config
from common.rng import make_track_rng
from data.track import Track


@dataclass
class RadarTrack:
    flight_id: str
    t: np.ndarray            # (K,) с, моменты обзоров
    truth: np.ndarray        # (K, 3) истинное положение ENU
    radar_pos: np.ndarray    # (3,) положение РЛС в ENU трассы
    polar: np.ndarray        # (K, 3) измеренные дальность (м), азимут, угол места (рад)
    meas: np.ndarray         # (K, 3) отметка в ENU
    meas_cov: np.ndarray     # (K, 3, 3) ковариация отметки в ENU
    detected: np.ndarray     # (K,) bool
    phase: np.ndarray        # (K,) int8, участок истины в момент обзора
    onsets: list             # индексы обзоров начала маневров
    vr: np.ndarray = None    # (K,) м/с, измеренная радиальная скорость (этап 12; nan - нет); None - без доплера


def to_polar(rel: np.ndarray) -> np.ndarray:
    """Вектор от РЛС к цели (N, 3) -> (дальность, азимут от севера по часовой, угол места)."""
    horiz = np.hypot(rel[:, 0], rel[:, 1])
    return np.stack([np.linalg.norm(rel, axis=1), np.arctan2(rel[:, 0], rel[:, 1]),
                     np.arctan2(rel[:, 2], horiz)], axis=1)


def polar_to_enu(polar: np.ndarray) -> np.ndarray:
    r, az, el = polar[:, 0], polar[:, 1], polar[:, 2]
    return np.stack([r * np.cos(el) * np.sin(az), r * np.cos(el) * np.cos(az), r * np.sin(el)], axis=1)


def converted_cov(polar: np.ndarray) -> np.ndarray:
    """Ковариация отметки в ENU через якобиан пересчёта (K, 3, 3)."""
    r, az, el = polar[:, 0], polar[:, 1], polar[:, 2]
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    jac = np.zeros((len(r), 3, 3))
    jac[:, 0] = np.stack([ce * sa, r * ce * ca, -r * se * sa], axis=1)
    jac[:, 1] = np.stack([ce * ca, -r * ce * sa, -r * se * ca], axis=1)
    jac[:, 2] = np.stack([se, np.zeros_like(r), r * ce], axis=1)
    sig = np.diag([config.RADAR_SIGMA_R ** 2, np.radians(config.RADAR_SIGMA_AZ_DEG) ** 2,
                   np.radians(config.RADAR_SIGMA_EL_DEG) ** 2])
    return jac @ sig @ np.transpose(jac, (0, 2, 1))


def true_radial_velocity(truth: np.ndarray, radar_pos: np.ndarray, period: float = config.SCAN_PERIOD,
                         vel: np.ndarray = None) -> np.ndarray:
    """
    Радиальная скорость (K,) по истине: скорость - назад по трём точкам (в обзоре 1 - по двум, в 0 -
    nan). vel (K, 3) - скорость по телеметрии (этап 18): где она есть, берётся она.
    """
    v = np.full_like(truth, np.nan)
    if len(truth) > 1:
        v[1] = (truth[1] - truth[0]) / period
    if len(truth) > 2:
        v[2:] = (3.0 * truth[2:] - 4.0 * truth[1:-1] + truth[:-2]) / (2.0 * period)
    if vel is not None:
        v = np.where(np.isfinite(vel).all(axis=1, keepdims=True), vel, v)
    rel = truth - radar_pos
    return np.einsum("ij,ij->i", v, rel) / np.linalg.norm(rel, axis=1)


def observe(track: Track) -> RadarTrack:
    """Отметки РЛС по истинной траектории трека."""
    rng = make_track_rng("radar", track.flight_id)
    n_scans = int(np.floor(track.duration_s / config.SCAN_PERIOD + 1e-9)) + 1
    t = np.arange(n_scans) * config.SCAN_PERIOD + track.t[0]
    truth = np.column_stack([np.interp(t, track.t, track.pos[:, j]) for j in range(3)])

    center = track.pos[:, :2].mean(axis=0)
    bearing = rng.uniform(0.0, 2.0 * np.pi)
    dist = rng.uniform(*config.RADAR_DIST_RANGE)
    radar_pos = np.array([center[0] + dist * np.sin(bearing), center[1] + dist * np.cos(bearing),
                          track.pos[:, 2].min() - config.RADAR_BELOW_MIN_Z])

    true_polar = to_polar(truth - radar_pos)
    noise = rng.standard_normal((n_scans, 3)) * np.array([
        config.RADAR_SIGMA_R, np.radians(config.RADAR_SIGMA_AZ_DEG), np.radians(config.RADAR_SIGMA_EL_DEG)])
    polar = true_polar + noise
    detected = rng.uniform(size=n_scans) < config.RADAR_PD
    meas = polar_to_enu(polar) + radar_pos

    idx = np.clip(np.searchsorted(track.t, t - 1e-9), 0, len(track.t) - 1)
    phase = track.phase[idx] if track.phase is not None else np.zeros(n_scans, dtype=np.int8)
    onsets = sorted({int(round((track.t[i] - t[0]) / config.SCAN_PERIOD)) for i in track.turn_onsets})
    onsets = [k for k in onsets if 0 <= k < n_scans]
    vr = None
    if config.DOPPLER_IN_IMM or config.DOPPLER_IN_NET:
        vel = None
        if config.DOPPLER_TRUTH == "telemetry" and track.vel_log is not None:
            vel = np.column_stack([np.interp(t, track.t, track.vel_log[:, j]) for j in range(3)])
        vr = true_radial_velocity(truth, radar_pos, vel=vel) + rng.standard_normal(n_scans) * config.RADAR_SIGMA_VR
    return RadarTrack(track.flight_id, t, truth, radar_pos, polar, meas, converted_cov(polar),
                      detected, phase, onsets, vr)
