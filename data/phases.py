"""
Разметка участков истинной траектории: прямая / вход в маневр / маневр / выход / медленно.

Метки - это эталон для разбора результатов и для прогноза начала маневра, а не
входы моделей. Поэтому здесь допустимо центральное сглаживание курса: метка в
момент k может зависеть от точек после k.

Маневр - участок, где |w| поднялась выше OMEGA_TURN_ON и не опустилась ниже
OMEGA_TURN_OFF (гистерезис), длительностью не меньше TURN_MIN_DURATION_S.
Первые ENTRY_DURATION_S маневра - "entry", EXIT_DURATION_S после него - "exit".
Где горизонтальная скорость ниже PHASE_V_MIN, курс не определён - "slow".

Проверяется на этапе 0 на подставных данных: прямая - "straight", окружность -
"entry" и затем "turn".
"""

import numpy as np

import config
from data.track import Track

PHASE_CODE = {name: i for i, name in enumerate(config.PHASES)}


def compute_kinematics(t: np.ndarray, pos: np.ndarray) -> tuple:
    """Горизонтальная скорость (м/с), курс (рад, от севера по часовой), |w| со знаком (град/с)."""
    n = len(t)
    v = np.zeros((n, 2))
    if n >= 2:
        v[1:] = np.diff(pos[:, :2], axis=0) / np.diff(t)[:, None]
        v[0] = v[1]
    speed = np.hypot(v[:, 0], v[:, 1])
    heading = np.arctan2(v[:, 0], v[:, 1])
    unwrapped = np.unwrap(heading)
    k = config.PHASE_SMOOTH_POINTS
    if k > 1 and n >= k:
        pad = k // 2
        padded = np.pad(unwrapped, pad, mode="edge")
        smooth = np.convolve(padded, np.ones(k) / k, mode="valid")
    else:
        smooth = unwrapped
    omega = np.degrees(np.gradient(smooth, t)) if n >= 2 else np.zeros(n)
    omega[speed < config.PHASE_V_MIN] = np.nan
    return speed, heading, omega


def label_phases(t: np.ndarray, speed: np.ndarray, omega: np.ndarray) -> tuple:
    """Коды участков (int8, по config.PHASES) и индексы начала маневров."""
    n = len(t)
    abs_w = np.abs(omega)
    in_turn = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        w = abs_w[i]
        if np.isnan(w):
            active = False
        elif not active and w >= config.OMEGA_TURN_ON:
            active = True
        elif active and w < config.OMEGA_TURN_OFF:
            active = False
        in_turn[i] = active

    phase = np.full(n, PHASE_CODE["straight"], dtype=np.int8)
    onsets = []
    i = 0
    while i < n:
        if not in_turn[i]:
            i += 1
            continue
        j = i
        while j < n and in_turn[j]:
            j += 1
        if t[j - 1] - t[i] >= config.TURN_MIN_DURATION_S:
            onsets.append(i)
            seg = np.arange(i, j)
            phase[seg] = PHASE_CODE["turn"]
            phase[seg[t[seg] - t[i] < config.ENTRY_DURATION_S]] = PHASE_CODE["entry"]
            after = np.arange(j, n)
            after = after[t[after] - t[j - 1] <= config.EXIT_DURATION_S]
            for a in after:
                if in_turn[a]:
                    break
                phase[a] = PHASE_CODE["exit"]
        i = j
    phase[speed < config.PHASE_V_MIN] = PHASE_CODE["slow"]
    return phase, onsets


def annotate(track: Track) -> Track:
    """Заполняет speed, omega, phase, turn_onsets у трека."""
    track.speed, _, track.omega = compute_kinematics(track.t, track.pos)
    track.phase, track.turn_onsets = label_phases(track.t, track.speed, track.omega)
    return track


def turn_omegas(tracks: list) -> np.ndarray:
    """|w| (град/с) во всех точках участков entry и turn."""
    codes = (PHASE_CODE["entry"], PHASE_CODE["turn"])
    vals = [np.abs(tr.omega[np.isin(tr.phase, codes)]) for tr in tracks]
    vals = np.concatenate(vals) if vals else np.array([])
    return vals[~np.isnan(vals)]


def phase_shares(tracks: list) -> dict:
    """Доли времени по участкам (по числу точек)."""
    codes = np.concatenate([tr.phase for tr in tracks]) if tracks else np.array([], dtype=np.int8)
    total = max(len(codes), 1)
    return {name: float((codes == code).sum()) / total for name, code in PHASE_CODE.items()}
