"""
Нарезка окон для корректора - единственное место, где она делается.

Пример - обзор k трассы: вход - окно из WINDOW_SCANS последних обзоров k-W+1..k,
цель - поправка к прогнозу IMM из обзора k на 1..H обзоров:

    цель[h] = истина(k + h) - прогноз_IMM(k, h)   (по горизонтали)

Всё переводится в систему «по курсу IMM в момент k»: ось 1 - поперёк курса (вправо),
ось 2 - вдоль курса, ось 3 - вверх; начало - оценка IMM в момент k. Так корректор не
зависит от положения и направления полёта.

Признаки обзора j (config.WINDOW_SCANS, комментарий в config.py) строятся только из
отметок и прогона IMM до момента j включительно: окно причинное. Истина используется
только в цели.

Проверяется на этапе 2: признаки, посчитанные по началу трассы, совпадают с
признаками полной трассы (причинность); нулевая поправка даёт ровно прогноз IMM;
поворот и обратный поворот возвращают исходный вектор.
"""

import numpy as np

import config
from filters.models import VX, VY, VZ, X, Y, Z

POS = [X, Y, Z]
VEL = [VX, VY, VZ]


def n_features(n_models: int) -> int:
    return 3 + 3 + 3 + 3 + n_models + n_models + 1


def heading(est: np.ndarray) -> np.ndarray:
    """Курс IMM (рад, от севера по часовой) по оценке скорости."""
    return np.arctan2(est[..., VX], est[..., VY])


def to_local(vec: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Поворот горизонтальных компонент (..., 2 или 3) в систему (поперёк, вдоль); psi - курс."""
    c, s = np.cos(psi), np.sin(psi)
    out = vec.copy()
    out[..., 0] = c * vec[..., 0] - s * vec[..., 1]
    out[..., 1] = s * vec[..., 0] + c * vec[..., 1]
    return out


def to_global(vec: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Обратный поворот к to_local."""
    c, s = np.cos(psi), np.sin(psi)
    out = vec.copy()
    out[..., 0] = c * vec[..., 0] + s * vec[..., 1]
    out[..., 1] = -s * vec[..., 0] + c * vec[..., 1]
    return out


def build_windows(rt, run, window: int = config.WINDOW_SCANS, horizon: int = config.HORIZON_SCANS,
                  with_target: bool = True) -> dict:
    """
    Все примеры одной трассы. Возвращает словарь:
      x (N, W, F) float32, y (N, H, 2) float32 (если with_target), k (N,), psi (N,).
    Пример есть, если окно целиком после инициализации IMM, k >= WARMUP_SCANS и
    (для цели) k + H внутри трассы.
    """
    k_total = len(rt.t)
    valid = ~np.isnan(run.est[:, 0])
    if not valid.any():
        return _empty(window, horizon, len(config.IMM_MODELS), with_target)
    first = int(np.argmax(valid))
    k_lo = max(first + window - 1, config.WARMUP_SCANS)
    k_hi = k_total - horizon if with_target else k_total
    ks = np.arange(k_lo, k_hi)
    if len(ks) == 0:
        return _empty(window, horizon, run.mu.shape[1], with_target)

    innov = np.nan_to_num(run.innov)
    nis = np.log1p(np.nan_to_num(run.nis))
    mu = run.mu
    rng_km = rt.polar[:, 0:1] / 1000.0

    idx = ks[:, None] + np.arange(-window + 1, 1)[None, :]       # (N, W)
    est_k = run.est[ks]                                            # (N, 9)
    psi = heading(est_k)                                           # (N,)
    psi_w = psi[:, None]
    origin = est_k[:, POS][:, None, :]                             # (N, 1, 3)
    feats = [
        to_local(rt.meas[idx] - origin, psi_w),
        to_local(run.est[idx][..., POS] - origin, psi_w),
        to_local(run.est[idx][..., VEL], psi_w),
        to_local(innov[idx], psi_w),
        mu[idx],
        nis[idx],
        rng_km[idx],
    ]
    x = np.concatenate(feats, axis=-1).astype(np.float32)
    out = {"x": x, "k": ks, "psi": psi}
    if with_target:
        h = np.arange(1, horizon + 1)
        target = rt.truth[ks[:, None] + h[None, :], :2] - run.forecast[ks][:, :, :2]   # (N, H, 2)
        out["y"] = to_local(target, psi[:, None]).astype(np.float32)
    ok = ~np.isnan(out["x"]).any(axis=(1, 2))
    if with_target:
        ok &= ~np.isnan(out["y"]).any(axis=(1, 2))
    return {key: val[ok] for key, val in out.items()}


def _empty(window, horizon, m, with_target):
    out = {"x": np.zeros((0, window, n_features(m)), np.float32), "k": np.zeros(0, int), "psi": np.zeros(0)}
    if with_target:
        out["y"] = np.zeros((0, horizon, 2), np.float32)
    return out


def apply_correction(forecast: np.ndarray, ks: np.ndarray, psi: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Прогноз гибрида: прогноз IMM (K, H, 3) плюс поправка (N, H, 2) в обзорах ks; высота - от IMM."""
    out = forecast.copy()
    out[ks, :, :2] += to_global(corr.astype(float), psi[:, None])
    return out
