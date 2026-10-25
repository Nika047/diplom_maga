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

Окна сквозных моделей (без IMM, этап 3) - build_windows_e2e: те же обзоры k, только
отметки РЛС; цель - смещение от последней отметки. Проверяется на этапе 3: нулевое
смещение даёт прогноз «последняя отметка», признаки причинные.

Метки начала маневра (этап 5) - onset_labels. Проверяется на этапе 5 на подставной
трассе с известными началами маневров.

Класс будущего движения «прямо / влево / вправо» (этап 14, цель гипотез) - turn_class.
Проверяется на этапе 14 на подставной трассе.

Доплер (этап 12): при config.DOPPLER_IN_NET и радиальной скорости у трассы к признакам
обзора добавляются измеренная радиальная скорость и её невязка к прогнозу IMM (группа
«доплер», в конце). Без флага признаки прежние. Причинность - та же проверка на этапе 12.
"""

import numpy as np

import config
from filters.models import VX, VY, VZ, X, Y, Z

POS = [X, Y, Z]
VEL = [VX, VY, VZ]


def n_features(n_models: int, doppler: bool = False) -> int:
    return 3 + 3 + 3 + 3 + n_models + n_models + 1 + (2 if doppler else 0)


def feature_groups(n_models: int, doppler: bool = False) -> dict:
    """Группы признаков гибрида -> индексы столбцов (порядок - как в build_windows)."""
    sizes = [("отметка", 3), ("оценка IMM", 3), ("скорость IMM", 3), ("невязка", 3),
             ("вероятности моделей", n_models), ("NIS моделей", n_models), ("дальность", 1)]
    if doppler:
        sizes.append(("доплер", 2))
    out, start = {}, 0
    for name, size in sizes:
        out[name] = list(range(start, start + size))
        start += size
    return out


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


def held(rt) -> tuple:
    """
    Отметка и полярные координаты, доступные в каждом обзоре (этап 6а): при пропуске
    удерживается последняя полученная отметка, до первой - nan. Без пропусков совпадают
    с rt.meas и rt.polar, поэтому признаки этапов 2-5 не меняются.
    """
    det = np.asarray(rt.detected, dtype=bool)
    if det.all():
        return rt.meas, rt.polar
    last = np.where(det, np.arange(len(det)), -1)
    last = np.maximum.accumulate(last)
    meas = np.where((last >= 0)[:, None], rt.meas[np.maximum(last, 0)], np.nan)
    polar = np.where((last >= 0)[:, None], rt.polar[np.maximum(last, 0)], np.nan)
    return meas, polar


def build_windows(rt, run, window: int = config.WINDOW_SCANS, horizon: int = config.HORIZON_SCANS,
                  with_target: bool = True) -> dict:
    """
    Все примеры одной трассы. Возвращает словарь:
      x (N, W, F) float32, y (N, H, 2) float32 (если with_target), k (N,), psi (N,).
    Пример есть, если окно целиком после инициализации IMM, k >= WARMUP_SCANS и
    (для цели) k + H внутри трассы.
    """
    k_total = len(rt.t)
    doppler = config.DOPPLER_IN_NET and rt.vr is not None and run.vr_innov is not None
    valid = ~np.isnan(run.est[:, 0])
    if not valid.any():
        return _empty(window, horizon, len(config.IMM_MODELS), with_target, doppler)
    first = int(np.argmax(valid))
    k_lo = max(first + window - 1, config.WARMUP_SCANS)
    k_hi = k_total - horizon if with_target else k_total
    ks = np.arange(k_lo, k_hi)
    if len(ks) == 0:
        return _empty(window, horizon, run.mu.shape[1], with_target, doppler)

    meas, polar = held(rt)
    innov = np.nan_to_num(run.innov)          # при пропуске невязки нет - 0
    nis = np.log1p(np.nan_to_num(run.nis))
    mu = run.mu
    rng_km = polar[:, 0:1] / 1000.0

    idx = ks[:, None] + np.arange(-window + 1, 1)[None, :]       # (N, W)
    est_k = run.est[ks]                                            # (N, 9)
    psi = heading(est_k)                                           # (N,)
    psi_w = psi[:, None]
    origin = est_k[:, POS][:, None, :]                             # (N, 1, 3)
    feats = [
        to_local(meas[idx] - origin, psi_w),
        to_local(run.est[idx][..., POS] - origin, psi_w),
        to_local(run.est[idx][..., VEL], psi_w),
        to_local(innov[idx], psi_w),
        mu[idx],
        nis[idx],
        rng_km[idx],
    ]
    if doppler:
        # Радиальная скорость - как отметка: при пропуске удерживается последняя; невязка при пропуске 0
        last = np.maximum.accumulate(np.where(rt.detected, np.arange(k_total), -1))
        vr = np.where(last >= 0, rt.vr[np.maximum(last, 0)], np.nan)
        feats += [vr[idx][..., None], np.nan_to_num(run.vr_innov)[idx][..., None]]
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


def _empty(window, horizon, m, with_target, doppler=False):
    out = {"x": np.zeros((0, window, n_features(m, doppler)), np.float32), "k": np.zeros(0, int),
           "psi": np.zeros(0)}
    if with_target:
        out["y"] = np.zeros((0, horizon, 2), np.float32)
    return out


def apply_correction(forecast: np.ndarray, ks: np.ndarray, psi: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Прогноз гибрида: прогноз IMM (K, H, 3) плюс поправка (N, H, 2) в обзорах ks; высота - от IMM."""
    out = forecast.copy()
    out[ks, :, :2] += to_global(corr.astype(float), psi[:, None])
    return out


# ============================================================================
# Сквозные модели (без IMM) - для сравнения: помогает ли IMM нейросети
# ============================================================================

N_FEATURES_E2E = 4


def build_windows_e2e(rt, ks: np.ndarray, window: int = config.WINDOW_SCANS,
                      horizon: int = config.HORIZON_SCANS, with_target: bool = True) -> dict:
    """
    Окна только из отметок РЛС для тех же обзоров ks, что у гибрида (одинаковые пары
    для сравнения). Система - по курсу, оценённому прямой по последним CV_LS_POINTS
    отметкам; начало - отметка в обзоре k.

    Признаки обзора j: отметка j - отметка k (3), дальность до РЛС, км (1). При
    пропуске - последняя полученная отметка (held).
    Цель - смещение истины(k + h) от отметки k по горизонтали (H, 2), а не поправка.
    """
    ks = np.asarray(ks, dtype=int)
    if len(ks) == 0:
        out = {"x": np.zeros((0, window, N_FEATURES_E2E), np.float32), "k": ks, "psi": np.zeros(0)}
        if with_target:
            out["y"] = np.zeros((0, horizon, 2), np.float32)
        return out
    meas, polar = held(rt)
    n_ls = config.CV_LS_POINTS
    ls_idx = ks[:, None] + np.arange(-n_ls + 1, 1)[None, :]
    tt = (np.arange(n_ls) - (n_ls - 1)) * config.SCAN_PERIOD
    tc = tt - tt.mean()
    vel = (tc[None, :, None] * meas[ls_idx][..., :2]).sum(axis=1) / (tc ** 2).sum()  # (N, 2) МНК-наклон
    psi = np.arctan2(vel[:, 0], vel[:, 1])
    idx = ks[:, None] + np.arange(-window + 1, 1)[None, :]
    origin = meas[ks][:, None, :]
    x = np.concatenate([to_local(meas[idx] - origin, psi[:, None]),
                        polar[idx][..., 0:1] / 1000.0], axis=-1).astype(np.float32)
    out = {"x": x, "k": ks, "psi": psi}
    if with_target:
        h = np.arange(1, horizon + 1)
        disp = rt.truth[ks[:, None] + h[None, :], :2] - meas[ks][:, None, :2]
        out["y"] = to_local(disp, psi[:, None]).astype(np.float32)
    return out


def apply_e2e(rt, ks: np.ndarray, psi: np.ndarray, disp: np.ndarray,
              horizon: int = config.HORIZON_SCANS) -> np.ndarray:
    """
    Прогноз сквозной модели (K, H, 3): отметка k + смещение; высота - отметки k; вне ks -
    nan. При пропуске в k - от последней полученной отметки (held).
    """
    meas, _ = held(rt)
    out = np.full((len(rt.t), horizon, 3), np.nan)
    out[ks, :, :2] = meas[ks][:, None, :2] + to_global(disp.astype(float), psi[:, None])
    out[ks, :, 2] = meas[ks][:, None, 2]
    return out


def onset_labels(rt, ks: np.ndarray, ahead: int = config.ONSET_AHEAD,
                 behind: int = config.ONSET_BEHIND) -> np.ndarray:
    """
    Метка начала маневра в обзорах ks (этап 5): 1, если есть начало маневра o (по
    разметке истины) с o - ahead <= k <= o + behind. Это цель, а не признак: метка
    смотрит в будущее, во входы сети не подаётся.
    """
    ks = np.asarray(ks)
    out = np.zeros(len(ks), dtype=np.float32)
    for o in rt.onsets:
        out[(ks >= o - ahead) & (ks <= o + behind)] = 1.0
    return out


def turn_class(rt, ks: np.ndarray, horizon: int = config.HORIZON_SCANS,
               threshold_deg: float = config.MULTI_TURN_DEG) -> np.ndarray:
    """
    Класс будущего движения в обзорах ks (этап 14): 0 - прямо, 1 - влево, 2 - вправо. По истине:
    курс хорды (k-1 -> k) и хорды (k+H-1 -> k+H); изменение меньше threshold_deg - «прямо»,
    иначе по знаку (курс от севера по часовой: рост - вправо). Цель, а не признак - смотрит в
    будущее, во входы сети не подаётся.
    """
    ks = np.asarray(ks)
    p = rt.truth[:, :2]

    def course(k):
        d = p[k] - p[k - 1]
        return np.arctan2(d[:, 0], d[:, 1])

    dpsi = np.degrees(np.angle(np.exp(1j * (course(ks + horizon) - course(ks)))))
    out = np.where(dpsi > 0, 2, 1)
    out[np.abs(dpsi) < threshold_deg] = 0
    return out.astype(np.int64)


# ============================================================================
# Этап 6в: данные для других схем гибридизации (вариант Б)
# ============================================================================

def model_offsets(run, ks: np.ndarray, psi: np.ndarray, p_pow: np.ndarray) -> tuple:
    """
    Схема «сеть выбирает модель»: в обзорах ks - смещения прогнозов моделей IMM от
    прогноза IMM в системе курса d (N, H, M, 2) и логарифмы весов IMM
    log(mu_k P^h) (N, H, M). Прогноз IMM - сумма прогнозов моделей с этими весами,
    поэтому sum_j w_j d_j = 0: при весах IMM поправка нулевая. run - прогон
    IMM.run(with_models=True), p_pow - IMM.p_pow (H, M, M).
    """
    mf = run.model_forecast[ks]                                             # (N, M, H, 2)
    d = mf - run.forecast[ks][:, None, :, :2]
    d = to_local(d, psi[:, None, None]).transpose(0, 2, 1, 3)               # (N, H, M, 2)
    w = np.einsum("nm,hmj->nhj", run.mu[ks], p_pow)
    return d.astype(np.float32), np.log(np.maximum(w, 1e-12)).astype(np.float32)


def sequence_arrays(rt) -> dict:
    """
    Схемы с сетью внутри фильтра: вся трасса по горизонтали - отметки z (K, 2),
    ковариации отметок R (K, 2, 2), признак обнаружения (K,), истина (K, 2). Истина
    нужна только для функции потерь, во вход сети не идёт.
    """
    return {"z": rt.meas[:, :2], "r": rt.meas_cov[:, :2, :2], "det": rt.detected.astype(np.float32),
            "truth": rt.truth[:, :2]}


def sequence_chunks(seqs: list, length: int, horizon: int = config.HORIZON_SCANS) -> tuple:
    """
    Куски трасс для обучения: длина length + horizon (на последних horizon обзорах
    только истина для прогноза), шаг length. Координаты - от первой отметки куска.
    Возвращает x (N, L + H, 7) = [z (2), R (4), обнаружение] и y (N, L + H, 2) - истина.
    """
    xs, ys = [], []
    for s in seqs:
        k_total = len(s["z"])
        for start in range(0, k_total - length - horizon + 1, length):
            sl = slice(start, start + length + horizon)
            origin = s["z"][start]
            xs.append(np.concatenate([s["z"][sl] - origin, s["r"][sl].reshape(-1, 4), s["det"][sl, None]], axis=1))
            ys.append(s["truth"][sl] - origin)
    if not xs:
        return np.zeros((0, length + horizon, 7), np.float32), np.zeros((0, length + horizon, 2), np.float32)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)
