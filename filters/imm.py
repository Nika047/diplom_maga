"""
IMM (Interacting Multiple Model) на моделях filters/models.py и прогноз на 1..H обзоров.

Фильтр Калмана с одной моделью CV - это тот же класс с IMM_MODELS = ("CV",):
при одной модели смешивание вырождается, и IMM совпадает с обычным фильтром
(расширенным - для нелинейной CT).

Шаг обзора k:
  1. смешивание оценок моделей по вероятностям перехода (IMM_P_STAY);
  2. прогноз и обновление каждой модели (EKF, обновление в форме Джозефа);
  3. вероятности моделей по правдоподобию невязок (в логарифмах, без переполнения);
  4. объединённая оценка - взвешенная сумма.
Без отметки (пропуск) делается только прогноз.

Прогноз из обзора k на h обзоров: каждая модель экстраполирует свою оценку h раз,
веса - вероятности моделей, продвинутые матрицей перехода на h шагов.

Инициализация - по первым двум обнаруженным отметкам (скорость - разность).
Сохраняются для этапа 2 (входы корректора): вероятности моделей, невязка
объединённого прогноза, нормированный квадрат невязки каждой модели.

Проверяется на этапе 1: на развороте растёт вероятность CT, IMM на маневрах точнее
фильтра Калмана с одной CV.
"""

from dataclasses import dataclass

import numpy as np

import config
from filters.models import DIM, X, Y, Z, VX, VY, VZ, AX, AY, W, make_models
from radar.model import RadarTrack

_H = np.zeros((3, DIM))
_H[0, X] = _H[1, Y] = _H[2, Z] = 1.0


@dataclass
class FilterRun:
    est: np.ndarray        # (K, DIM) объединённая оценка после обновления (nan до инициализации)
    mu: np.ndarray         # (K, M) вероятности моделей
    innov: np.ndarray      # (K, 3) невязка объединённого прогноза на этот обзор
    nis: np.ndarray        # (K, M) нормированный квадрат невязки каждой модели
    forecast: np.ndarray   # (K, H, 3) прогноз положения на 1..H обзоров из обзора k


def transition_matrix(m: int, p_stay: float) -> np.ndarray:
    if m == 1:
        return np.ones((1, 1))
    p = np.full((m, m), (1.0 - p_stay) / (m - 1))
    np.fill_diagonal(p, p_stay)
    return p


class IMM:
    def __init__(self, model_names=config.IMM_MODELS, mu0=config.IMM_MU0, p_stay=config.IMM_P_STAY,
                 horizon: int = config.HORIZON_SCANS):
        self.models = make_models(model_names)
        self.m = len(self.models)
        self.mu0 = np.asarray(mu0 if len(mu0) == self.m else [1.0 / self.m] * self.m, dtype=float)
        self.mu0 = self.mu0 / self.mu0.sum()
        self.p = transition_matrix(self.m, p_stay)
        self.horizon = horizon
        self.p_pow = [np.linalg.matrix_power(self.p, h) for h in range(1, horizon + 1)]

    def _init_state(self, z0, z1, r0, r1, dt):
        x = np.zeros(DIM)
        x[[X, Y, Z]] = z1
        x[[VX, VY, VZ]] = (z1 - z0) / dt
        p = np.zeros((DIM, DIM))
        p[np.ix_([X, Y, Z], [X, Y, Z])] = r1
        p[np.ix_([VX, VY, VZ], [VX, VY, VZ])] = (r0 + r1) / dt ** 2
        p[np.ix_([X, Y, Z], [VX, VY, VZ])] = r1 / dt
        p[np.ix_([VX, VY, VZ], [X, Y, Z])] = r1 / dt
        p[AX, AX] = p[AY, AY] = config.INIT_SIGMA_A ** 2
        p[W, W] = config.INIT_SIGMA_W ** 2
        return x, p

    def _forecast(self, xs, mu, t):
        out = np.zeros((self.horizon, 3))
        paths = []
        for model, x in zip(self.models, xs):
            path = []
            for _ in range(self.horizon):
                x = model.f(x, t)
                path.append(x[[X, Y, Z]])
            paths.append(np.array(path))
        paths = np.stack(paths)  # (M, H, 3)
        for h in range(self.horizon):
            w = mu @ self.p_pow[h]
            out[h] = w @ paths[:, h]
        return out

    def run(self, rt: RadarTrack) -> FilterRun:
        k_total = len(rt.t)
        est = np.full((k_total, DIM), np.nan)
        mu_out = np.full((k_total, self.m), np.nan)
        innov = np.full((k_total, 3), np.nan)
        nis = np.full((k_total, self.m), np.nan)
        forecast = np.full((k_total, self.horizon, 3), np.nan)

        det = np.flatnonzero(rt.detected)
        if len(det) < 2:
            return FilterRun(est, mu_out, innov, nis, forecast)
        k0, k1 = det[0], det[1]
        x, p = self._init_state(rt.meas[k0], rt.meas[k1], rt.meas_cov[k0], rt.meas_cov[k1],
                                rt.t[k1] - rt.t[k0])
        xs = [x.copy() for _ in range(self.m)]
        ps = [p.copy() for _ in range(self.m)]
        mu = self.mu0.copy()
        t_prev = rt.t[k1]
        est[k1] = x
        mu_out[k1] = mu
        forecast[k1] = self._forecast(xs, mu, config.SCAN_PERIOD)

        for k in range(k1 + 1, k_total):
            dt = rt.t[k] - t_prev
            t_prev = rt.t[k]
            # 1. Смешивание
            c = mu @ self.p
            mix = (self.p * mu[:, None]) / np.maximum(c[None, :], 1e-300)  # mix[i, j] = P(i | j)
            x0, p0 = [], []
            for j in range(self.m):
                xj = sum(mix[i, j] * xs[i] for i in range(self.m))
                pj = sum(mix[i, j] * (ps[i] + np.outer(xs[i] - xj, xs[i] - xj)) for i in range(self.m))
                x0.append(xj)
                p0.append(pj)
            # 2. Прогноз и обновление моделей
            loglik = np.zeros(self.m)
            x_pred_comb = np.zeros(DIM)
            preds = []
            for j, model in enumerate(self.models):
                f = model.jac(x0[j], dt)
                xp = model.f(x0[j], dt)
                pp = f @ p0[j] @ f.T + model.q(dt)
                preds.append((xp, pp))
                x_pred_comb += c[j] * xp
            if rt.detected[k]:
                z, r = rt.meas[k], rt.meas_cov[k]
                innov[k] = z - x_pred_comb[[X, Y, Z]]
                for j, (xp, pp) in enumerate(preds):
                    nu = z - _H @ xp
                    s = _H @ pp @ _H.T + r
                    s_inv = np.linalg.inv(s)
                    gain = pp @ _H.T @ s_inv
                    xs[j] = xp + gain @ nu
                    ikh = np.eye(DIM) - gain @ _H
                    ps[j] = ikh @ pp @ ikh.T + gain @ r @ gain.T
                    d2 = float(nu @ s_inv @ nu)
                    nis[k, j] = d2
                    _, logdet = np.linalg.slogdet(s)
                    loglik[j] = -0.5 * (d2 + logdet + 3 * np.log(2 * np.pi))
                # 3. Вероятности моделей
                logw = loglik + np.log(np.maximum(c, 1e-300))
                logw -= logw.max()
                mu = np.exp(logw)
                mu /= mu.sum()
            else:
                for j, (xp, pp) in enumerate(preds):
                    xs[j], ps[j] = xp, pp
                mu = c
            # 4. Объединённая оценка и прогноз
            est[k] = sum(mu[j] * xs[j] for j in range(self.m))
            mu_out[k] = mu
            forecast[k] = self._forecast(xs, mu, config.SCAN_PERIOD)
        return FilterRun(est, mu_out, innov, nis, forecast)
