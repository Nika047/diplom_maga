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

Ковариация прогноза (этап 6б, строб) - по запросу run(rt, with_cov=True): каждая
модель продвигает свою ковариацию h раз (F P F^T + Q), смесь - с теми же весами,
что прогноз, плюс разброс прогнозов моделей. Без запроса прогон не меняется.

Прогнозы отдельных моделей (этап 6в, сеть пересчитывает веса моделей) - по запросу
run(rt, with_models=True): те же пути, из которых складывается прогноз IMM.

Встраивание сети (этап 6г), без него прогон не меняется:
  - run(rt, mode_logit=...) - добавка к логарифмам вероятностей моделей в каждом обзоре
    (схема Б4: сеть управляет переключением моделей);
  - extra_models - дополнительные режимы (схема Б5: прогноз сети как режим IMM). У такой
    модели может быть set_scan(k) - номер текущего обзора - и forecast_path(x) - свой
    прогноз на 1..H обзоров вместо повторения f.

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
    forecast_cov: np.ndarray = None  # (K, H, 2, 2) ковариация прогноза по горизонтали (with_cov)
    model_forecast: np.ndarray = None  # (K, M, H, 2) прогноз каждой модели по горизонтали (with_models)


def transition_matrix(m: int, p_stay: float) -> np.ndarray:
    if m == 1:
        return np.ones((1, 1))
    p = np.full((m, m), (1.0 - p_stay) / (m - 1))
    np.fill_diagonal(p, p_stay)
    return p


class IMM:
    def __init__(self, model_names=None, mu0=None, p_stay=None, horizon: int = None, extra_models=None):
        # Значения по умолчанию читаются из config в момент создания, а не при импорте:
        # этап 4 подменяет параметры IMM найденными генетическим алгоритмом.
        model_names = config.IMM_MODELS if model_names is None else model_names
        mu0 = config.IMM_MU0 if mu0 is None else mu0
        p_stay = config.IMM_P_STAY if p_stay is None else p_stay
        horizon = config.HORIZON_SCANS if horizon is None else horizon
        self.models = make_models(model_names) + list(extra_models or [])
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

    def _model_paths(self, xs, t) -> np.ndarray:
        """Прогноз положения каждой модели на 1..H обзоров (M, H, 3)."""
        paths = []
        for model, x in zip(self.models, xs):
            if hasattr(model, "forecast_path"):
                paths.append(model.forecast_path(x, t))
                continue
            path = []
            for _ in range(self.horizon):
                x = model.f(x, t)
                path.append(x[[X, Y, Z]])
            paths.append(np.array(path))
        return np.stack(paths)

    def _forecast(self, xs, mu, t, paths=None):
        out = np.zeros((self.horizon, 3))
        paths = self._model_paths(xs, t) if paths is None else paths  # (M, H, 3)
        for h in range(self.horizon):
            w = mu @ self.p_pow[h]
            out[h] = w @ paths[:, h]
        return out

    def _forecast_cov(self, xs, ps, mu, t):
        """Ковариация прогноза по горизонтали (H, 2, 2): смесь моделей с весами mu @ P^h."""
        out = np.zeros((self.horizon, 2, 2))
        means, covs = [], []
        for model, x, p in zip(self.models, xs, ps):
            m_path, c_path = [], []
            for _ in range(self.horizon):
                f = model.jac(x, t)
                x = model.f(x, t)
                p = f @ p @ f.T + model.q(t)
                m_path.append(x[[X, Y]])
                c_path.append(p[np.ix_([X, Y], [X, Y])])
            means.append(m_path)
            covs.append(c_path)
        means, covs = np.array(means), np.array(covs)   # (M, H, 2), (M, H, 2, 2)
        for h in range(self.horizon):
            w = mu @ self.p_pow[h]
            mean = w @ means[:, h]
            d = means[:, h] - mean
            out[h] = np.einsum("m,mij->ij", w, covs[:, h] + d[:, :, None] * d[:, None, :])
        return out

    def _set_scan(self, k: int) -> None:
        for model in self.models:
            if hasattr(model, "set_scan"):
                model.set_scan(k)

    def run(self, rt: RadarTrack, with_cov: bool = False, with_models: bool = False,
            mode_logit: np.ndarray = None) -> FilterRun:
        """mode_logit - (K, M) добавка к логарифмам вероятностей моделей после обновления (nan - нет)."""
        k_total = len(rt.t)
        est = np.full((k_total, DIM), np.nan)
        mu_out = np.full((k_total, self.m), np.nan)
        innov = np.full((k_total, 3), np.nan)
        nis = np.full((k_total, self.m), np.nan)
        forecast = np.full((k_total, self.horizon, 3), np.nan)
        fcov = np.full((k_total, self.horizon, 2, 2), np.nan) if with_cov else None
        mfc = np.full((k_total, self.m, self.horizon, 2), np.nan) if with_models else None

        def forecast_at(k, xs, ps, mu):
            paths = self._model_paths(xs, config.SCAN_PERIOD)
            forecast[k] = self._forecast(xs, mu, config.SCAN_PERIOD, paths)
            if with_cov:
                fcov[k] = self._forecast_cov(xs, ps, mu, config.SCAN_PERIOD)
            if with_models:
                mfc[k] = paths[..., :2]

        det = np.flatnonzero(rt.detected)
        if len(det) < 2:
            return FilterRun(est, mu_out, innov, nis, forecast, fcov, mfc)
        k0, k1 = det[0], det[1]
        x, p = self._init_state(rt.meas[k0], rt.meas[k1], rt.meas_cov[k0], rt.meas_cov[k1],
                                rt.t[k1] - rt.t[k0])
        xs = [x.copy() for _ in range(self.m)]
        ps = [p.copy() for _ in range(self.m)]
        mu = self.mu0.copy()
        t_prev = rt.t[k1]
        est[k1] = x
        mu_out[k1] = mu
        self._set_scan(k1)
        forecast_at(k1, xs, ps, mu)

        for k in range(k1 + 1, k_total):
            dt = rt.t[k] - t_prev
            t_prev = rt.t[k]
            self._set_scan(k)
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
            if mode_logit is not None and not np.isnan(mode_logit[k]).any() and np.ptp(mode_logit[k]) > 0:
                mu = mu * np.exp(mode_logit[k] - mode_logit[k].max())
                mu /= mu.sum()
            # 4. Объединённая оценка и прогноз
            est[k] = sum(mu[j] * xs[j] for j in range(self.m))
            mu_out[k] = mu
            forecast_at(k, xs, ps, mu)
        return FilterRun(est, mu_out, innov, nis, forecast, fcov, mfc)
