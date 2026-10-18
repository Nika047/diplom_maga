"""
Модели движения для фильтров на общем векторе состояния

    x = [x, y, z, vx, vy, vz, ax, ay, w]

(ENU, м, м/с, м/с^2; w - скорость разворота, рад/с, положительная - против часовой
стрелки, как в математической системе x - восток, y - север). Общий вектор нужен
IMM: оценки разных моделей смешиваются покомпонентно. Компоненты, которых в модели
нет, она обнуляет (строка перехода нулевая, малый шум оставляет ковариацию
положительно определённой).

  CV - постоянная скорость;
  CA - постоянное горизонтальное ускорение;
  CT - координированный разворот с постоянной w (нелинейная, якобиан численный).
По вертикали все модели - постоянная скорость с шумом FILTER_SIGMA_A_Z.

Проверяется на этапе 1: CT на окружности даёт точный прогноз, CV и CA на прямой -
тоже (известный ответ).
"""

import numpy as np

import config

X, Y, Z, VX, VY, VZ, AX, AY, W = range(9)
DIM = 9
_EPS_Q = np.array([0, 0, 0, 0, 0, 0, 1e-4, 1e-4, 1e-6])


def _dwna(t: float, sigma: float) -> np.ndarray:
    """Дискретный белый шум ускорения: ковариация (позиция, скорость)."""
    return sigma ** 2 * np.array([[t ** 4 / 4, t ** 3 / 2], [t ** 3 / 2, t ** 2]])


def _dwja(t: float, sigma: float) -> np.ndarray:
    """Дискретный белый шум рывка: ковариация (позиция, скорость, ускорение)."""
    g = np.array([t ** 3 / 6, t ** 2 / 2, t])
    return sigma ** 2 * np.outer(g, g)


class MotionModel:
    name = ""

    def f(self, x: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def jac(self, x: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def q(self, t: float) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _q_vertical(q: np.ndarray, t: float) -> None:
        q[np.ix_([Z, VZ], [Z, VZ])] = _dwna(t, config.FILTER_SIGMA_A_Z)


class CV(MotionModel):
    name = "CV"

    def jac(self, x, t):
        f = np.eye(DIM)
        f[X, VX] = f[Y, VY] = f[Z, VZ] = t
        f[AX, AX] = f[AY, AY] = f[W, W] = 0.0
        return f

    def f(self, x, t):
        return self.jac(x, t) @ x

    def q(self, t):
        q = np.diag(_EPS_Q)
        for p, v in ((X, VX), (Y, VY)):
            q[np.ix_([p, v], [p, v])] = _dwna(t, config.FILTER_SIGMA_A_CV)
        self._q_vertical(q, t)
        return q


class CA(MotionModel):
    name = "CA"

    def jac(self, x, t):
        f = np.eye(DIM)
        f[X, VX] = f[Y, VY] = f[Z, VZ] = t
        f[X, AX] = f[Y, AY] = t ** 2 / 2
        f[VX, AX] = f[VY, AY] = t
        f[W, W] = 0.0
        return f

    def f(self, x, t):
        return self.jac(x, t) @ x

    def q(self, t):
        q = np.diag(_EPS_Q)
        for p, v, a in ((X, VX, AX), (Y, VY, AY)):
            q[np.ix_([p, v, a], [p, v, a])] = _dwja(t, config.FILTER_SIGMA_J_CA)
        self._q_vertical(q, t)
        return q


class CT(MotionModel):
    name = "CT"

    def f(self, x, t):
        out = x.copy()
        w = x[W]
        th = w * t
        if abs(w) < 1e-6:
            s_w, c_w = t, 0.5 * w * t * t  # пределы sin(th)/w и (1 - cos(th))/w
        else:
            s_w, c_w = np.sin(th) / w, (1.0 - np.cos(th)) / w
        vx, vy = x[VX], x[VY]
        out[X] = x[X] + s_w * vx - c_w * vy
        out[Y] = x[Y] + c_w * vx + s_w * vy
        out[VX] = np.cos(th) * vx - np.sin(th) * vy
        out[VY] = np.sin(th) * vx + np.cos(th) * vy
        out[Z] = x[Z] + t * x[VZ]
        out[AX] = out[AY] = 0.0
        return out

    def jac(self, x, t):
        f = np.zeros((DIM, DIM))
        for j in range(DIM):
            h = 1e-6 * max(1.0, abs(x[j]))
            dx = np.zeros(DIM)
            dx[j] = h
            f[:, j] = (self.f(x + dx, t) - self.f(x - dx, t)) / (2 * h)
        return f

    def q(self, t):
        q = np.diag(_EPS_Q)
        for p, v in ((X, VX), (Y, VY)):
            q[np.ix_([p, v], [p, v])] = _dwna(t, config.FILTER_SIGMA_A_CT)
        q[W, W] = config.FILTER_SIGMA_W_CT ** 2 * t
        self._q_vertical(q, t)
        return q


MODELS = {"CV": CV, "CA": CA, "CT": CT}


def make_models(names) -> list:
    return [MODELS[n]() for n in names]
