"""
Режим IMM «прогноз нейросети» (этап 6г, схема Б5; по идее PR-IMM).

Сеть (корректор A) работает на «тени» - обычном IMM по тем же отметкам - и для каждого
обзора k даёт прогноз положения на 1..H обзоров. В рабочем IMM это ещё один режим рядом с
CV и CT:
  - предсказание в обзор k: положение по горизонтали - прогноз сети из обзора k - 1 на 1
    обзор, ковариация положения - ошибка этого прогноза на val (pos_cov); скорость и высота -
    по модели CV из смешанного состояния;
  - прогноз из обзора k на 1..H обзоров - прогноз сети из обзора k;
  - пока прогноза сети нет (первые обзоры трассы), режим ведёт себя как CV.
Вероятность режима IMM считает как обычно - по невязке: если сеть предсказала отметку
точнее моделей движения, её вес растёт. Тень, а не рабочий IMM, нужна, чтобы входы сети не
зависели от её же выхода.

Проверяется на этапе 6г: при прогнозе сети, равном прогнозу CV, режим совпадает с CV.
"""

import numpy as np

import config
from filters.models import CV, VX, VY, X, Y, Z


class NeuralMode(CV):
    name = "NN"

    def __init__(self, forecast: np.ndarray, pos_cov: np.ndarray):
        """forecast - (K, H, 3) прогноз сети по обзорам (nan - нет); pos_cov - (2, 2) ошибка прогноза на 1 обзор."""
        self.forecast, self.pos_cov, self.k = forecast, pos_cov, 0

    def set_scan(self, k: int) -> None:
        self.k = k

    def _prev(self):
        if self.k < 1:
            return None
        p = self.forecast[self.k - 1, 0, :2]
        return None if np.isnan(p).any() else p

    @staticmethod
    def _cv(x, t):
        return CV.jac(None, x, t) @ x     # CV.f вызывает self.jac, а здесь он переопределён

    def f(self, x, t):
        out = self._cv(x, t)
        p = self._prev()
        if p is not None:
            out[[X, Y]] = p
        return out

    def jac(self, x, t):
        f = CV.jac(self, x, t)
        if self._prev() is not None:
            f[[X, Y], :] = 0.0       # положение берётся у сети, а не из смешанного состояния
        return f

    def q(self, t):
        q = super().q(t)
        if self._prev() is not None:
            q[np.ix_([X, Y], [X, Y])] = self.pos_cov
            q[np.ix_([X, Y], [VX, VY])] = 0.0
            q[np.ix_([VX, VY], [X, Y])] = 0.0
        return q

    def forecast_path(self, x, t) -> np.ndarray:
        path = []
        for _ in range(config.HORIZON_SCANS):
            x = self._cv(x, t)
            path.append(x[[X, Y, Z]])
        path = np.array(path)
        nn = self.forecast[self.k, :, :2]
        if not np.isnan(nn).any():
            path[:, :2] = nn
        return path
