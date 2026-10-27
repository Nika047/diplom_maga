"""
Схемы гибридизации варианта Б (этап 6в). Схема A - поправка после IMM - это корректоры
models/gru.py и др.

ModelWeightGRU - «сеть выбирает модель». GRU по окну признаков гибрида (то же окно, что у
корректора) выдаёт поправку к логарифмам весов моделей IMM на каждом горизонте; прогноз -
сумма прогнозов моделей с новыми весами. Голова нулевая: до обучения веса = веса IMM
(mu_k P^h), прогноз = прогноз IMM. Поправка к IMM ограничена выпуклой оболочкой
прогнозов моделей: сеть не может «увести» прогноз туда, куда не ведёт ни одна модель.

NeuralKF - фильтр Калмана с моделью CV по горизонтали на PyTorch (дифференцируемый) с
сетью внутри:
  mode="gain"  - по типу KalmanNet: GRU добавляет поправку к коэффициенту усиления
                 (ограниченную tanh: по положению до 1, по скорости до 1 / T);
  mode="noise" - GRU задаёт множитель СКО шума ускорения на каждом обзоре.
Голова нулевая: до обучения - обычный фильтр Калмана с СКО шума sigma0. Прогноз на h
обзоров - продолжение оценки с постоянной скоростью. Признаки GRU в обзоре k - только
из отметок до k и состояния фильтра (причинно), в системе курса фильтра. Исходный
KalmanNet - с одной линейной моделью; дифференцируемый IMM с нелинейной CT - вне этой
работы.

Этап 6г:
  ModeGRU - классификатор маневра по окну признаков «тени» IMM (схема Б4: его выход
            управляет вероятностями моделей рабочего IMM, filters/imm.py mode_logit);
  WeightResidualGRU - схема A + Б1: веса моделей IMM и поправка к прогнозу одной сетью.

Проверяется на этапе 6в: нулевая поправка до обучения (ModelWeightGRU = IMM), фильтр
без сети близок к фильтру Калмана этапа 1, loss на одном батче падает. На этапе 6г -
A + Б1 до обучения = IMM, Б4 при нулевой добавке = IMM.
"""

import torch
from torch import nn

import config

N_SEQ_FEATURES = 6


class ModelWeightGRU(nn.Module):
    def __init__(self, n_features: int, horizon: int, n_models: int, hidden: int = config.CORRECTOR_HIDDEN,
                 layers: int = config.CORRECTOR_LAYERS, dropout: float = config.CORRECTOR_DROPOUT):
        super().__init__()
        self.horizon, self.n_models = horizon, n_models
        self.rnn = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * n_models)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])                        # (N, H * M) поправки к логарифмам весов


def model_weight_correction(logits: torch.Tensor, d: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    """Поправка к IMM (N, H, 2): сумма смещений моделей d (N, H, M, 2) с весами softmax(logw + logits)."""
    w = torch.softmax(logw + logits.view(logw.shape), dim=-1)
    return (w[..., None] * d).sum(dim=2)


def model_weight_loss(horizon: int, n_models: int):
    """
    MSE поправки. Цель упакована в одну строку (N, 2H + 2HM + HM): нормированная поправка
    y, нормированные смещения моделей d, логарифмы весов IMM - так подходит общий
    training/trainer.train.
    """
    def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n, a, b = len(y), 2 * horizon, 2 * horizon + 2 * horizon * n_models
        corr = model_weight_correction(pred, y[:, a:b].view(n, horizon, n_models, 2),
                                       y[:, b:].view(n, horizon, n_models))
        return ((corr - y[:, :a].view(n, horizon, 2)) ** 2).mean()
    return loss


def _rot(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Матрица перехода в систему курса (B, 2, 2): local = R @ global, как data/windows.to_local."""
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


class NeuralKF(nn.Module):
    def __init__(self, mode: str, sigma0: float, hidden: int = 64, horizon: int = config.HORIZON_SCANS,
                 period: float = config.SCAN_PERIOD, doppler: bool = False, sigma_vr: float = 0.5):
        super().__init__()
        if mode not in ("gain", "noise"):
            raise ValueError(mode)
        self.mode, self.sigma0, self.horizon, self.period = mode, sigma0, horizon, period
        self.doppler, self.sigma_vr = doppler, sigma_vr
        self.cell = nn.GRUCell(N_SEQ_FEATURES, hidden)
        self.head = nn.Linear(hidden, 8 if mode == "gain" else 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        t = period
        f = torch.eye(4)
        f[0, 2] = f[1, 3] = t
        q1 = torch.zeros(4, 4)                               # дискретный белый шум ускорения, СКО 1
        for p, v in ((0, 2), (1, 3)):
            q1[p, p], q1[p, v], q1[v, p], q1[v, v] = t ** 4 / 4, t ** 3 / 2, t ** 3 / 2, t ** 2
        self.register_buffer("f", f)
        self.register_buffer("q1", q1)
        self.register_buffer("steps", torch.arange(1, horizon + 1, dtype=torch.float32) * t)

    def _forecast(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, None, :2] + self.steps[None, :, None] * x[:, None, 2:]          # (B, H, 2)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq (B, L, 7) = [z (2), R (4), обнаружение]; первые две отметки обнаружены. Выход (B, L, H, 2).
        С доплером (версия 2) seq (B, L, 11): ещё [vr, cos(угла места) * ux, cos(...) * uy, годность] -
        после обновления по положению скалярное обновление по радиальной скорости с H = [0, 0, hx, hy]
        (проекция горизонтальной скорости на луч визирования; СКО sigma_vr).
        """
        b, length, _ = seq.shape
        z, r, det = seq[..., :2], seq[..., 2:6].reshape(b, length, 2, 2), seq[..., 6]
        t = self.period
        dtype = seq.dtype
        x = torch.cat([z[:, 1], (z[:, 1] - z[:, 0]) / t], dim=1)
        p = torch.zeros(b, 4, 4, dtype=dtype, device=seq.device)
        p[:, :2, :2] = r[:, 1]
        p[:, 2:, 2:] = (r[:, 0] + r[:, 1]) / t ** 2
        p[:, :2, 2:] = p[:, 2:, :2] = r[:, 1] / t
        out = [torch.full((b, self.horizon, 2), float("nan"), dtype=dtype, device=seq.device), self._forecast(x)]
        h = torch.zeros(b, self.cell.hidden_size, dtype=dtype, device=seq.device)
        z_prev = z[:, 1]
        eye = torch.eye(4, dtype=dtype, device=seq.device)
        for k in range(2, length):
            xp = x @ self.f.T
            pp = self.f @ p @ self.f.T
            nu = z[:, k] - xp[:, :2]
            speed = torch.linalg.norm(xp[:, 2:], dim=1).clamp_min(1e-3)
            c, s = xp[:, 3] / speed, xp[:, 2] / speed                    # курс от севера по часовой
            rot = _rot(c, s)
            loc = lambda v: (rot @ v[..., None])[..., 0]                  # noqa: E731
            spread = torch.log1p((pp[:, 0, 0] + pp[:, 1, 1] + r[:, k, 0, 0] + r[:, k, 1, 1]) / 1e3)
            feat = torch.cat([loc(nu) / 50.0, loc(z[:, k] - z_prev) / 50.0, (speed / 30.0)[:, None],
                              spread[:, None]], dim=1) * det[:, k, None]
            h = self.cell(feat, h)
            o = self.head(h)
            sigma = self.sigma0 * torch.exp(o[:, 0].clamp(-3.0, 3.0)) if self.mode == "noise" else \
                torch.full((b,), self.sigma0, dtype=dtype, device=seq.device)
            pp = pp + sigma[:, None, None] ** 2 * self.q1
            s_mat = pp[:, :2, :2] + r[:, k]
            # inv_ex: при расходимости фильтра - nan в прогнозе (его ловит проверка этапа), а не исключение
            gain = pp[:, :, :2] @ torch.linalg.inv_ex(s_mat).inverse      # (B, 4, 2)
            if self.mode == "gain":
                # Поправка ограничена (по положению до 1, по скорости до 1 / T): без ограничения
                # усиление уходило туда, где фильтр расходится (смоук этапа 6в)
                rt_ = rot.transpose(1, 2)
                d_pos = rt_ @ torch.tanh(o[:, :4]).view(b, 2, 2) @ rot
                d_vel = rt_ @ (torch.tanh(o[:, 4:]) / t).view(b, 2, 2) @ rot
                gain = gain + torch.cat([d_pos, d_vel], dim=1)
            gain = gain * det[:, k, None, None]                           # пропуск - без обновления
            x = xp + (gain @ nu[..., None])[..., 0]
            ikh = eye - torch.cat([gain, torch.zeros(b, 4, 2, dtype=dtype, device=seq.device)], dim=2)  # I - K H
            p = ikh @ pp @ ikh.transpose(1, 2) + gain @ r[:, k] @ gain.transpose(1, 2)
            if self.doppler:
                hv = torch.cat([torch.zeros(b, 2, dtype=dtype, device=seq.device), seq[:, k, 8:10]], dim=1)   # (B, 4)
                ok = seq[:, k, 10] * det[:, k]
                s_v = (hv * (p @ hv[..., None])[..., 0]).sum(dim=1) + self.sigma_vr ** 2          # (B,)
                kv = (p @ hv[..., None])[..., 0] / s_v[:, None] * ok[:, None]                      # (B, 4)
                x = x + kv * (seq[:, k, 7] - (hv * x).sum(dim=1))[:, None]
                ikv = eye - kv[..., None] * hv[:, None, :]
                p = ikv @ p @ ikv.transpose(1, 2) + (kv[..., None] * kv[:, None, :]) * self.sigma_vr ** 2
            out.append(self._forecast(x))
            z_prev = torch.where(det[:, k, None] > 0, z[:, k], z_prev)
        return torch.stack(out, dim=1)


def sequence_loss(burn: int, scale: float, horizon: int = config.HORIZON_SCANS):
    """MSE прогноза из обзоров k >= burn на 1..H обзоров по истине y (B, L, 2); единица - scale м."""
    def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        length = y.shape[1]
        ks = torch.arange(burn, length - horizon, device=y.device)
        target = torch.stack([y[:, ks + h] for h in range(1, horizon + 1)], dim=2)   # (B, K, H, 2)
        return (((pred[:, ks] - target) / scale) ** 2).mean()
    return loss


# ============================================================================
# Этап 6г
# ============================================================================

class ModeGRU(nn.Module):
    """
    Схема Б4: GRU по окну признаков гибрида оценивает, идёт ли маневр (логит P(разворот)).
    Голова нулевая: до обучения P = 0,5, добавка к вероятностям моделей IMM одинакова для
    всех моделей и ничего не меняет.
    """

    def __init__(self, n_features: int, hidden: int = config.CORRECTOR_HIDDEN, layers: int = config.CORRECTOR_LAYERS,
                 dropout: float = config.CORRECTOR_DROPOUT):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def bce_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(pred, y)


class WeightResidualGRU(nn.Module):
    """
    Схема A + Б1: одна GRU с двумя головами - поправка к весам моделей IMM (как Б1) и
    поправка к получившемуся прогнозу (как A). Обе головы нулевые: до обучения = IMM.
    Выход (N, H * M + 2H): логиты весов, затем поправка.
    """

    def __init__(self, n_features: int, horizon: int, n_models: int, hidden: int = config.CORRECTOR_HIDDEN,
                 layers: int = config.CORRECTOR_LAYERS, dropout: float = config.CORRECTOR_DROPOUT):
        super().__init__()
        self.horizon, self.n_models = horizon, n_models
        self.rnn = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * n_models + horizon * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def weight_residual_loss(horizon: int, n_models: int):
    """MSE суммарной поправки; цель упакована как у model_weight_loss: [y (2H), d (2HM), log w (HM)]."""
    def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n, a, b = len(y), 2 * horizon, 2 * horizon + 2 * horizon * n_models
        hm = horizon * n_models
        corr = model_weight_correction(pred[:, :hm], y[:, a:b].view(n, horizon, n_models, 2),
                                       y[:, b:].view(n, horizon, n_models))
        corr = corr + pred[:, hm:].view(n, horizon, 2)
        return ((corr - y[:, :a].view(n, horizon, 2)) ** 2).mean()
    return loss
