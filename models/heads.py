"""
Корректор любой архитектуры с дополнительными головами для гипотез этапов 6-7 (версия 2).

HeadedCorrector берёт признак последнего обзора окна у обычного корректора (GRU, LSTM, Transformer, Mamba,
стопка): перехватывается вход его выходного слоя `head`. Поверх признака - свои головы:
  main   - n_hyp гипотез поправки к прогнозу IMM на 1..H обзоров (n_hyp = 1 - обычный корректор; 0 - без поправки,
           для распознавателя). Нулевая инициализация: до обучения гипотезы = IMM;
  prob   - логиты вероятностей гипотез (n_hyp > 1);
  onset  - логит «маневр начнётся в горизонте прогноза или недавно начался» (гипотеза «начало маневра»);
  turn   - логит «в горизонте прогноза будет поворот» (распознаватель модульной схемы).
Выход (N, n_hyp * 2H [+ n_hyp] [+ 1] [+ 1]) в этом порядке; родной выходной слой backbone не используется.

Функция потерь spec_loss работает с упакованной целью (N, 2H + 4) = [поправка (2H), вес, класс, начало, поворот].
"""

import torch
from torch import nn
from torch.nn import functional as F

import config
from models import build_corrector

TARGET_EXTRA = 4         # столбцы цели после поправки: вес, класс, начало маневра, поворот


class HeadedCorrector(nn.Module):
    def __init__(self, corrector: str, n_features: int, horizon: int, window: int, hp: dict, n_hyp: int = 1,
                 onset: bool = False, turn: bool = False, zero_init: bool = True):
        super().__init__()
        self.horizon, self.n_hyp, self.has_onset, self.has_turn = horizon, n_hyp, onset, turn
        self.backbone = build_corrector(corrector, n_features, horizon, window, **hp)
        dim = self.backbone.head.in_features
        self._feat = None
        self.backbone.head.register_forward_hook(lambda m, inp, out: setattr(self, "_feat", inp[0]))
        self.main = nn.Linear(dim, n_hyp * horizon * 2) if n_hyp > 0 else None
        if self.main is not None and zero_init:
            nn.init.zeros_(self.main.weight)
            nn.init.zeros_(self.main.bias)
        self.prob = nn.Linear(dim, n_hyp) if n_hyp > 1 else None
        self.onset = nn.Linear(dim, 1) if onset else None
        self.turn = nn.Linear(dim, 1) if turn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone(x)
        f = self._feat
        outs = [m(f) for m in (self.main, self.prob, self.onset, self.turn) if m is not None]
        return torch.cat(outs, dim=1)


def parts(pred: torch.Tensor, n_hyp: int, horizon: int, onset: bool, turn: bool) -> dict:
    """Выход сети -> {"hyps" (N, K, 2H), "logits" (N, K), "onset" (N,), "turn" (N,)} (отсутствующих ключей нет)."""
    out, col, a = {}, 0, 2 * horizon
    if n_hyp > 0:
        out["hyps"] = pred[:, :n_hyp * a].reshape(len(pred), n_hyp, a)
        col = n_hyp * a
    if n_hyp > 1:
        out["logits"] = pred[:, col:col + n_hyp]
        col += n_hyp
    if onset:
        out["onset"] = pred[:, col]
        col += 1
    if turn:
        out["turn"] = pred[:, col]
    return out


def spec_loss(horizon: int, n_hyp: int, semantic: bool, weight: bool, onset: bool, turn: bool,
              ce_weight: float = config.MULTI_CE_WEIGHT, onset_weight: float = config.ONSET_LOSS_WEIGHT,
              eps: float = config.MULTI_WTA_EPS):
    """
    Потери набора голов. Цель - (N, 2H + 4) = [поправка, вес, класс будущего движения, начало маневра, поворот].
    Регрессия: MSE (с весом примера, если weight); при n_hyp > 1 - semantic: учится гипотеза класса (как этап 14
    версии 1), иначе winner-takes-all; плюс перекрёстная энтропия вероятностей. Начало маневра и поворот -
    BCE с весом onset_weight.
    """
    a = 2 * horizon

    def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = parts(pred, n_hyp, horizon, onset, turn)
        total = pred.new_zeros(())
        if n_hyp > 0:
            mse = ((p["hyps"] - y[:, None, :a]) ** 2).mean(dim=2)                 # (N, K)
            if n_hyp == 1:
                per = mse[:, 0]
            elif semantic:
                cls = y[:, a + 1].long()
                per = mse.gather(1, cls[:, None])[:, 0]
                total = total + ce_weight * F.cross_entropy(p["logits"], cls)
            else:
                best = mse.detach().argmin(dim=1)
                per = (1 - eps) * mse.gather(1, best[:, None])[:, 0] + eps * mse.mean(dim=1)
                total = total + ce_weight * F.cross_entropy(p["logits"], best)
            if weight:
                w = y[:, a]
                total = total + (w * per).sum() / w.sum()
            else:
                total = total + per.mean()
        if onset:
            total = total + onset_weight * F.binary_cross_entropy_with_logits(p["onset"], y[:, a + 2])
        if turn:
            total = total + onset_weight * F.binary_cross_entropy_with_logits(p["turn"], y[:, a + 3])
        return total
    return loss
