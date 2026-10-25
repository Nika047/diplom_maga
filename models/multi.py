"""
Корректор с несколькими вариантами прогноза (этап 14): GRU как в models/gru.py и две головы.

  - K гипотез поправки к прогнозу IMM на 1..H обзоров (config.MULTI_HYPOTHESES);
  - логиты вероятностей гипотез.
Выход - (N, K * 2H + K): гипотезы по строкам (гипотеза за гипотезой, как y.reshape(N, 2H)),
затем логиты. Цель для функций потерь - (N, 2H + 1): нормированная поправка и номер
класса / гипотезы в последнем столбце.

Два способа обучения:
  semantic_loss - гипотезы «прямо / влево / вправо»: учится гипотеза класса будущего движения
                  истины (data/windows.turn_class), вероятности - перекрёстная энтропия к классу;
  wta_loss      - «свободные» гипотезы (winner takes all): учится ближайшая к истине гипотеза
                  (остальные - с долей MULTI_WTA_EPS, чтобы не застывали), вероятности - к ней.
У семантических гипотез голова нулевая (до обучения все = IMM); у свободных - обычная
случайная инициализация: одинаковые гипотезы WTA не развела бы.

Проверяется на этапе 14: формы, нулевые семантические гипотезы до обучения, градиенты;
семантический loss учит только гипотезу класса; overfit one batch.
"""

import torch
from torch import nn
from torch.nn import functional as F

import config


class MultiHypGRU(nn.Module):
    def __init__(self, n_features: int, horizon: int, n_hyp: int = config.MULTI_HYPOTHESES,
                 hidden: int = config.CORRECTOR_HIDDEN, layers: int = config.CORRECTOR_LAYERS,
                 dropout: float = config.CORRECTOR_DROPOUT, zero_init: bool = True):
        super().__init__()
        self.horizon, self.n_hyp = horizon, n_hyp
        self.rnn = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_hyp * horizon * 2)
        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.prob_head = nn.Linear(hidden, n_hyp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1]
        return torch.cat([self.head(last), self.prob_head(last)], dim=1)


def split(pred: torch.Tensor, n_hyp: int) -> tuple:
    """Выход сети -> (гипотезы (N, K, 2H), логиты (N, K))."""
    return pred[:, :-n_hyp].reshape(len(pred), n_hyp, -1), pred[:, -n_hyp:]


def _hyp_mse(pred: torch.Tensor, y: torch.Tensor, n_hyp: int) -> tuple:
    hyps, logits = split(pred, n_hyp)
    mse = ((hyps - y[:, None, :-1]) ** 2).mean(dim=2)            # (N, K)
    return mse, logits


def semantic_loss(n_hyp: int = config.MULTI_HYPOTHESES, ce_weight: float = config.MULTI_CE_WEIGHT):
    def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse, logits = _hyp_mse(pred, y, n_hyp)
        cls = y[:, -1].long()
        return mse.gather(1, cls[:, None]).mean() + ce_weight * F.cross_entropy(logits, cls)
    return loss


def wta_loss(n_hyp: int = config.MULTI_HYPOTHESES, ce_weight: float = config.MULTI_CE_WEIGHT,
             eps: float = config.MULTI_WTA_EPS):
    def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse, logits = _hyp_mse(pred, y, n_hyp)
        best = mse.detach().argmin(dim=1)
        main = mse.gather(1, best[:, None]).mean()
        return (1 - eps) * main + eps * mse.mean() + ce_weight * F.cross_entropy(logits, best)
    return loss
