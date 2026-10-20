"""
Корректор с прогнозом начала маневра (этап 5): GRU как в models/gru.py и две головы.

  - поправка к прогнозу IMM на 1..H обзоров, нулевая инициализация: до обучения = IMM;
  - логит вероятности начала маневра (метка - config.ONSET_AHEAD, ONSET_BEHIND).
Выход - (N, 2H + 1): поправка по строкам (как y.reshape(N, 2H)) и логит. Обучение
совместное, функция потерь - multitask_loss.

Слои создаются в том же порядке, что у GRUCorrector, а голова начала маневра -
последней: при том же seed GRU и голова поправки инициализируются одинаково, и
отличие от однозадачного корректора - только в функции потерь.

Проверяется на этапе 5: смоук forward / backward, нулевая поправка до обучения,
overfit one batch.
"""

import torch
from torch import nn
from torch.nn import functional as F

import config


class OnsetGRU(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = config.CORRECTOR_HIDDEN,
                 layers: int = config.CORRECTOR_LAYERS, dropout: float = config.CORRECTOR_DROPOUT):
        super().__init__()
        self.horizon = horizon
        self.rnn = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.onset_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1]
        return torch.cat([self.head(last), self.onset_head(last)], dim=1)


def multitask_loss(weight: float = config.ONSET_LOSS_WEIGHT):
    """MSE поправки + weight * BCE начала маневра; цель - (N, 2H + 1), последний столбец - метка 0/1."""
    def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (((pred[:, :-1] - y[:, :-1]) ** 2).mean()
                + weight * F.binary_cross_entropy_with_logits(pred[:, -1], y[:, -1]))
    return loss
