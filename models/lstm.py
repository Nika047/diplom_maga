"""
Корректор на LSTM - то же устройство, что GRU (models/gru.py): рекуррентная сеть
по окну, поправка по последнему скрытому состоянию.

Проверяется на этапе 3: смоук forward / backward, нулевой выход до обучения,
overfit one batch.
"""

import torch
from torch import nn

import config


class LSTMCorrector(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = config.CORRECTOR_HIDDEN,
                 layers: int = config.CORRECTOR_LAYERS, dropout: float = config.CORRECTOR_DROPOUT):
        super().__init__()
        self.horizon = horizon
        self.rnn = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).view(-1, self.horizon, 2)
