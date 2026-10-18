"""
Корректор на Transformer (кодировщик): каждый обзор окна - токен, обучаемое
позиционное кодирование, самовнимание по всему окну, поправка по последнему токену.

Маска внимания не нужна: окно целиком из прошлого относительно момента прогноза k.
Сложность самовнимания растёт квадратично с длиной окна (у Mamba - линейно); при
окне 20 обзоров это не ограничение, сравнение идёт по точности.

Проверяется на этапе 3: смоук forward / backward, нулевой выход до обучения,
overfit one batch.
"""

import torch
from torch import nn

import config


class TransformerCorrector(nn.Module):
    def __init__(self, n_features: int, horizon: int, d_model: int = config.CORRECTOR_HIDDEN,
                 layers: int = config.CORRECTOR_LAYERS, heads: int = config.TRANSFORMER_HEADS,
                 dropout: float = config.CORRECTOR_DROPOUT, window: int = config.WINDOW_SCANS):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.pos[:, -x.shape[1]:]
        h = self.norm(self.encoder(h))
        return self.head(h[:, -1]).view(-1, self.horizon, 2)
