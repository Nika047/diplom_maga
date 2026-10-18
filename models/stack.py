"""
Корректор «стопка блоков» для эволюционного поиска архитектуры (этап 4б).

Архитектура задаётся списком типов блоков, например ["Mamba", "Attention", "GRU"]:
  вход -> линейный слой в ширину d_model (+ позиционное кодирование, если есть внимание);
  каждый блок: x = x + Dropout(Блок(LayerNorm(x))) - остаточная связь;
  у блока внимания остаточная связь и нормализация свои (TransformerEncoderLayer);
  -> LayerNorm -> поправка по последнему обзору (последний слой нулевой: до обучения = IMM).
Типы блоков: GRU, LSTM, Mamba (models/mamba.py), Attention (самовнимание, 4 головы).

Проверяется на этапе 4б: смоук всех типов блоков, нулевой выход до обучения,
причинность стопки без внимания.
"""

import torch
from torch import nn

import config
from models.mamba import MambaBlock

BLOCK_TYPES = ("GRU", "LSTM", "Mamba", "Attention")


class _RNNBlock(nn.Module):
    def __init__(self, kind: str, d_model: int):
        super().__init__()
        cls = nn.GRU if kind == "GRU" else nn.LSTM
        self.rnn = cls(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class StackCorrector(nn.Module):
    def __init__(self, n_features: int, horizon: int, blocks: list, d_model: int = config.CORRECTOR_HIDDEN,
                 dropout: float = config.CORRECTOR_DROPOUT, window: int = config.WINDOW_SCANS):
        super().__init__()
        if not blocks:
            raise ValueError("Нужен хотя бы один блок")
        self.horizon, self.kinds = horizon, list(blocks)
        self.inp = nn.Linear(n_features, d_model)
        self.pos = (nn.Parameter(torch.zeros(1, window, d_model) + 0.02 * torch.randn(1, window, d_model))
                    if "Attention" in self.kinds else None)
        self.blocks, self.norms = nn.ModuleList(), nn.ModuleList()
        for kind in self.kinds:
            if kind == "Attention":
                block = nn.TransformerEncoderLayer(d_model, config.TRANSFORMER_HEADS, dim_feedforward=2 * d_model,
                                                   dropout=dropout, batch_first=True, norm_first=True)
            elif kind == "Mamba":
                block = MambaBlock(d_model)
            else:
                block = _RNNBlock(kind, d_model)
            self.blocks.append(block)
            self.norms.append(nn.LayerNorm(d_model))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        if self.pos is not None:
            h = h + self.pos[:, -x.shape[1]:]
        for kind, block, norm in zip(self.kinds, self.blocks, self.norms):
            h = block(h) if kind == "Attention" else h + self.drop(block(norm(h)))
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.sequence(x)[:, -1]).view(-1, self.horizon, 2)
