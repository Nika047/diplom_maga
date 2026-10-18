"""
Нейросетевые корректоры прогноза IMM.

Все корректоры: вход (B, W, F) - окно признаков, выход (B, H, 2) - нормированная
поправка по горизонтали. Последний слой инициализирован нулями: необученный
корректор даёт нулевую поправку, то есть ровно прогноз IMM.

Те же сети используются и как сквозные модели без IMM (этап 3): тогда вход - окно
только из отметок, выход - смещение от последней отметки.
"""

import config
from models.gru import GRUCorrector
from models.lstm import LSTMCorrector
from models.mamba import MambaCorrector
from models.transformer import TransformerCorrector

CORRECTORS = {"GRU": GRUCorrector, "LSTM": LSTMCorrector, "Transformer": TransformerCorrector,
              "Mamba": MambaCorrector}


def build_corrector(name: str, n_features: int, horizon: int, window: int = config.WINDOW_SCANS):
    """window нужен только Transformer (размер позиционного кодирования)."""
    if name == "Transformer":
        return CORRECTORS[name](n_features, horizon, window=window)
    return CORRECTORS[name](n_features, horizon)
