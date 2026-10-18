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
from models.stack import BLOCK_TYPES, StackCorrector  # noqa: F401 - этап 4б
from models.transformer import TransformerCorrector

CORRECTORS = {"GRU": GRUCorrector, "LSTM": LSTMCorrector, "Transformer": TransformerCorrector,
              "Mamba": MambaCorrector}


def build_corrector(name: str, n_features: int, horizon: int, window: int = config.WINDOW_SCANS,
                    hidden: int = None, layers: int = None, dropout: float = None):
    """
    window нужен только Transformer (размер позиционного кодирования). hidden, layers,
    dropout - гиперпараметры (этап 4); None - значения из config. hidden - ширина
    скрытого состояния (у Transformer и Mamba - d_model).
    """
    kwargs = {}
    if hidden is not None:
        kwargs["d_model" if name in ("Transformer", "Mamba") else "hidden"] = hidden
    if layers is not None:
        kwargs["layers"] = layers
    if dropout is not None:
        kwargs["dropout"] = dropout
    if name == "Transformer":
        kwargs["window"] = window
    return CORRECTORS[name](n_features, horizon, **kwargs)
