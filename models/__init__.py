"""
Нейросетевые корректоры прогноза IMM.

Все корректоры: вход (B, W, F) - окно признаков, выход (B, H, 2) - нормированная
поправка по горизонтали. Последний слой инициализирован нулями: необученный
корректор даёт нулевую поправку, то есть ровно прогноз IMM.
"""

from models.gru import GRUCorrector

CORRECTORS = {"GRU": GRUCorrector}


def build_corrector(name: str, n_features: int, horizon: int):
    return CORRECTORS[name](n_features, horizon)
