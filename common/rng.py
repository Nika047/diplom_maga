"""
Генераторы случайных чисел проекта.

Все потоки порождаются из config.RANDOM_SEED в порядке config.RNG_STREAMS, поэтому
поток с данным именем всегда одинаков, независимо от того, какие ещё потоки
использовались в запуске.
"""

import numpy as np

import config


def make_rng(stream: str, seed: int = config.RANDOM_SEED) -> np.random.Generator:
    """Генератор для именованного потока из config.RNG_STREAMS."""
    if stream not in config.RNG_STREAMS:
        raise ValueError(f"Неизвестный поток случайных чисел: {stream}")
    children = np.random.SeedSequence(seed).spawn(len(config.RNG_STREAMS))
    return np.random.default_rng(children[config.RNG_STREAMS.index(stream)])
