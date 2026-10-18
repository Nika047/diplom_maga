"""
Генераторы случайных чисел проекта.

Все потоки порождаются из config.RANDOM_SEED в порядке config.RNG_STREAMS, поэтому
поток с данным именем всегда одинаков, независимо от того, какие ещё потоки
использовались в запуске.
"""

import zlib

import numpy as np

import config


def make_rng(stream: str, seed: int = config.RANDOM_SEED) -> np.random.Generator:
    """Генератор для именованного потока из config.RNG_STREAMS."""
    if stream not in config.RNG_STREAMS:
        raise ValueError(f"Неизвестный поток случайных чисел: {stream}")
    children = np.random.SeedSequence(seed).spawn(len(config.RNG_STREAMS))
    return np.random.default_rng(children[config.RNG_STREAMS.index(stream)])


def make_track_rng(stream: str, key: str, seed: int = config.RANDOM_SEED) -> np.random.Generator:
    """
    Генератор потока stream для одного объекта (например, трассы flight_id).

    Зависит только от seed, имени потока и ключа, поэтому шум одной трассы не
    меняется, если другие трассы добавлены, убраны или переставлены.
    """
    if stream not in config.RNG_STREAMS:
        raise ValueError(f"Неизвестный поток случайных чисел: {stream}")
    entropy = [seed, config.RNG_STREAMS.index(stream), zlib.crc32(key.encode("utf-8"))]
    return np.random.default_rng(np.random.SeedSequence(entropy))
