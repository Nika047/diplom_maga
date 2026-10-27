"""
Доверительные интервалы бутстрепом по полётам (версия 2) и правило отбора гипотез.

Единица пересэмплирования - полёт целиком (все трассы одного лога; у синтетики - одна
траектория): окна и обзоры внутри полёта сильно связаны, и пересэмплирование по обзорам дало бы
слишком узкие интервалы. Для каждого полёта хранятся сумма горизонтальных ошибок и число пар
(k, h); ADE выборки = сумма сумм / сумма чисел.

Парное сравнение: оба метода считаются на одних и тех же полётах, и в каждой выборке полёты
берутся одни и те же, поэтому интервал разности уже, чем у разности двух интервалов.

Проверялось на подставных данных с известным ответом: постоянная разность даёт вырожденный
интервал, шум нулевого среднего - интервал, накрывающий 0; результат воспроизводим (поток "bootstrap").
"""

import numpy as np

import config
from common.rng import make_rng


def flight_sums(pairs_per_track: list, groups: list, mask_fn=None) -> tuple:
    """
    По трассам -> по полётам. pairs_per_track - eval.metrics.forecast_pairs для каждой трассы,
    groups - ключ полёта каждой трассы (log_file или flight_id), mask_fn(pairs) -> булева маска
    пар (участок, горизонт). Возвращает (ключи полётов, суммы ошибок, числа пар).
    """
    acc = {}
    for pairs, key in zip(pairs_per_track, groups):
        dist = np.linalg.norm(pairs["err"][:, :2], axis=1)
        if mask_fn is not None:
            dist = dist[mask_fn(pairs)]
        total, count = acc.get(key, (0.0, 0))
        acc[key] = (total + float(dist.sum()), count + int(len(dist)))
    keys = sorted(acc)
    return (keys, np.array([acc[k][0] for k in keys]), np.array([acc[k][1] for k in keys], dtype=float))


def _resample_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def ade_interval(sums: np.ndarray, counts: np.ndarray, n_boot: int = config.BOOTSTRAP_N,
                 level: float = config.BOOTSTRAP_LEVEL, rng: np.random.Generator = None) -> dict:
    """ADE и его интервал по полётам: {"ade", "lo", "hi", "n_flights"}."""
    rng = rng or make_rng("bootstrap")
    idx = _resample_indices(len(sums), n_boot, rng)
    ades = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    return {"ade": float(sums.sum() / counts.sum()), "lo": float(np.quantile(ades, alpha)),
            "hi": float(np.quantile(ades, 1.0 - alpha)), "n_flights": int(len(sums))}


def paired_difference(sums_a: np.ndarray, counts_a: np.ndarray, sums_b: np.ndarray, counts_b: np.ndarray,
                      n_boot: int = config.BOOTSTRAP_N, level: float = config.BOOTSTRAP_LEVEL,
                      rng: np.random.Generator = None) -> dict:
    """
    Разность ADE метода A и метода B (A - B) на одних и тех же полётах с интервалом.
    {"diff", "lo", "hi", "rel", "better"}: rel - разность в долях ADE метода B (основы),
    better - интервал целиком левее нуля (A значимо точнее B).
    """
    if sums_a.shape != sums_b.shape:
        raise ValueError("Парное сравнение: наборы полётов методов должны совпадать")
    rng = rng or make_rng("bootstrap")
    idx = _resample_indices(len(sums_a), n_boot, rng)
    ade_a = sums_a[idx].sum(axis=1) / counts_a[idx].sum(axis=1)
    ade_b = sums_b[idx].sum(axis=1) / counts_b[idx].sum(axis=1)
    diff = ade_a - ade_b
    alpha = (1.0 - level) / 2.0
    lo, hi = float(np.quantile(diff, alpha)), float(np.quantile(diff, 1.0 - alpha))
    base = float(sums_b.sum() / counts_b.sum())
    point = float(sums_a.sum() / counts_a.sum() - base)
    return {"diff": point, "lo": lo, "hi": hi, "rel": point / base, "better": hi < 0.0}


def selection_rule(result: dict) -> bool:
    """Правило отбора гипотезы (PLAN.md, 5а): на val интервал разности с основой целиком левее нуля."""
    return bool(result["better"])
