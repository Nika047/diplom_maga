"""
Сравнение методов и отбор по val (версия 2): парный бутстреп по полётам и правило выбора основы.

  compare_methods - разность ADE двух методов с интервалом (метод a минус метод b);
  choose_base     - основа: лучшая по val; если есть методы, не значимо хуже лучшей (интервал разности
                    с лучшей накрывает 0), берётся самый быстрый из них по времени отклика на процессоре.
"""

import numpy as np

from common.rng import make_rng
from eval import bootstrap
from eval.bootstrap import selection_rule  # noqa: F401  правило отбора гипотез (PLAN.md, 5а)
from pipeline.registry import Registry


def compare_methods(reg: Registry, a: str, b: str, name: str = "val", n_boot: int = None) -> dict:
    import config

    ka, sa, ca = reg.flights(a, name)
    kb, sb, cb = reg.flights(b, name)
    if ka != kb:
        raise ValueError(f"Наборы полётов методов {a} и {b} на {name} не совпадают")
    return bootstrap.paired_difference(sa, ca, sb, cb, n_boot=n_boot or config.BOOTSTRAP_N, rng=make_rng("bootstrap"))


def compare_across(reg_a: Registry, a: str, reg_b: Registry, b: str, name: str = "val", n_boot: int = None) -> dict:
    """Разность ADE (a минус b) для методов из разных реестров (случаев) на одних и тех же полётах набора."""
    import config

    ka, sa, ca = reg_a.flights(a, name)
    kb, sb, cb = reg_b.flights(b, name)
    if ka != kb:
        raise ValueError(f"Наборы полётов {a} и {b} на {name} не совпадают")
    if not np.allclose(ca, cb):
        # обзоры оценки у случаев одни (ks по случаю без доплера); расхождение числа пар недопустимо
        raise ValueError(f"Числа пар {a} и {b} на {name} не совпадают")
    return bootstrap.paired_difference(sa, ca, sb, cb, n_boot=n_boot or config.BOOTSTRAP_N, rng=make_rng("bootstrap"))


def choose_base(reg: Registry, methods: list, name: str = "val", n_boot: int = None) -> dict:
    """{"best": лучший по ADE, "base": выбранная основа, "rows": [{method, ade, diff, lo, hi, ms, equal}]}."""
    ades = {m: reg.metric(m, name)[0] for m in methods}
    best = min(methods, key=ades.get)
    rows = []
    for m in methods:
        d = compare_methods(reg, m, best, name, n_boot) if m != best else {"diff": 0.0, "lo": 0.0, "hi": 0.0}
        ms = reg.extra(m, "total_cpu_ms", float("nan"))
        rows.append({"method": m, "ade": ades[m], "diff": d["diff"], "lo": d["lo"], "hi": d["hi"], "ms": ms,
                     "equal": m == best or d["lo"] <= 0.0})        # не значимо хуже лучшей
    equal = [r for r in rows if r["equal"]]
    base = min(equal, key=lambda r: (np.nan_to_num(r["ms"], nan=1e9), r["ade"]))["method"]
    return {"best": best, "base": base, "rows": rows}


def ade_ci(reg: Registry, method: str, name: str = "val", n_boot: int = None) -> tuple:
    """(ADE, нижняя, верхняя граница 95% интервала по полётам) метода на наборе."""
    import config

    _, sums, counts = reg.flights(method, name)
    ci = bootstrap.ade_interval(sums, counts, n_boot=n_boot or config.BOOTSTRAP_N, rng=make_rng("bootstrap"))
    return ci["ade"], ci["lo"], ci["hi"]
