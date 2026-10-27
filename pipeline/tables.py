"""Таблицы сводок версии 2 по записям реестра: метрики по seed, время отклика, затраты, разница с эталоном."""

import numpy as np

from pipeline.registry import Registry

COLS = (("ade", "ADE"), ("fde", "FDE"), ("straight", "прямая"), ("entry", "вход"), ("turn", "маневр"),
        ("exit", "выход"))


def _fmt(mean: float, std: float, n: int) -> str:
    return f"{mean:.1f}" if n == 1 else f"{mean:.1f}±{std:.1f}"


def method_table(reg: Registry, methods: list, name: str, ref: str = None, with_cost: bool = True) -> list:
    """Строки таблицы: метрики по набору name (среднее ± СКО по seed), к эталону ref (ADE, %), стоимость."""
    width = max(len(m) for m in methods) + 1
    head = f"{'метод':{width}s}" + "".join(f" {t:>11s}" for _, t in COLS) + (f" {'к эталону':>10s}" if ref else "")
    if with_cost:
        head += f" {'мс CPU':>7s} {'мс GPU':>7s} {'парам.':>8s} {'обуч., с':>9s}"
    lines = [head]
    ref_ade = reg.metric(ref, name)[0] if ref else None
    for m in methods:
        n = len(reg.seeds(m))
        row = f"{m:{width}s}" + "".join(f" {_fmt(*reg.metric(m, name, k), n):>11s}" for k, _ in COLS)
        if ref:
            row += f" {reg.metric(m, name)[0] / ref_ade - 1:+10.1%}"
        if with_cost:
            ex = reg.data[m][str(reg.seeds(m)[0])]["extra"]
            gpu = ex.get("total_gpu_ms")
            tsec = [reg.data[m][str(s)]["extra"].get("train_seconds") for s in reg.seeds(m)]
            tsec = [t for t in tsec if t is not None]
            row += (f" {ex.get('total_cpu_ms', float('nan')):7.2f} {gpu if gpu is not None else float('nan'):7.2f}"
                    f" {ex.get('n_params', 0):8d} {np.mean(tsec) if tsec else float('nan'):9.0f}")
        lines.append(row)
    return lines
