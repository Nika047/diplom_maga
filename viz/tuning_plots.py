"""Рисунки подбора и сравнения (версия 2): сходимость генетического подбора, фронт Парето «точность - время»."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from viz.filter_plots import method_color  # noqa: E402
from viz.plots import _save  # noqa: E402


def plot_convergence(studies: dict, path: Path, title: str, ylabel: str = "лучшая ADE, м") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, values in studies.items():
        if not len(values):
            continue
        best = np.minimum.accumulate(values)
        ax.plot(np.arange(1, len(best) + 1), best, "-", color=method_color(name), label=name)
    ax.set_xlabel("номер попытки")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_pareto(points: dict, path: Path, title: str, xlabel: str = "время отклика, мс",
                ylabel: str = "ADE val, м") -> None:
    """points: {метод: [(время, ADE, на_фронте)]}; фронт Парето соединён линией."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, pts in points.items():
        if not pts:
            continue
        xs, ys, front = (np.array(v) for v in zip(*pts))
        color = method_color(name)
        ax.scatter(xs, ys, s=14, color=color, alpha=0.35)
        f = front.astype(bool)
        if f.any():
            order = np.argsort(xs[f])
            ax.plot(xs[f][order], ys[f][order], "o-", color=color, label=name, ms=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_bars_ci(rows: list, path: Path, title: str, ylabel: str = "ADE, м") -> None:
    """rows: [(подпись, значение, нижняя граница, верхняя граница)] - столбцы с интервалом."""
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(rows) + 2), 4.8))
    x = np.arange(len(rows))
    vals = np.array([r[1] for r in rows])
    lo = np.array([r[1] - r[2] for r in rows])
    hi = np.array([r[3] - r[1] for r in rows])
    ax.bar(x, vals, color="#455a64", yerr=[lo, hi], capsize=3, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(max(0.0, vals.min() * 0.85), None)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    _save(fig, path)
