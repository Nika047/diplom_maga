"""
Рисунки этапа 1: отметки и оценки фильтров, вероятности моделей IMM, кривые
динамической ошибки, ошибка по горизонтам и по участкам.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from viz.plots import PHASE_COLORS, PHASE_RU, _save  # noqa: E402

METHOD_COLORS = {"CV": "#7f7f7f", "KF": "#1976d2", "IMM": "#d32f2f", "IMM+GRU": "#2e7d32",
                 "IMM+LSTM": "#6a1b9a", "IMM+Transformer": "#ef6c00", "IMM+Mamba": "#00838f",
                 "GRU (без IMM)": "#9ccc65", "Mamba (без IMM)": "#80deea"}
MODEL_COLORS = {"CV": "#1976d2", "CA": "#2e7d32", "CT": "#d32f2f"}


def _shade_phases(ax, t, phase) -> None:
    for code, name in enumerate(config.PHASES):
        if name == "straight":
            continue
        ax.fill_between(t, 0, 1, where=phase == code, color=PHASE_COLORS[name], alpha=0.25,
                        transform=ax.get_xaxis_transform(), step="mid", lw=0)


def plot_track_filters(rt, run, path: Path, title: str, k_range=None) -> None:
    """Слева - план (истина, отметки, оценка IMM), справа - вероятности моделей во времени."""
    k0, k1 = k_range if k_range else (0, len(rt.t))
    sl = slice(k0, k1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    ax1.plot(rt.meas[sl, 0], rt.meas[sl, 1], ".", color="#9e9e9e", ms=4, label="отметки РЛС")
    ax1.plot(rt.truth[sl, 0], rt.truth[sl, 1], "-", color="black", lw=1.2, label="истина")
    ax1.plot(run.est[sl, 0], run.est[sl, 1], "-", color="#d32f2f", lw=1.0, label="оценка IMM")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("x (восток), м")
    ax1.set_ylabel("y (север), м")
    ax1.legend(fontsize=8)
    t = (rt.t[sl] - rt.t[k0]) / 60.0
    _shade_phases(ax2, t, rt.phase[sl])
    for j, name in enumerate(config.IMM_MODELS):
        ax2.plot(t, run.mu[sl, j], color=MODEL_COLORS[name], lw=1.0, label=f"вероятность {name}")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("время, мин")
    ax2.set_ylabel("вероятность модели")
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{title}\n(подсветка: оранжевый - вход в маневр, красный - маневр, синий - выход)", fontsize=10)
    _save(fig, path)


def plot_dynamic_curves(curves: dict, levels: dict, path: Path, title: str) -> None:
    """curves: {(метод, h): (m, кривая, число)}; levels: {(метод, h): уровень прямой}."""
    hs = sorted({h for _, h in curves})
    fig, axes = plt.subplots(1, len(hs), figsize=(6.5 * len(hs), 4.2), squeeze=False)
    for ax, h in zip(axes[0], hs):
        for (method, hh), (ms, curve, counts) in curves.items():
            if hh != h:
                continue
            color = METHOD_COLORS[method]
            ax.plot(ms, curve, "-o", ms=3, color=color, label=method)
            ax.axhline(levels[(method, h)], color=color, ls=":", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("обзоров от начала маневра")
        ax.set_ylabel("средняя горизонтальная ошибка, м")
        ax.set_title(f"горизонт {h} обз. ({h * config.SCAN_PERIOD:.0f} с)", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title}\n(пунктир - средняя ошибка того же метода на прямой)", fontsize=10)
    fig.subplots_adjust(top=0.82)
    _save(fig, path)


def plot_by_horizon(ade: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    hs = np.arange(1, config.HORIZON_SCANS + 1) * config.SCAN_PERIOD
    for method, vals in ade.items():
        ax.plot(hs, vals, "-o", color=METHOD_COLORS[method], label=method)
    ax.set_xlabel("горизонт прогноза, с")
    ax.set_ylabel("ADE по горизонтали, м")
    ax.legend()
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_by_phase(table: dict, path: Path, title: str) -> None:
    """table: {метод: {участок: ADE}}."""
    phases = [p for p in config.PHASES if p != "slow"]
    methods = list(table)
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(phases))
    for i, method in enumerate(methods):
        ax.bar(x + i * width, [table[method].get(p, np.nan) for p in phases], width,
               color=METHOD_COLORS[method], label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([PHASE_RU[p] for p in phases])
    ax.set_ylabel("ADE по горизонтали, м")
    ax.legend()
    ax.set_title(title, fontsize=10)
    _save(fig, path)
