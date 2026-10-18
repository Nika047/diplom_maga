"""
PNG-рисунки этапов (matplotlib без окна, бэкенд Agg).

Цвета участков едины для всех рисунков: прямая - серый, вход в маневр - оранжевый,
маневр - красный, выход - синий, медленно - светло-зелёный.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from data.phases import turn_omegas  # noqa: E402

PHASE_COLORS = {"straight": "#9e9e9e", "entry": "#ff9800", "turn": "#d32f2f",
                "exit": "#1976d2", "slow": "#a5d6a7"}
PHASE_RU = {"straight": "прямая", "entry": "вход", "turn": "маневр", "exit": "выход", "slow": "медленно"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _phase_legend(ax) -> None:
    handles = [plt.Line2D([], [], marker="o", ls="", color=PHASE_COLORS[p], label=PHASE_RU[p])
               for p in config.PHASES]
    ax.legend(handles=handles, loc="best", fontsize=7)


def plot_tracks_phases(tracks: list, path: Path, title: str, ncols: int = 4) -> None:
    """Сетка траекторий в плане (x - восток, y - север), точки окрашены по участкам."""
    n = len(tracks)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, tr in zip(axes.flat, tracks):
        ax.set_visible(True)
        ax.plot(tr.pos[:, 0], tr.pos[:, 1], color="#cccccc", lw=0.8, zorder=1)
        for code, name in enumerate(config.PHASES):
            sel = tr.phase == code
            ax.scatter(tr.pos[sel, 0], tr.pos[sel, 1], s=6, color=PHASE_COLORS[name], zorder=2)
        ax.plot(*tr.pos[0, :2], marker="^", color="black", ms=7, zorder=3)
        ax.set_title(f"{tr.flight_id}\n{tr.duration_s / 60:.1f} мин", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    _phase_legend(axes.flat[0])
    fig.suptitle(title + "  (треугольник - начало; оси в метрах)", fontsize=11)
    _save(fig, path)


def plot_track_timeline(track, path: Path) -> None:
    """Скорость и скорость разворота во времени с подсветкой участков."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    t = track.t / 60.0
    for ax in (ax1, ax2):
        for code, name in enumerate(config.PHASES):
            if name == "straight":
                continue
            sel = track.phase == code
            ax.fill_between(t, 0, 1, where=sel, color=PHASE_COLORS[name], alpha=0.3,
                            transform=ax.get_xaxis_transform(), step="mid", lw=0)
    ax1.plot(t, track.speed, lw=0.8, color="black")
    ax1.set_ylabel("скорость, м/с")
    ax2.plot(t, track.omega, lw=0.8, color="black")
    for level in (config.OMEGA_TURN_ON, config.OMEGA_TURN_OFF):
        ax2.axhline(level, color="gray", ls="--", lw=0.6)
        ax2.axhline(-level, color="gray", ls="--", lw=0.6)
    ax2.set_ylabel("w, град/с")
    ax2.set_xlabel("время, мин")
    _phase_legend(ax1)
    fig.suptitle(f"{track.flight_id}: разметка участков (пунктир - пороги входа и выхода из маневра)")
    _save(fig, path)


def _moving_speed(tracks: list) -> np.ndarray:
    vals = [tr.speed[tr.speed >= config.PHASE_V_MIN] for tr in tracks]
    return np.concatenate(vals) if vals else np.array([])


def plot_distributions(real: list, synth: list, path: Path) -> None:
    """Гистограммы скорости и |w| на маневрах: логи (train) против синтетики."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for data, label, color in ((_moving_speed(real), "логи (train)", "#1976d2"),
                               (_moving_speed(synth), "синтетика", "#d32f2f")):
        ax1.hist(data, bins=60, range=(0, 60), density=True, alpha=0.5, label=label, color=color)
    ax1.set_xlabel("горизонтальная скорость, м/с")
    ax1.set_ylabel("плотность")
    ax1.legend()
    for data, label, color in ((turn_omegas(real), "логи (train)", "#1976d2"),
                               (turn_omegas(synth), "синтетика", "#d32f2f")):
        ax2.hist(data, bins=60, range=(0, 40), density=True, alpha=0.5, label=label, color=color)
    ax2.set_xlabel("|w| на участках маневра, град/с")
    ax2.legend()
    fig.suptitle("Сравнение распределений логов и синтетики")
    _save(fig, path)


def plot_phase_shares(shares: dict, path: Path) -> None:
    """Столбцы долей участков; shares: {группа: {участок: доля}}."""
    groups = list(shares)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(groups)), 4))
    bottom = np.zeros(len(groups))
    for name in config.PHASES:
        vals = np.array([shares[g].get(name, 0.0) for g in groups])
        ax.bar(groups, vals, bottom=bottom, color=PHASE_COLORS[name], label=PHASE_RU[name])
        bottom += vals
    ax.set_ylabel("доля времени")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Доли участков по типам аппаратов и синтетике")
    _save(fig, path)
