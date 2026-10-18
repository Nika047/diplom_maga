"""Рисунки обучения: кривые loss по эпохам."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from viz.plots import _save  # noqa: E402


def plot_history(history: dict, path: Path, title: str) -> None:
    epochs = [e["epoch"] for e in history["epochs"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, [e["train_loss"] for e in history["epochs"]], "-o", ms=3, label="train")
    ax.plot(epochs, [e["val_loss"] for e in history["epochs"]], "-o", ms=3, label="val")
    ax.axhline(history["val_loss_initial"], color="gray", ls=":", label="val до обучения (нулевой выход)")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("MSE нормированной поправки")
    ax.legend()
    ax.set_title(title, fontsize=10)
    _save(fig, path)
