"""
Один эксперимент с корректором: нормализация по train, overfit one batch, обучение,
прогноз на val, сохранение весов и кривой loss. Общий для этапа 3 и эксперимента
с длиной окна, чтобы у всех моделей был один и тот же порядок действий.

Проверки (в сводку): overfit one batch, «модель обучилась» (val ниже, чем у нулевого
выхода), train loss падает.
"""

import time
from pathlib import Path

import torch

import config
from common.rng import make_rng
from models import build_corrector
from training import dataset
from training.trainer import overfit_one_batch, predict, train
from viz import train_plots


def slug(label: str) -> str:
    return label.replace("+", "_").replace(" ", "_").replace("(", "").replace(")", "")


def run_experiment(label: str, corrector: str, mode: str, ws_tr, ws_val, params: dict, device,
                   summary, log, smoke: bool, out_dir: Path, art_dir: Path,
                   window: int = config.WINDOW_SCANS) -> dict:
    """Возвращает {"pred": поправки или смещения на val в метрах (N, H, 2), "history": ...}."""
    tr, va = ws_tr.view(mode), ws_val.view(mode)
    norm = dataset.Normalizer.fit(tr)
    x_tr, y_tr, x_val, y_val = norm.x(tr.x), norm.y(tr.y), norm.x(va.x), norm.y(va.y)

    rng = make_rng("training")
    batch_idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector(corrector, x_tr.shape[-1], config.HORIZON_SCANS, window)
    first, last = overfit_one_batch(model, x_tr[batch_idx], y_tr[batch_idx], config.OVERFIT_STEPS, 3e-3, device)
    summary.check(f"{label}: overfit one batch", last < config.OVERFIT_TARGET * first,
                  f"loss {first:.4f} -> {last:.4f}")

    log.info("Обучение %s", label)
    rng = make_rng("training")
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector(corrector, x_tr.shape[-1], config.HORIZON_SCANS, window)
    t0 = time.time()
    history, _ = train(model, x_tr, y_tr, x_val, y_val, params, device, rng)
    train_s = time.time() - t0
    ep = history["epochs"]
    summary.check(f"{label}: модель обучилась (val ниже начального)",
                  history["best_val_loss"] < history["val_loss_initial"],
                  f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
    summary.check(f"{label}: train loss падает", len(ep) >= 2 and ep[-1]["train_loss"] < ep[0]["train_loss"],
                  f"{ep[0]['train_loss']:.4f} -> {ep[-1]['train_loss']:.4f} за {len(ep)} эпох")

    t0 = time.time()
    out = norm.y_inv(predict(model, x_val, device))
    infer_us = (time.time() - t0) / max(len(x_val), 1) * 1e6
    name = slug(label)
    art_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "corrector": corrector,
                "mode": mode, "window": window, "smoke": smoke}, art_dir / f"{name}{'_smoke' if smoke else ''}.pt")
    history.update({"train_seconds": train_s, "infer_us_per_sample": infer_us, "device": str(device),
                    "params": params, "n_params": sum(p.numel() for p in model.parameters()), "window": window})
    train_plots.plot_history(history, out_dir / f"loss_{name}.png", f"Обучение {label}")
    return {"pred": out, "history": history}
