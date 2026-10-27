"""
Один эксперимент с корректором: нормализация по train, overfit one batch, обучение,
прогноз на val, сохранение весов и кривой loss. Общий для этапа 3 и эксперимента
с длиной окна, чтобы у всех моделей был один и тот же порядок действий.

Проверки (в сводку): overfit one batch, «модель обучилась» (val ниже, чем у нулевого
выхода), train loss падает.

Хранилище (версия 2): store + store_key=(этап, случай) - готовая модель берётся из хранилища
без обучения (проверки воспроизводятся из сохранённых), прерванное обучение продолжается с
контрольной точки (training/store.py).

Веса примеров (этап 13): weights=(веса train, веса val) - цель упаковывается вместе с весом
(trainer.weighted_mse), и обучение, overfit и ранняя остановка идут по взвешенному loss. Без
весов всё как раньше.
"""

import re
import time
from pathlib import Path

import numpy as np
import torch

import config
from common.rng import make_rng
from models import build_corrector
from training import dataset
from training.store import ModelStore
from training.trainer import overfit_one_batch, pick_device, predict, train, weighted_mse
from viz import train_plots


def slug(label: str) -> str:
    """Имя файла по подписи: без символов, недопустимых в Windows (двоеточие, косая черта, звёздочка и т. д.)."""
    return re.sub(r"[^\w.,-]+", "_", label.replace("+", "_")).strip("_")


def run_experiment(label: str, corrector: str, mode: str, ws_tr, ws_val, params: dict, device,
                   summary, log, smoke: bool, out_dir: Path, art_dir: Path,
                   window: int = config.WINDOW_SCANS, seed: int = config.RANDOM_SEED,
                   model_kwargs: dict = None, overfit: bool = True, extra_sets: dict = None,
                   weights: tuple = None, store: ModelStore = None, store_key: tuple = None) -> dict:
    """
    Возвращает {"pred": поправки или смещения на val в метрах (N, H, 2), "history": ...,
    "extra": {имя: прогноз на наборе extra_sets[имя]}}.

    seed - для инициализации сети и порядка батчей (этап 4: несколько seed);
    model_kwargs - гиперпараметры сети (hidden, layers, dropout);
    extra_sets - другие наборы окон (test, holdout), на которых нужен прогноз;
    store, store_key - хранилище моделей и (этап, случай) ключа (подпись и seed берутся отсюда).
    """
    model_kwargs = model_kwargs or {}
    tr, va = ws_tr.view(mode), ws_val.view(mode)
    key = (store_key[0], store_key[1], label, seed) if store is not None else None
    if key is not None and store.has(*key):
        return _reuse(store, key, label, corrector, mode, window, model_kwargs, va, extra_sets, summary, log,
                      out_dir)
    norm = dataset.Normalizer.fit(tr)
    x_tr, y_tr, x_val, y_val = norm.x(tr.x), norm.y(tr.y), norm.x(va.x), norm.y(va.y)
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        summary.check(name, passed, detail)
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    loss_fn = None
    if weights is not None:
        def pack(y, w):
            return np.concatenate([y.reshape(len(y), -1), np.asarray(w, np.float32)[:, None]], axis=1)
        y_tr, y_val, loss_fn = pack(y_tr, weights[0]), pack(y_val, weights[1]), weighted_mse

    if overfit:
        rng = make_rng("training", seed)
        batch_idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
        torch.manual_seed(seed)
        model = build_corrector(corrector, x_tr.shape[-1], config.HORIZON_SCANS, window, **model_kwargs)
        first, last = overfit_one_batch(model, x_tr[batch_idx], y_tr[batch_idx], config.OVERFIT_STEPS, 3e-3,
                                        device, loss_fn=loss_fn)
        check(f"{label}: overfit one batch", last < config.OVERFIT_TARGET * first,
              f"loss {first:.4f} -> {last:.4f}")

    log.info("Обучение %s", label)
    rng = make_rng("training", seed)
    torch.manual_seed(seed)
    model = build_corrector(corrector, x_tr.shape[-1], config.HORIZON_SCANS, window, **model_kwargs)
    t0 = time.time()
    ckpt = store.checkpoint(*key) if key is not None else None
    history, _ = train(model, x_tr, y_tr, x_val, y_val, params, device, rng, loss_fn=loss_fn, ckpt_path=ckpt)
    train_s = time.time() - t0
    ep = history["epochs"]
    check(f"{label}: модель обучилась (val ниже начального)",
          history["best_val_loss"] < history["val_loss_initial"],
          f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
    check(f"{label}: train loss падает", len(ep) >= 2 and ep[-1]["train_loss"] < ep[0]["train_loss"],
          f"{ep[0]['train_loss']:.4f} -> {ep[-1]['train_loss']:.4f} за {len(ep)} эпох")

    t0 = time.time()
    out = norm.y_inv(predict(model, x_val, device))
    infer_us = (time.time() - t0) / max(len(x_val), 1) * 1e6
    extra = {key: norm.y_inv(predict(model, norm.x(ws.view(mode).x), device))
             for key, ws in (extra_sets or {}).items()}
    name = slug(label)
    art_dir.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "corrector": corrector,
               "mode": mode, "window": window, "seed": seed, "model_kwargs": model_kwargs, "smoke": smoke}
    torch.save(payload, art_dir / f"{name}{'_smoke' if smoke else ''}.pt")
    history.update({"train_seconds": train_s, "infer_us_per_sample": infer_us, "device": str(device),
                    "params": params, "n_params": sum(p.numel() for p in model.parameters()), "window": window,
                    "seed": seed, "model_kwargs": model_kwargs})
    if key is not None:
        store.save(*key, payload, {"history": history, "checks": checks})
    train_plots.plot_history(history, out_dir / f"loss_{name}.png", f"Обучение {label}")
    return {"pred": out, "history": history, "extra": extra}


def _reuse(store: ModelStore, key: tuple, label: str, corrector: str, mode: str, window: int, model_kwargs: dict,
           va, extra_sets: dict, summary, log, out_dir: Path) -> dict:
    """Готовая модель из хранилища: проверки из сохранённых, прогноз без обучения."""
    payload, meta = store.load(*key)
    log.info("%s: модель из хранилища (%s), обучение пропущено", label, store.folder(*key))
    for c in meta["checks"]:
        summary.check(c["name"], c["passed"], c["detail"] + " [из хранилища]")
    norm = dataset.Normalizer.from_dict(payload["normalizer"])
    x_val = norm.x(va.x)
    model = build_corrector(corrector, x_val.shape[-1], config.HORIZON_SCANS, window, **model_kwargs)
    model.load_state_dict(payload["state_dict"])
    t0 = time.time()
    out = norm.y_inv(predict(model, x_val, pick_device()))
    meta["history"]["infer_us_per_sample"] = (time.time() - t0) / max(len(x_val), 1) * 1e6
    extra = {k: norm.y_inv(predict(model, norm.x(ws.view(mode).x), pick_device()))
             for k, ws in (extra_sets or {}).items()}
    train_plots.plot_history(meta["history"], out_dir / f"loss_{slug(label)}.png", f"Обучение {label}")
    return {"pred": out, "history": meta["history"], "extra": extra}

