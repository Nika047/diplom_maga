"""
Обучение произвольной сети через хранилище моделей (схемы Б этапа 5, гипотезы этапов 6-7).

fit_model: если модель (этап, случай, подпись, seed) уже в хранилище - веса и проверки берутся оттуда
(обучение пропускается); иначе - проверка overfit one batch на первом seed, обучение с контрольной точкой
(прерванное обучение продолжается), сохранение. Возвращает (сеть, история, payload).
"""

import time

import numpy as np
import torch

import config
from common.rng import make_rng
from pipeline.stage import Ctx
from training.trainer import overfit_one_batch, train
from viz import train_plots


def fit_model(ctx: Ctx, stage: str, label: str, seed: int, build, x_tr, y_tr, x_val, y_val, params: dict,
              loss_fn=None, overfit: dict = None, payload_extra: dict = None, plot: bool = True) -> tuple:
    """
    build() -> новая сеть. overfit - проверка на одном батче (только для первого seed вызывающим):
    {"batch": 256, "lr": 3e-3, "steps": config.OVERFIT_STEPS, "clip": None, "ok": lambda first, last: bool}.
    """
    summary, log, store = ctx.summary, ctx.log, ctx.store
    key = (stage, ctx.case, label, seed)
    name = f"{label}, seed {seed}"
    if store.has(*key):
        payload, meta = store.load(*key)
        log.info("%s: модель из хранилища, обучение пропущено", name)
        for c in meta["checks"]:
            summary.check(c["name"], c["passed"], c["detail"] + " [из хранилища]")
        model = build()
        model.load_state_dict(payload["state_dict"])
        return model, meta["history"], payload
    checks = []

    def check(title: str, ok: bool, detail: str) -> None:
        summary.check(title, ok, detail)
        checks.append({"name": title, "passed": bool(ok), "detail": detail})

    if overfit:
        rng = make_rng("training", seed)
        idx = rng.choice(len(x_tr), size=min(overfit.get("batch", 256), len(x_tr)), replace=False)
        torch.manual_seed(seed)
        first, last = overfit_one_batch(build(), x_tr[idx], y_tr[idx], overfit.get("steps", config.OVERFIT_STEPS),
                                        overfit.get("lr", 3e-3), ctx.device, loss_fn=loss_fn, clip=overfit.get("clip"))
        ok = overfit.get("ok", lambda f, l: l < config.OVERFIT_TARGET * f)(first, last)
        check(f"{name}: overfit one batch", ok, f"loss {first:.4f} -> {last:.4f}")
    rng = make_rng("training", seed)
    torch.manual_seed(seed)
    model = build()
    t0 = time.time()
    log.info("Обучение %s", name)
    history, _ = train(model, x_tr, y_tr, x_val, y_val, params, ctx.device, rng, loss_fn=loss_fn,
                       ckpt_path=store.checkpoint(*key))
    history["train_seconds"] = time.time() - t0
    history["n_params"] = int(sum(p.numel() for p in model.parameters()))
    check(f"{name}: модель обучилась (val ниже начального)", history["best_val_loss"] < history["val_loss_initial"],
          f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
    payload = {"state_dict": model.state_dict(), "seed": seed, "smoke": ctx.smoke, **(payload_extra or {})}
    store.save(*key, payload, {"history": history, "checks": checks})
    if plot:
        slug = "".join(ch if ch.isalnum() else "_" for ch in label)
        train_plots.plot_history(history, ctx.out_dir / f"loss_{slug}_{seed}.png", f"Обучение: {name}")
    return model, history, payload


def ms_per_scan_sequence(model: torch.nn.Module, width: int, length: int = 60) -> float:
    """Время шага фильтра с сетью внутри (NeuralKF) на процессоре, мс на обзор: проход по куску длины length."""
    from eval.latency import median_ms

    model = model.to("cpu").eval()
    x = torch.zeros(1, length, width)
    x[..., 2:6] = torch.eye(2).reshape(1, 1, 4)
    x[..., 6] = 1.0

    def fn():
        with torch.no_grad():
            model(x)

    return median_ms(fn, repeats=5) / length
