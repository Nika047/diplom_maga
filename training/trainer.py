"""
Обучение корректора: AdamW, MSE по нормированной поправке, ранняя остановка по val.

Лучшие веса (минимум val loss) возвращаются вместе с историей по эпохам. Оценка на
val - без teacher forcing (у корректора его и нет: прогноз за один проход).

Проверяется на этапе 2: overfit one batch, падение loss при обучении.
"""

import time

import numpy as np
import torch
from torch import nn

from common.logging_setup import get_logger


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(n: int, batch: int, rng: np.random.Generator = None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch):
        yield order[i:i + batch]


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for idx in _batches(len(x), batch):
        out.append(model(torch.from_numpy(x[idx]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + tuple(model.head.bias.shape))


@torch.no_grad()
def mse(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch: int = 4096) -> float:
    model.eval()
    total, n = 0.0, 0
    for idx in _batches(len(x), batch):
        pred = model(torch.from_numpy(x[idx]).to(device))
        total += float(((pred - torch.from_numpy(y[idx]).to(device)) ** 2).sum())
        n += pred.numel()
    return total / max(n, 1)


def overfit_one_batch(model: nn.Module, x: np.ndarray, y: np.ndarray, steps: int, lr: float,
                      device: torch.device) -> tuple:
    """Обучение на одном батче. Возвращает (loss в начале, loss в конце)."""
    model.to(device).train()
    xb, yb = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = last = None
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(xb) - yb) ** 2).mean()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    return first, last


def train(model: nn.Module, x_tr, y_tr, x_val, y_val, params: dict, device: torch.device,
          rng: np.random.Generator) -> tuple:
    """Возвращает (история по эпохам, лучшие веса)."""
    log = get_logger()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 1e-4))
    history = []
    best_val, best_state, bad = float("inf"), None, 0
    val0 = mse(model, x_val, y_val, device)
    log.info("val loss до обучения (нулевая поправка = IMM): %.4f", val0)
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for idx in _batches(len(x_tr), params["batch"], rng):
            xb = torch.from_numpy(x_tr[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(y_tr[idx]).to(device, non_blocking=True)
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.get("clip", 1.0))
            opt.step()
            total += float(loss) * len(idx)
            n += len(idx)
        train_loss = total / max(n, 1)
        val_loss = mse(model, x_val, y_val, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": time.time() - t0})
        log.info("эпоха %3d: train %.4f, val %.4f, %.1f с", epoch, train_loss, val_loss, time.time() - t0)
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= params["patience"]:
                log.info("ранняя остановка: val не улучшался %d эпох", bad)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_loss_initial": val0, "best_val_loss": best_val, "epochs": history}, best_state
