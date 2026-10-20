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


def _mse_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).mean()


@torch.no_grad()
def mse(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch: int = 4096,
        loss_fn=None) -> float:
    """Средний loss на наборе. loss_fn(pred, y) - своя функция потерь (этап 5), по умолчанию MSE."""
    model.eval()
    total, n = 0.0, 0
    for idx in _batches(len(x), batch):
        pred = model(torch.from_numpy(x[idx]).to(device))
        yb = torch.from_numpy(y[idx]).to(device)
        if loss_fn is None:
            total += float(((pred - yb) ** 2).sum())
            n += pred.numel()
        else:
            total += float(loss_fn(pred, yb)) * len(idx)
            n += len(idx)
    return total / max(n, 1)


def overfit_one_batch(model: nn.Module, x: np.ndarray, y: np.ndarray, steps: int, lr: float,
                      device: torch.device, loss_fn=None) -> tuple:
    """Обучение на одном батче. Возвращает (loss в начале, loss в конце).

    Dropout на время проверки отключается: это регуляризация, она мешает запомнить
    батч, и при dropout 0,3 маленькая сеть не доходит до порога (этап 4б).
    """
    model.to(device).train()
    saved = []
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.RNNBase, nn.MultiheadAttention)):
            attr = "p" if isinstance(mod, nn.Dropout) else "dropout"
            saved.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, 0.0)
    xb, yb = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or _mse_loss
    first = last = None
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    for mod, attr, value in saved:
        setattr(mod, attr, value)
    return first, last


def train(model: nn.Module, x_tr, y_tr, x_val, y_val, params: dict, device: torch.device,
          rng: np.random.Generator, quiet: bool = False, loss_fn=None) -> tuple:
    """
    Возвращает (история по эпохам, лучшие веса). quiet - без строки на каждую эпоху
    (подбор). loss_fn(pred, y) - своя функция потерь, по умолчанию MSE.
    """
    log = get_logger()
    log_epoch = log.debug if quiet else log.info
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 1e-4))
    history = []
    best_val, best_state, bad = float("inf"), None, 0
    val0 = mse(model, x_val, y_val, device, loss_fn=loss_fn)
    log_epoch("val loss до обучения (нулевой выход: у гибрида = IMM): %.4f", val0)
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for idx in _batches(len(x_tr), params["batch"], rng):
            xb = torch.from_numpy(x_tr[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(y_tr[idx]).to(device, non_blocking=True)
            opt.zero_grad()
            loss = (loss_fn or _mse_loss)(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.get("clip", 1.0))
            opt.step()
            total += float(loss.detach()) * len(idx)
            n += len(idx)
        train_loss = total / max(n, 1)
        val_loss = mse(model, x_val, y_val, device, loss_fn=loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": time.time() - t0})
        log_epoch("эпоха %3d: train %.4f, val %.4f, %.1f с", epoch, train_loss, val_loss, time.time() - t0)
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= params["patience"]:
                log_epoch("ранняя остановка: val не улучшался %d эпох", bad)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_loss_initial": val0, "best_val_loss": best_val, "epochs": history}, best_state
