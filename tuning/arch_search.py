"""
Эволюционный многокритериальный поиск архитектуры гибрида (этап 4б).

Геном попытки:
  - число блоков 1..ARCH_MAX_BLOCKS и тип каждого (GRU, LSTM, Mamba, Attention);
  - ширина d_model (TUNE_HIDDEN), dropout, lr, батч, weight decay;
  - отбор признаков: для каждой группы (data/windows.feature_groups) - брать или нет;
    хотя бы одна группа должна остаться, иначе попытка отбрасывается.
Цели (обе минимизируются, NSGA-II строит фронт Парето):
  1) ADE гибрида на val, м;
  2) время прямого прохода сети по одному окну, мс (батч 1, eval/latency.net_ms; версия 2).

Время - замер, а не подсчёт операций, поэтому на другом устройстве другое; на одном
компьютере между попытками сравнимо (небольшой шум замера).

Проверяется на этапе 4б: бюджет выполнен, фронт Парето не пуст, исходная архитектура
(стопка GRU, GRU, все признаки) входит в подбор первой попыткой.
"""

import time

import numpy as np
import optuna
import torch

import config
from common.rng import make_rng
from data.windows import feature_groups
from eval.latency import net_ms
from models import BLOCK_TYPES, StackCorrector
from training import dataset
from training.trainer import predict, train

DEFAULT_ARCH = {"n_blocks": 2, "block_0": "GRU", "block_1": "GRU", "hidden": config.CORRECTOR_HIDDEN,
                "dropout": config.CORRECTOR_DROPOUT, "batch": 512, "lr": 1e-3, "weight_decay": 1e-4}


def default_params(n_models: int, doppler: bool = False) -> dict:
    params = dict(DEFAULT_ARCH)
    params.update({f"use_{i}": True for i in range(len(feature_groups(n_models, doppler)))})
    return params


def decode(params: dict, n_models: int, doppler: bool = False) -> dict:
    """Параметры попытки -> {blocks, hidden, dropout, columns, groups, train}. doppler - в окнах есть группа «доплер»."""
    groups = feature_groups(n_models, doppler)
    names = list(groups)
    used = [names[i] for i in range(len(names)) if params.get(f"use_{i}", True)]
    columns = [c for name in used for c in groups[name]]
    blocks = [params[f"block_{i}"] for i in range(params["n_blocks"])]
    return {"blocks": blocks, "hidden": params["hidden"], "dropout": params["dropout"], "columns": columns,
            "groups": used, "train": {"batch": params["batch"], "lr": params["lr"],
                                      "weight_decay": params["weight_decay"]}}


class ArchObjective:
    def __init__(self, ws_tr, ws_val, device, n_models: int, window: int, seed: int = config.RANDOM_SEED,
                 epochs: int = config.TUNE_EPOCHS, patience: int = config.TUNE_PATIENCE, doppler: bool = False):
        self.norm = dataset.Normalizer.fit(ws_tr)
        self.x_tr, self.y_tr = self.norm.x(ws_tr.x), self.norm.y(ws_tr.y)
        self.x_val, self.y_val = self.norm.x(ws_val.x), self.norm.y(ws_val.y)
        self.y_val_m = ws_val.y
        self.device, self.n_models, self.window = device, n_models, window
        self.seed, self.epochs, self.patience, self.doppler = seed, epochs, patience, doppler

    def __call__(self, trial: optuna.Trial):
        n_blocks = trial.suggest_int("n_blocks", 1, config.ARCH_MAX_BLOCKS)
        for i in range(n_blocks):
            trial.suggest_categorical(f"block_{i}", list(BLOCK_TYPES))
        trial.suggest_categorical("hidden", list(config.TUNE_HIDDEN))
        trial.suggest_float("dropout", *config.TUNE_DROPOUT)
        trial.suggest_categorical("batch", list(config.TUNE_BATCH))
        trial.suggest_float("lr", *config.TUNE_LR, log=True)
        trial.suggest_float("weight_decay", *config.TUNE_WEIGHT_DECAY, log=True)
        for i in range(len(feature_groups(self.n_models, self.doppler))):
            trial.suggest_categorical(f"use_{i}", [True, False])
        arch = decode(trial.params, self.n_models, self.doppler)
        if not arch["columns"]:
            raise optuna.TrialPruned("не выбрано ни одной группы признаков")
        cols = arch["columns"]
        rng = make_rng("training", self.seed)
        torch.manual_seed(self.seed)
        model = StackCorrector(len(cols), config.HORIZON_SCANS, arch["blocks"], arch["hidden"], arch["dropout"],
                               self.window)
        params = {**arch["train"], "epochs": self.epochs, "patience": self.patience}
        t0 = time.time()
        history, _ = train(model, self.x_tr[..., cols], self.y_tr, self.x_val[..., cols], self.y_val, params,
                           self.device, rng, quiet=True)
        pred = self.norm.y_inv(predict(model, self.x_val[..., cols], self.device))
        ade = float(np.linalg.norm(pred - self.y_val_m, axis=-1).mean())
        latency = net_ms(model, len(cols), self.window, self.device)
        trial.set_user_attr("epochs", len(history["epochs"]))
        trial.set_user_attr("seconds", time.time() - t0)
        trial.set_user_attr("n_params", sum(p.numel() for p in model.parameters()))
        trial.set_user_attr("blocks", arch["blocks"])
        trial.set_user_attr("groups", arch["groups"])
        return ade, latency      # м и мс на обзор (батч 1)


def feature_importance(trials: list, n_models: int, top_share: float = 0.2, doppler: bool = False) -> list:
    """
    Важность групп признаков по попыткам поиска: средняя ADE попыток с группой и без
    неё и доля лучших попыток (top_share по ADE), где группа выбрана.
    """
    names = list(feature_groups(n_models, doppler))
    ade = np.array([t.values[0] for t in trials])
    top = ade <= np.quantile(ade, top_share) if len(ade) else np.zeros(0, bool)
    rows = []
    for i, name in enumerate(names):
        used = np.array([bool(t.params.get(f"use_{i}", True)) for t in trials])
        rows.append({"group": name,
                     "ade_with": float(ade[used].mean()) if used.any() else float("nan"),
                     "ade_without": float(ade[~used].mean()) if (~used).any() else float("nan"),
                     "share_in_top": float(used[top].mean()) if top.any() else float("nan")})
    return rows


def block_statistics(trials: list, top_share: float = 0.2) -> list:
    """Доля каждого типа блока среди всех блоков в лучших попытках и средняя ADE попыток с ним."""
    ade = np.array([t.values[0] for t in trials])
    top = ade <= np.quantile(ade, top_share) if len(ade) else np.zeros(0, bool)
    rows = []
    for kind in BLOCK_TYPES:
        has = np.array([kind in t.user_attrs.get("blocks", []) for t in trials])
        top_blocks = [b for t, is_top in zip(trials, top) if is_top for b in t.user_attrs.get("blocks", [])]
        rows.append({"block": kind,
                     "ade_with": float(ade[has].mean()) if has.any() else float("nan"),
                     "ade_without": float(ade[~has].mean()) if (~has).any() else float("nan"),
                     "share_in_top_blocks": top_blocks.count(kind) / max(len(top_blocks), 1)})
    return rows
