"""
Двухцелевой генетический подбор корректора (этап 2 версии 2): цели - ADE на val и время отклика сети
(мс на один обзор, батч 1). NSGA-II (Optuna) строит фронт Парето «точность - скорость».

Геном: окно (TUNE_WINDOWS), ширина, число слоёв, dropout, батч, lr, weight decay; в случаях с доплером ещё
net_input - подавать ли в сеть доплерные признаки (в IMM доплер входит всегда). Первая попытка - параметры
основы (default_params): найденное не хуже исходного.
"""

import time

import numpy as np
import optuna
import torch

import config
from common.rng import make_rng
from eval.latency import net_ms
from models import build_corrector
from training import dataset
from training.trainer import predict, train
from tuning import search


def default_params(allow_net_input: bool) -> dict:
    out = dict(search.default_corrector_params())
    if allow_net_input:
        out["net_input"] = True
    return out


class CorrectorObjective:
    """data: {(окно, net_input): (окна train, окна val)}; нормированные массивы кэшируются по ключу."""

    def __init__(self, corrector: str, mode: str, data: dict, device, seed: int = config.RANDOM_SEED,
                 epochs: int = config.TUNE_EPOCHS, patience: int = config.TUNE_PATIENCE):
        self.corrector, self.mode, self.data = corrector, mode, data
        self.device, self.seed, self.epochs, self.patience = device, seed, epochs, patience
        self.allow_net_input = mode == "hybrid" and any(not net for (_, net) in data)
        self._cache = {}

    def _prepared(self, key):
        if key not in self._cache:
            ws_tr, ws_val = self.data[key]
            tr, va = ws_tr.view(self.mode), ws_val.view(self.mode)
            norm = dataset.Normalizer.fit(tr)
            self._cache[key] = (norm, norm.x(tr.x), norm.y(tr.y), norm.x(va.x), norm.y(va.y), va.y)
        return self._cache[key]

    def __call__(self, trial: optuna.Trial):
        window = trial.suggest_categorical("window", list(config.TUNE_WINDOWS))
        net_input = trial.suggest_categorical("net_input", [True, False]) if self.allow_net_input else True
        hp = {"hidden": trial.suggest_categorical("hidden", list(config.TUNE_HIDDEN)),
              "layers": trial.suggest_categorical("layers", list(config.TUNE_LAYERS)),
              "dropout": trial.suggest_float("dropout", *config.TUNE_DROPOUT)}
        params = {"batch": trial.suggest_categorical("batch", list(config.TUNE_BATCH)),
                  "lr": trial.suggest_float("lr", *config.TUNE_LR, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", *config.TUNE_WEIGHT_DECAY, log=True),
                  "epochs": self.epochs, "patience": self.patience}
        key = (window, net_input if self.mode == "hybrid" else True)
        norm, x_tr, y_tr, x_val, y_val, y_val_m = self._prepared(key)
        rng = make_rng("training", self.seed)
        torch.manual_seed(self.seed)
        model = build_corrector(self.corrector, x_tr.shape[-1], config.HORIZON_SCANS, window, **hp)
        t0 = time.time()
        history, _ = train(model, x_tr, y_tr, x_val, y_val, params, self.device, rng, quiet=True)
        pred = norm.y_inv(predict(model, x_val, self.device))
        trial.set_user_attr("epochs", len(history["epochs"]))
        trial.set_user_attr("seconds", time.time() - t0)
        trial.set_user_attr("n_params", int(sum(p.numel() for p in model.parameters())))
        ms = net_ms(model, x_tr.shape[-1], window, self.device)
        return float(np.linalg.norm(pred - y_val_m, axis=-1).mean()), ms


def pick(trials: list, near: float = config.ARCH_NEAR_BEST) -> tuple:
    """(лучшая по ADE, самая быстрая из не хуже лучшей больше чем на near) среди завершённых попыток."""
    best = min(trials, key=lambda t: t.values[0])
    close = [t for t in trials if t.values[0] <= best.values[0] * (1.0 + near)]
    return best, min(close, key=lambda t: t.values[1])


def split(params: dict) -> tuple:
    """Параметры попытки -> (окно, доплер во входе сети, гиперпараметры сети, параметры обучения)."""
    window, hp, train_params = search.split_params(params)
    return window, bool(params.get("net_input", True)), hp, train_params
