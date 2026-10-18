"""
Генетический подбор на Optuna (сэмплер NSGA-II - генетический алгоритм: популяция,
скрещивание, мутация, отбор).

  - параметры IMM: шумы моделей CV и CT и вероятность остаться в модели; цель - ADE
    прогноза IMM по горизонтали на реальных трассах train;
  - гиперпараметры корректора: ширина, число слоёв, dropout, lr, батч, weight decay,
    длина окна (из TUNE_WINDOWS); цель - ADE на val (м).

Первая попытка каждого подбора - исходные параметры (enqueue), поэтому лучшее
найденное не хуже исходного. Попытки хранятся в SQLite (cache/stage4/optuna.db):
прерванный подбор продолжается с места остановки, бюджет считается по завершённым
попыткам.

Проверяется на этапе 4: бюджет одинаков для всех моделей, лучшее не хуже исходного.
"""

import contextlib
import time
from pathlib import Path

import numpy as np
import optuna
import torch

import config
from common.logging_setup import get_logger
from common.rng import make_rng
from data.phases import PHASE_CODE
from eval import metrics
from filters.imm import IMM
from models import build_corrector
from training import dataset
from training.trainer import predict, train

IMM_KEYS = tuple(config.TUNE_IMM_SPACE) + ("IMM_P_STAY",)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@contextlib.contextmanager
def config_override(values: dict):
    """Временная подмена значений config."""
    old = {key: getattr(config, key) for key in values}
    for key, val in values.items():
        setattr(config, key, val)
    try:
        yield
    finally:
        for key, val in old.items():
            setattr(config, key, val)


def default_imm_params() -> dict:
    return {key: getattr(config, key) for key in IMM_KEYS}


def default_corrector_params() -> dict:
    """Параметры этапа 3 - первая попытка подбора."""
    window = config.WINDOW_SCANS if config.WINDOW_SCANS in config.TUNE_WINDOWS else config.TUNE_WINDOWS[0]
    return {"window": window, "hidden": config.CORRECTOR_HIDDEN, "layers": config.CORRECTOR_LAYERS,
            "dropout": config.CORRECTOR_DROPOUT, "batch": 512, "lr": 1e-3, "weight_decay": 1e-4}


def _sampler(seed: int, population: int = None) -> optuna.samplers.BaseSampler:
    return optuna.samplers.NSGAIISampler(population_size=population or config.TUNE_POPULATION, seed=seed)


def open_study(name: str, storage_path: Path, seed: int = config.RANDOM_SEED, n_objectives: int = 1,
               population: int = None) -> optuna.Study:
    """
    Подбор с хранением в SQLite. При продолжении сэмплер получает seed + число уже
    сделанных попыток: с тем же seed он заново предложил бы тех же кандидатов, что в
    начале. Подбор без перерыва воспроизводим; продолженный зависит от места остановки.
    n_objectives > 1 - многокритериальный подбор (все цели минимизируются).
    """
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=name, storage=f"sqlite:///{storage_path.as_posix()}",
                                directions=["minimize"] * n_objectives, sampler=_sampler(seed, population),
                                load_if_exists=True)
    if study.trials:
        study.sampler = _sampler(seed + len(study.trials), population)
    return study


def completed(study: optuna.Study) -> list:
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def default_trial(study: optuna.Study):
    """Завершённая попытка с исходными параметрами (None, если её ещё нет)."""
    return next((t for t in completed(study) if t.user_attrs.get("default")), None)


def run_budget(study: optuna.Study, objective, n_trials: int, first_params: dict) -> None:
    """Доводит число завершённых попыток до n_trials; одна из них - исходные параметры first_params."""
    log = get_logger()
    # Попытки, оборванные остановкой процесса, остаются в базе в состоянии RUNNING:
    # помечаются как FAIL, чтобы не мешать продолжению и не считаться в бюджете.
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study._storage.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)
        log.warning("Подбор %s: попытка %d была прервана, помечена как неудачная", study.study_name, t.number)
    waiting = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    if default_trial(study) is None and not waiting:
        study.enqueue_trial(first_params, user_attrs={"default": True})
    remaining = n_trials - len(completed(study))
    if remaining <= 0:
        log.info("Подбор %s: бюджет %d уже выполнен", study.study_name, n_trials)
        return

    def _log(st, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            params = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in trial.params.items()}
            if len(st.directions) == 1:
                log.info("  %s: попытка %d - %.2f м (лучшее %.2f), %s", st.study_name, trial.number, trial.value,
                         st.best_value, params)
            else:
                log.info("  %s: попытка %d - %s, %s", st.study_name, trial.number,
                         [round(v, 3) for v in trial.values], params)
        else:
            log.warning("  %s: попытка %d - %s", st.study_name, trial.number, trial.state.name)

    study.optimize(objective, n_trials=remaining, callbacks=[_log], catch=(RuntimeError,))


# ============================================================================
# Шаг 1: IMM
# ============================================================================

def imm_objective(rts: list):
    def objective(trial: optuna.Trial) -> float:
        values = {key: trial.suggest_float(key, lo, hi, log=True) for key, (lo, hi) in config.TUNE_IMM_SPACE.items()}
        values["IMM_P_STAY"] = trial.suggest_float("IMM_P_STAY", *config.TUNE_IMM_P_STAY)
        with config_override(values):
            imm = IMM()
            pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, imm.run(rt).forecast) for rt in rts])
        s = metrics.summarize(pairs)
        turn = metrics.summarize(metrics.select(pairs, np.isin(pairs["phase"], (PHASE_CODE["entry"], PHASE_CODE["turn"]))))
        trial.set_user_attr("maneuver_ade", turn["ade"])
        return s["ade"]
    return objective


# ============================================================================
# Шаг 2: корректоры
# ============================================================================

class CorrectorObjective:
    """Цель подбора одного корректора; нормированные окна кэшируются по (окно, режим)."""

    def __init__(self, corrector: str, mode: str, windows: dict, device, seed: int = config.RANDOM_SEED,
                 epochs: int = config.TUNE_EPOCHS, patience: int = config.TUNE_PATIENCE):
        self.corrector, self.mode, self.windows = corrector, mode, windows
        self.device, self.seed, self.epochs, self.patience = device, seed, epochs, patience
        self._cache = {}

    def _data(self, window: int):
        if window not in self._cache:
            ws_tr, ws_val = self.windows[window]
            tr, va = ws_tr.view(self.mode), ws_val.view(self.mode)
            norm = dataset.Normalizer.fit(tr)
            self._cache[window] = (norm, norm.x(tr.x), norm.y(tr.y), norm.x(va.x), norm.y(va.y), va.y)
        return self._cache[window]

    def __call__(self, trial: optuna.Trial) -> float:
        window = trial.suggest_categorical("window", list(config.TUNE_WINDOWS))
        hp = {"hidden": trial.suggest_categorical("hidden", list(config.TUNE_HIDDEN)),
              "layers": trial.suggest_categorical("layers", list(config.TUNE_LAYERS)),
              "dropout": trial.suggest_float("dropout", *config.TUNE_DROPOUT)}
        params = {"batch": trial.suggest_categorical("batch", list(config.TUNE_BATCH)),
                  "lr": trial.suggest_float("lr", *config.TUNE_LR, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", *config.TUNE_WEIGHT_DECAY, log=True),
                  "epochs": self.epochs, "patience": self.patience}
        norm, x_tr, y_tr, x_val, y_val, y_val_m = self._data(window)
        rng = make_rng("training", self.seed)
        torch.manual_seed(self.seed)
        model = build_corrector(self.corrector, x_tr.shape[-1], config.HORIZON_SCANS, window, **hp)
        t0 = time.time()
        history, _ = train(model, x_tr, y_tr, x_val, y_val, params, self.device, rng, quiet=True)
        pred = norm.y_inv(predict(model, x_val, self.device))
        trial.set_user_attr("epochs", len(history["epochs"]))
        trial.set_user_attr("seconds", time.time() - t0)
        trial.set_user_attr("n_params", sum(p.numel() for p in model.parameters()))
        return float(np.linalg.norm(pred - y_val_m, axis=-1).mean())


def split_params(params: dict) -> tuple:
    """Параметры попытки -> (окно, гиперпараметры сети, параметры обучения)."""
    hp = {key: params[key] for key in ("hidden", "layers", "dropout")}
    train_params = {key: params[key] for key in ("batch", "lr", "weight_decay")}
    return params["window"], hp, train_params
