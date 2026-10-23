"""
Разбиение реальных полётов на выборки - единственное место, где оно считается.

Единица разбиения - лог целиком (все куски одного полёта в одной выборке), иначе
соседние участки одного полёта попали бы и в обучение, и в тест.
  - тип HOLDOUT_AIRCRAFT целиком -> "holdout" (проверка обобщения на новый тип) - только в
    варианте данных base; в merged он делится вместе с остальными;
  - симулятор -> "sim" (в обучение не идёт, пока USE_SIM_IN_TRAIN = False);
  - типы вне USED_AIRCRAFT -> "excluded";
  - остальные логи каждого типа -> train / val / test в долях VAL_SHARE и TEST_SHARE.
В варианте base синтетика в разбиении не участвует: она только в train (и в отдельном
наборе анализа). В варианте merged этап 0 делит сгенерированные траектории на train / val /
test в тех же долях (split_synth), а load_sets добавляет их к реальным val и test.

Проверяется на этапе 0: выборки не пересекаются по логам, отложенного типа нет в
train / val / test.
"""

import json
import pickle
from pathlib import Path

import numpy as np

import config

SPLITS = ("train", "val", "test", "holdout", "sim", "excluded")


def make_splits(log_aircraft: dict, rng: np.random.Generator) -> dict:
    """log_aircraft: {имя лога: тип}. Возвращает {имя лога: выборка}."""
    result = {}
    by_type = {}
    for log_file in sorted(log_aircraft):
        aircraft = log_aircraft[log_file]
        if aircraft == config.HOLDOUT_AIRCRAFT and config.USE_HOLDOUT:
            result[log_file] = "holdout"
        elif aircraft in config.SIM_AIRCRAFT:
            result[log_file] = "train" if config.USE_SIM_IN_TRAIN else "sim"
        elif aircraft not in config.USED_AIRCRAFT:
            result[log_file] = "excluded"
        else:
            by_type.setdefault(aircraft, []).append(log_file)

    for aircraft in sorted(by_type):
        logs = list(by_type[aircraft])
        order = rng.permutation(len(logs))
        logs = [logs[i] for i in order]
        n = len(logs)
        n_test = max(1, round(config.TEST_SHARE * n)) if n >= 3 else 0
        n_val = max(1, round(config.VAL_SHARE * n)) if n >= 4 else 0
        for i, log_file in enumerate(logs):
            if i < n_test:
                result[log_file] = "test"
            elif i < n_test + n_val:
                result[log_file] = "val"
            else:
                result[log_file] = "train"
    return result


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_synth(tracks: list, rng: np.random.Generator) -> dict:
    """Вариант merged: сгенерированные траектории каждого сценария -> train / val / test в долях VAL_SHARE и TEST_SHARE."""
    out = {"train": [], "val": [], "test": []}
    for scenario in config.SYNTH_SCENARIOS:
        trs = [tr for tr in tracks if tr.scenario == scenario]
        order = rng.permutation(len(trs))
        n_test, n_val = round(config.TEST_SHARE * len(trs)), round(config.VAL_SHARE * len(trs))
        for i, j in enumerate(order):
            name = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out[name].append(trs[j])
    return out


def load_sets(cache0: Path) -> dict:
    """
    Выборки из кэша этапа 0: train / val / test (и holdout в варианте base) - реальные куски;
    "synth" - синтетика train; в варианте merged к val и test добавлены сгенерированные
    траектории (tracks_synth_eval.pkl). "_splits" - разбиение логов.
    """
    with open(cache0 / "tracks_real.pkl", "rb") as fh:
        real = pickle.load(fh)
    with open(cache0 / "tracks_synth_train.pkl", "rb") as fh:
        synth = pickle.load(fh)
    splits = load_splits(cache0 / "splits.json")
    names = ("train", "val", "test") + (("holdout",) if config.USE_HOLDOUT else ())
    sets = {name: [tr for tr in real if splits.get(tr.log_file) == name] for name in names}
    eval_path = cache0 / "tracks_synth_eval.pkl"
    if not config.USE_HOLDOUT:
        with open(eval_path, "rb") as fh:
            synth_eval = pickle.load(fh)
        for name in ("val", "test"):
            sets[name] = sets[name] + synth_eval[name]
    sets["synth"] = synth
    sets["_splits"] = splits
    return sets
