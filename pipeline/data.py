"""
Данные прогона для этапов версии 2: выборки этапа 0 плюс расширение реальных полётов train.

Обучение везде - реальные полёты train + их расширение (переразметка РЛС, отражение, растяжение по
скорости) + синтетика (решение пользователя 26.10.2026). Исключения - только в этапе 3, где долю
каждой части меняют намеренно (train_tracks(aug=..., synth_share=...)).

Смоук: подвыборка (settings.Settings) - реальные и синтетика train, их копии и наборы оценки.
"""

import pickle

import numpy as np

import config
from data import splits as splits_mod
from pipeline.paths import RunPaths
from pipeline.settings import Settings

AUG_KINDS = {"radar": "переразметка РЛС", "mirror": "отражение", "speed": "растяжение по скорости"}


def _every(tracks: list, step: int) -> list:
    return tracks[::step] if step > 1 else tracks


def load_sets(paths: RunPaths, st: Settings) -> dict:
    """
    {"train", "val", "test", ("holdout"), "synth", "aug": {вид: копии}, "_splits"}. val и test в merged
    содержат и сгенерированные траектории (data/splits.load_sets).
    """
    sets = splits_mod.load_sets(paths.stage0)
    with open(paths.stage0 / "tracks_aug.pkl", "rb") as fh:
        aug = pickle.load(fh)
    if st.smoke:
        sets["train"] = _every(sets["train"], st.real_step)
        sets["synth"] = _every(sets["synth"], st.synth_step)
        for name in config.EVAL_SETS:
            sets[name] = _every(sets[name], st.eval_step)
        kept = {tr.flight_id for tr in sets["train"]}
        aug = {kind: [c for c in cs if c.flight_id.split("~")[0] in kept] for kind, cs in aug.items()}
    sets["aug"] = aug
    return sets


def train_tracks(sets: dict, aug: bool = True, synth_share: float = 1.0, kinds=None) -> list:
    """Треки обучения: реальные train, при aug - их копии выбранных видов, затем доля synth_share синтетики."""
    out = list(sets["train"])
    if aug:
        for kind in (kinds or AUG_KINDS):
            out += sets["aug"][kind]
    n_synth = int(round(len(sets["synth"]) * synth_share))
    return out + sets["synth"][:n_synth]


def flight_key(track) -> str:
    """Единица пересэмплирования бутстрепа: лог целиком у реальных полётов, траектория у синтетики."""
    return track.log_file or track.flight_id


def describe(sets: dict) -> list:
    """Строки сводки о составе данных."""
    def hours(trs):
        return sum(float(tr.t[-1] - tr.t[0]) for tr in trs) / 3600.0

    aug_n = sum(len(v) for v in sets["aug"].values())
    lines = [f"реальные train: {len(sets['train'])} трасс, {hours(sets['train']):.1f} ч",
             f"расширение реальных: {aug_n} копий, {hours([c for v in sets['aug'].values() for c in v]):.1f} ч",
             f"синтетика train: {len(sets['synth'])} траекторий, {hours(sets['synth']):.1f} ч"]
    for name in config.EVAL_SETS:
        n_syn = sum(tr.source == "synth" for tr in sets[name])
        lines.append(f"{name}: {len(sets[name])} трасс (из них сгенерированных {n_syn}), {hours(sets[name]):.1f} ч")
    return lines
