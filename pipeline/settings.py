"""
Настройки этапов версии 2, зависящие от устройства и режима (полный запуск / смоук).

Смоук на рабочем ПК проверяет конвейер на подвыборке и крошечных бюджетах: числа смоука для диплома
не годятся, его файлы лежат в отдельных папках _smoke (pipeline/paths.py).
"""

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Settings:
    smoke: bool
    epochs: int              # итоговое обучение: предел эпох
    patience: int
    seeds: tuple
    tune_epochs: int         # попытка подбора: предел эпох
    tune_patience: int
    imm_trials: int
    trials: int              # подбор корректоров (на модель, случай)
    arch_trials: int
    hyp_trials: int          # подбор для гипотез и сочетаний
    real_step: int           # смоук: каждый N-й реальный трек train / синтетический / расширенный / оценочный
    synth_step: int
    eval_step: int
    kf_epochs: int           # схемы с сетью внутри фильтра (Б2, Б3)
    kf_patience: int
    kf_lrs: tuple
    chunk: int
    alphas: tuple
    boot_n: int


FULL = Settings(smoke=False, epochs=80, patience=10, seeds=config.FINAL_SEEDS, tune_epochs=config.TUNE_EPOCHS,
                tune_patience=config.TUNE_PATIENCE, imm_trials=config.TUNE_IMM_TRIALS, trials=config.TUNE_TRIALS,
                arch_trials=config.ARCH_TRIALS, hyp_trials=config.HYP_TUNE_TRIALS, real_step=1, synth_step=1,
                eval_step=1, kf_epochs=40, kf_patience=6, kf_lrs=(1e-3, 3e-3, 1e-2), chunk=config.HYB_CHUNK_SCANS,
                alphas=config.HYB_B4_ALPHAS, boot_n=config.BOOTSTRAP_N)
SMOKE = Settings(smoke=True, epochs=2, patience=2, seeds=(42, 43), tune_epochs=2, tune_patience=2, imm_trials=3,
                 trials=3, arch_trials=3, hyp_trials=2, real_step=8, synth_step=25, eval_step=3, kf_epochs=2,
                 kf_patience=2, kf_lrs=(1e-3, 1e-2), chunk=30, alphas=(0.5, 2.0), boot_n=200)


def get(smoke: bool) -> Settings:
    return SMOKE if smoke else FULL
