"""
Случаи доплера версии 2 (решение пользователя 26-27.10.2026) и временная подмена config.

Три случая во всех этапах с моделями:
  nodop    - РЛС без доплера;
  dop_traj - доплер, истинная скорость по трём последним GPS-точкам трассы (этапы 12-17 версии 1);
  dop_tele - доплер, истинная скорость GPS-приёмника из лога (этап 18 версии 1).
У синтетики оба доплера одинаковы. Доверие фильтра к доплеру (FILTER_SIGMA_VR) подбирается в
каждом случае с доплером и передаётся в overrides.
"""

import contextlib

import config

CASES = ("nodop", "dop_traj", "dop_tele")
CASE_TITLES = {"nodop": "без доплера", "dop_traj": "доплер по траектории", "dop_tele": "доплер по телеметрии"}


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


def overrides(case: str, net_input: bool = True, sigma_vr: float = None) -> dict:
    """
    Значения config для случая. net_input - доплер во входе сети (в случаях с доплером его
    выбирает подбор; в IMM доплер входит всегда); sigma_vr - СКО радиальной скорости, которое
    закладывает фильтр (None - как у РЛС).
    """
    if case not in CASES:
        raise ValueError(f"Неизвестный случай {case!r}, ожидается один из {CASES}")
    if case == "nodop":
        return {"DOPPLER_IN_IMM": False, "DOPPLER_IN_NET": False, "DOPPLER_TRUTH": "positions",
                "FILTER_SIGMA_VR": None}
    return {"DOPPLER_IN_IMM": True, "DOPPLER_IN_NET": net_input,
            "DOPPLER_TRUTH": "positions" if case == "dop_traj" else "telemetry", "FILTER_SIGMA_VR": sigma_vr}


@contextlib.contextmanager
def use_case(case: str, net_input: bool = True, sigma_vr: float = None):
    """Включает случай на время блока (флаги доплера в config), потом возвращает прежние значения."""
    with config_override(overrides(case, net_input, sigma_vr)):
        yield


def stage_dir(stage: str, case: str, root=None):
    """Каталог результатов этапа для случая: results/<прогон>/<этап>/<случай>/ (создаётся)."""
    path = (root or config.RESULTS_DIR) / stage / case
    path.mkdir(parents=True, exist_ok=True)
    return path
