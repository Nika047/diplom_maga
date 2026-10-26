"""
Траектория (истина) - единый формат для реальных полётов и синтетики.

Координаты ENU в метрах (x - восток, y - север, z - вверх) с началом в первой
точке траектории. Время - секунды от начала траектории.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Track:
    flight_id: str          # "<имя файла>#<номер куска>" или "synth_<сценарий>_<номер>"
    source: str             # "real" | "sim" | "synth"
    aircraft: str           # тип аппарата (для синтетики - "synth")
    t: np.ndarray           # (N,) с
    pos: np.ndarray         # (N, 3) м, ENU
    log_file: str = ""      # исходный файл (для real / sim), единица разбиения
    firmware: str = ""      # "inav" | "ardupilot" | "unknown" | "" для синтетики
    scenario: str = ""      # сценарий генератора (для синтетики)
    lat: Optional[np.ndarray] = None   # (N,) градусы, только для логов - для карты
    lon: Optional[np.ndarray] = None
    flight_mode: Optional[list] = None  # коды режима по точкам, только для анализа
    # (N, 3) м/с ENU по телеметрии GPS (GSpd, Hdg, VSpd), только логи; нужна лишь для доплера по
    # телеметрии (этап 18, data/flight_logs.attach_log_velocity), во входы не идёт
    vel_log: Optional[np.ndarray] = None
    # Заполняются разметкой участков (data/phases.py)
    speed: Optional[np.ndarray] = None   # (N,) м/с, горизонтальная
    omega: Optional[np.ndarray] = None   # (N,) град/с, скорость разворота (nan - не определена)
    phase: Optional[np.ndarray] = None   # (N,) int8, индекс в config.PHASES
    turn_onsets: list = field(default_factory=list)  # индексы начала маневров

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def path_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.pos[:, :2], axis=0), axis=1).sum())

    def __len__(self) -> int:
        return len(self.t)
