"""
Расширение реальных полётов для обучения (этап 9): новые трассы из тех же истинных траекторий.

  radar_copy    - та же истина под новым именем: у трассы свои положение РЛС и шум отметок
                  (radar/model.observe тянет их из потока по flight_id);
  mirror        - зеркальное отражение по оси север - юг (x -> -x): развороты меняют
                  направление, скорость и |w| те же;
  speed_scale   - растяжение по скорости: горизонтальные координаты умножаются на f при том же
                  времени. Скорость растёт в f раз, скорость разворота та же, радиус и нужный
                  крен (tg phi = V w / g) - в f раз. Высота не меняется.

Каждая копия - новая трасса со своим именем ("<имя>~<вид>"), источником "real" и тем же
log_file: расширяется только train, разбиение по логам не затрагивается. Тип аппарата и
режим полёта во входы не идут и здесь только переносятся.

Растяжение допускается, только если трасса остаётся в пределах калибровки генератора
(speed_scale_allowed): квантили скорости внутри [v_min, v_max], нужный крен на маневре - не
больше phi_max.

Проверяется на этапе 9 на подставных траекториях: отражение сохраняет скорость и |w| и
меняет знак w, растяжение умножает скорость на f и сохраняет w; копия РЛС даёт ту же
истину и другие отметки.
"""

from dataclasses import replace

import numpy as np

import config
from data import phases
from data.generator import Limits
from data.track import Track


def _copy(track: Track, suffix: str, pos: np.ndarray, vel_scale: tuple = (1.0, 1.0, 1.0)) -> Track:
    """vel_scale - множители скорости по телеметрии (vel_log) по осям, как у координат."""
    vel = None if track.vel_log is None else track.vel_log * np.asarray(vel_scale)
    new = replace(track, flight_id=f"{track.flight_id}~{suffix}", pos=pos, lat=None, lon=None,
                  speed=None, omega=None, phase=None, turn_onsets=[], vel_log=vel)
    return phases.annotate(new)


def radar_copy(track: Track, j: int) -> Track:
    return _copy(track, f"radar{j}", track.pos.copy())


def mirror(track: Track) -> Track:
    pos = track.pos.copy()
    pos[:, 0] = -pos[:, 0]
    return _copy(track, "mirror", pos, (-1.0, 1.0, 1.0))


def speed_scale(track: Track, f: float) -> Track:
    pos = track.pos.copy()
    pos[:, :2] *= f
    return _copy(track, f"speed{f:g}", pos, (f, f, 1.0))


def speed_scale_allowed(track: Track, f: float, limits: Limits) -> bool:
    """Растянутая в f раз трасса в пределах калибровки: скорость и крен (по квантилям, как калибровка)."""
    v = track.speed * f
    lo, hi = np.quantile(v, config.CALIB_V_QUANTILES)
    if lo < limits.v_min or hi > limits.v_max:
        return False
    turn = np.isin(track.phase, (phases.PHASE_CODE["entry"], phases.PHASE_CODE["turn"])) & np.isfinite(track.omega)
    if not turn.any():
        return True
    bank = v[turn] * np.radians(np.abs(track.omega[turn])) / config.G
    return float(np.quantile(bank, config.CALIB_PHI_QUANTILE)) <= np.tan(np.radians(limits.phi_max_deg))


def augment(tracks: list, kind: str, limits: Limits) -> tuple:
    """
    Копии треков одного вида kind ("radar", "mirror", "speed"). Возвращает (копии, число
    пропущенных растяжений вне пределов).
    """
    out, skipped = [], 0
    for tr in tracks:
        if kind == "radar":
            out += [radar_copy(tr, j + 1) for j in range(config.AUG_RADAR_COPIES)]
        elif kind == "mirror":
            out.append(mirror(tr))
        elif kind == "speed":
            for f in config.AUG_SPEED_FACTORS:
                if speed_scale_allowed(tr, f, limits):
                    out.append(speed_scale(tr, f))
                else:
                    skipped += 1
        else:
            raise ValueError(kind)
    return out, skipped
