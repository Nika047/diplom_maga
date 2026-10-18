"""
Генератор маневрирующих траекторий класса самолётных БПЛА (истина, без шума).

Модель - точечная кинематика с ограничениями, общими для класса, без моделей
конкретных аппаратов:
  - скорость V в [v_min, v_max], продольное ускорение не больше A_MAX;
  - скорость разворота не больше min(g * tan(phi_max) / V, предел по логам)
    (координированный разворот), вход в крен - инерционное звено с постоянной TAU_ROLL;
  - скорость и |w| разворотов выбираются из распределений логов, а не равномерно;
  - вертикальная скорость не больше vz_max, инерционное звено TAU_VZ;
  - курс от севера по часовой: x += V sin(psi) dt, y += V cos(psi) dt.
Интегрирование с шагом SYNTH_DT_SIM, выдача с шагом SYNTH_DT_OUT. Время по
построению согласовано с длиной пути, скачков нет. Шум добавляет только модель РЛС.

Траектория задаётся планом - списком отрезков (длительность, команда разворота,
команда скорости, команда вертикальной скорости). Сценарии собирают планы.
Пределы калибруются по реальным логам выборки train (calibrate_limits).

Проверяется на этапе 0: скорость, |w| и |vz| в пределах, нет скачков, путь
согласован со временем, повторный запуск с тем же seed даёт тот же результат.
"""

from dataclasses import dataclass, asdict

import numpy as np

import config
from data.track import Track


@dataclass
class Limits:
    v_min: float
    v_max: float
    phi_max_deg: float
    vz_max: float
    omega_cap_deg: float     # верхний предел скорости разворота по логам, град/с
    speed_q: list            # квантильная функция скорости (равномерная сетка вероятностей)
    omega_q: list            # квантильная функция |w| на маневрах
    turn_pairs: list         # пары (скорость, |w|) точек маневра логов, по возрастанию |w|
    # Измеренная по 2-секундным отметкам |w| ниже заданной (инерция крена, сглаживание
    # курса). Множитель подбирается так, чтобы совпали именно измеренные распределения.
    omega_scale: float = 1.0

    def omega_max_deg(self, v):
        """Предельная заданная скорость разворота (град/с) при скорости v: крен и предел по логам."""
        phys = np.degrees(config.G * np.tan(np.radians(self.phi_max_deg)) / np.maximum(v, 1.0))
        return np.minimum(phys, self.omega_cap_deg * self.omega_scale)

    def sample_speed(self, rng: np.random.Generator) -> float:
        grid = np.linspace(0.0, 1.0, len(self.speed_q))
        return float(np.interp(rng.uniform(), grid, self.speed_q))

    def sample_turn(self, rng: np.random.Generator, u_range=(0.0, 1.0)) -> tuple:
        """
        Пара (скорость, |w|) реальной точки маневра; u_range - диапазон по рангу |w|.

        Скорость и |w| в логах связаны (резкие развороты - на малой скорости), поэтому
        выбираются вместе, а не независимо.
        """
        n = len(self.turn_pairs)
        i = min(int(rng.uniform(*u_range) * n), n - 1)
        v, w = self.turn_pairs[i]
        v = float(np.clip(v, self.v_min, self.v_max))
        return v, float(min(w * self.omega_scale, self.omega_max_deg(v)))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    duration: float      # с
    omega_cmd: float     # град/с, со знаком (+ вправо)
    v_cmd: float         # м/с
    vz_cmd: float = 0.0  # м/с


def calibrate_limits(tracks: list) -> tuple:
    """
    Пределы и распределения по реальным трекам с разметкой. Возвращает (Limits, сведения).

    Скорость и |w| на маневрах берутся квантильными функциями (между нижним и верхним
    квантилем калибровки), поэтому синтетика повторяет распределения логов, а не
    равномерное.
    """
    speeds, phis, omegas, turn_v, vzs = [], [], [], [], []
    for tr in tracks:
        moving = tr.speed >= config.PHASE_V_MIN
        speeds.append(tr.speed[moving])
        turning = moving & ~np.isnan(tr.omega) & (np.abs(np.nan_to_num(tr.omega)) >= config.OMEGA_TURN_ON)
        w = np.abs(tr.omega[turning])
        omegas.append(w)
        turn_v.append(tr.speed[turning])
        phis.append(np.degrees(np.arctan(np.radians(w) * tr.speed[turning] / config.G)))
        if len(tr.t) >= 2:
            vzs.append(np.abs(np.diff(tr.pos[:, 2]) / np.diff(tr.t)))
    speeds, phis, omegas, turn_v, vzs = (np.concatenate(a) for a in (speeds, phis, omegas, turn_v, vzs))
    q_lo, q_hi = config.CALIB_V_QUANTILES
    grid = np.linspace(q_lo, q_hi, config.CALIB_GRID_POINTS)
    cap = float(np.quantile(omegas, config.CALIB_OMEGA_CAP_QUANTILE))
    order = np.argsort(omegas, kind="stable")
    pairs = [[float(turn_v[i]), float(omegas[i])] for i in order if omegas[i] <= cap]
    limits = Limits(
        v_min=float(np.quantile(speeds, q_lo)),
        v_max=float(np.quantile(speeds, q_hi)),
        phi_max_deg=float(np.clip(np.quantile(phis, config.CALIB_PHI_QUANTILE), 15.0, 60.0)),
        vz_max=float(np.clip(np.quantile(vzs, config.CALIB_VZ_QUANTILE), 1.0, 15.0)),
        omega_cap_deg=cap,
        speed_q=[float(x) for x in np.quantile(speeds, grid)],
        omega_q=[float(x) for x in np.quantile(omegas, grid)],
        turn_pairs=pairs,
    )
    info = {"points_speed": int(len(speeds)), "points_turn": int(len(omegas)),
            "points_vz": int(len(vzs))}
    return limits, info


def simulate(plan: list, limits: Limits, v0: float, psi0_deg: float, z0: float) -> tuple:
    """Интегрирует план. Возвращает (t, pos) с шагом SYNTH_DT_OUT."""
    dt = config.SYNTH_DT_SIM
    out_every = int(round(config.SYNTH_DT_OUT / dt))
    x = y = 0.0
    z = z0
    v = float(np.clip(v0, limits.v_min, limits.v_max))
    psi = np.radians(psi0_deg)
    omega = 0.0  # рад/с
    vz = 0.0
    ts, ps = [0.0], [(x, y, z)]
    step = 0
    for seg in plan:
        for _ in range(int(round(seg.duration / dt))):
            w_lim = np.radians(float(limits.omega_max_deg(v)))
            w_cmd = float(np.clip(np.radians(seg.omega_cmd), -w_lim, w_lim))
            omega += (w_cmd - omega) * dt / config.TAU_ROLL
            omega = float(np.clip(omega, -w_lim, w_lim))
            v_target = float(np.clip(seg.v_cmd, limits.v_min, limits.v_max))
            v += float(np.clip(v_target - v, -config.A_MAX * dt, config.A_MAX * dt))
            vz_cmd = seg.vz_cmd if (z > 30.0 or seg.vz_cmd > 0) else 0.0
            vz += (np.clip(vz_cmd, -limits.vz_max, limits.vz_max) - vz) * dt / config.TAU_VZ
            psi += omega * dt
            x += v * np.sin(psi) * dt
            y += v * np.cos(psi) * dt
            z += vz * dt
            step += 1
            if step % out_every == 0:
                ts.append(step * dt)
                ps.append((x, y, z))
    return np.array(ts), np.array(ps)


# ============================================================================
# Сценарии
# ============================================================================

def _straight(rng, v):
    return Segment(rng.uniform(*config.SYNTH_STRAIGHT_S), 0.0, v)


def _turn(rng, limits, angle_deg=None, u_range=(0.0, 1.0), vz=0.0):
    """Разворот на угол angle_deg; скорость и |w| - пара из маневров логов (u_range - по рангу |w|)."""
    angle = angle_deg if angle_deg is not None else rng.uniform(*config.SYNTH_TURN_ANGLE_DEG)
    v, rate = limits.sample_turn(rng, u_range)
    sign = rng.choice([-1.0, 1.0])
    return Segment(angle / rate, sign * rate, v, vz)


def make_plan(scenario: str, rng: np.random.Generator, limits: Limits, v: float) -> list:
    if scenario == "straight":
        return [_straight(rng, v), Segment(rng.uniform(20, 60), 0.0, limits.sample_speed(rng)),
                _straight(rng, v)]
    if scenario == "turn":
        return [_straight(rng, v), _turn(rng, limits), _straight(rng, v)]
    if scenario == "zigzag":
        plan = [_straight(rng, v)]
        sign = rng.choice([-1.0, 1.0])
        for _ in range(int(rng.integers(3, 7))):
            seg = _turn(rng, limits, angle_deg=rng.uniform(30, 90))
            seg.omega_cmd = sign * abs(seg.omega_cmd)
            sign = -sign
            plan += [seg, Segment(rng.uniform(4, 20), 0.0, v)]
        return plan + [_straight(rng, v)]
    if scenario == "loiter":
        return [_straight(rng, v),
                _turn(rng, limits, angle_deg=360.0 * rng.uniform(1.0, 3.0), u_range=(0.0, 0.7)),
                _straight(rng, v)]
    if scenario == "u_turn":
        return [_straight(rng, v), _turn(rng, limits, angle_deg=180.0, u_range=(0.5, 1.0)),
                _straight(rng, v)]
    if scenario == "climb_turn":
        vz = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * limits.vz_max
        return [_straight(rng, v), _turn(rng, limits, vz=vz), Segment(rng.uniform(10, 40), 0.0, v, vz),
                _straight(rng, v)]
    if scenario == "mixed":
        plan = [_straight(rng, v)]
        for _ in range(int(rng.integers(*config.SYNTH_MIXED_SEGMENTS))):
            v_seg = limits.sample_speed(rng)
            kind = rng.choice(["straight", "turn", "turn", "climb"])
            if kind == "straight":
                plan.append(Segment(rng.uniform(5, 60), 0.0, v_seg))
            elif kind == "turn":
                plan.append(_turn(rng, limits))
            else:
                vz = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * limits.vz_max
                plan.append(Segment(rng.uniform(10, 40), 0.0, v_seg, vz))
        return plan + [_straight(rng, v)]
    raise ValueError(f"Неизвестный сценарий: {scenario}")


def generate(counts: dict, limits: Limits, rng: np.random.Generator, prefix: str) -> list:
    """Генерирует траектории по сценариям в порядке config.SYNTH_SCENARIOS."""
    tracks = []
    for scenario in config.SYNTH_SCENARIOS:
        for k in range(counts.get(scenario, 0)):
            v = limits.sample_speed(rng)
            psi0 = rng.uniform(0.0, 360.0)
            z0 = rng.uniform(*config.SYNTH_ALT_RANGE)
            plan = make_plan(scenario, rng, limits, v)
            t, pos = simulate(plan, limits, v, psi0, z0)
            pos = pos - pos[0]
            tracks.append(Track(flight_id=f"{prefix}_{scenario}_{k:04d}", source="synth",
                                aircraft="synth", t=t, pos=pos, scenario=scenario))
    return tracks
