"""
Проверки данных этапа 0 версии 2: расширение реальных полётов и скорость по телеметрии.

  check_augment   - известный ответ расширений на подставной трассе (отражение сохраняет скорость
                    и |w| и меняет знак w, растяжение умножает скорость на f при той же w, копия
                    РЛС - та же истина и другие отметки);
  check_velocity  - скорость GPS-приёмника из лога (Track.vel_log) согласована со скоростью по
                    координатам (медианы), для доплера по телеметрии;
  check_doppler   - известный ответ радиальной скорости, СКО шума доплера, отметки положения от
                    доплера не меняются, заданная скорость даёт ту же формулу.
"""

import numpy as np

import config
from data import augment, phases
from data.track import Track
from radar.model import observe, true_radial_velocity


def probe_track() -> Track:
    """Подставная трасса: 60 с прямо на север, разворот 10 град/с вправо 90 с, 60 с прямо; 20 м/с."""
    t = np.arange(0.0, 212.0, 2.0)
    w = np.where((t >= 60.0) & (t < 150.0), np.radians(10.0), 0.0)
    heading = np.concatenate([[0.0], np.cumsum(w[:-1] * 2.0)])
    v = 20.0 * np.stack([np.sin(heading), np.cos(heading)], axis=1)
    xy = np.vstack([[0.0, 0.0], np.cumsum(v[:-1] * 2.0, axis=0)])
    pos = np.column_stack([xy, np.full(len(t), 100.0)])
    return phases.annotate(Track("aug_test#0", "real", "test", t, pos, log_file="aug_test.csv"))


def check_augment(summary) -> None:
    tr = probe_track()
    m, s, r = augment.mirror(tr), augment.speed_scale(tr, 1.15), augment.radar_copy(tr, 1)
    ok_w = np.isfinite(tr.omega)
    d_mv, d_mw = np.abs(m.speed - tr.speed).max(), np.abs(m.omega[ok_w] + tr.omega[ok_w]).max()
    summary.check("Отражение: скорость та же, w меняет знак, участки те же",
                  d_mv < 1e-9 and d_mw < 1e-9 and np.array_equal(m.phase, tr.phase) and tr.omega[ok_w].max() > 9.0,
                  f"|dV| {d_mv:.1e} м/с, |w + w'| {d_mw:.1e} град/с, |w| на развороте {tr.omega[ok_w].max():.1f} град/с")
    d_sv, d_sw = np.abs(s.speed / tr.speed - 1.15).max(), np.abs(s.omega[ok_w] - tr.omega[ok_w]).max()
    summary.check("Растяжение x1,15: скорость в 1,15 раза, w та же", d_sv < 1e-9 and d_sw < 1e-9,
                  f"|V'/V - 1,15| {d_sv:.1e}, |w' - w| {d_sw:.1e} град/с")
    a, b = observe(tr), observe(r)
    same = np.abs(a.truth - b.truth).max()
    summary.check("Копия РЛС: та же истина, другие РЛС и отметки",
                  same < 1e-9 and np.linalg.norm(a.radar_pos - b.radar_pos) > 1.0 and np.abs(a.meas - b.meas).max() > 1.0,
                  f"|истина - истина'| {same:.1e} м, РЛС разнесены на {np.linalg.norm(a.radar_pos - b.radar_pos):.0f} м")


def velocity_agreement(tracks: list) -> dict:
    """По типам аппаратов: отношение модулей горизонтальной скорости (телеметрия / координаты), отличие
    курса, °, и модуль разности горизонтальной скорости, м/с - на точках со скоростью больше 8 м/с."""
    rows = {}
    for tr in tracks:
        if tr.vel_log is None or len(tr.t) < 5:
            continue
        v = (tr.pos[2:] - tr.pos[:-2]) / (tr.t[2:] - tr.t[:-2])[:, None]
        vl = tr.vel_log[1:-1]
        ok = np.isfinite(vl).all(axis=1) & (np.hypot(v[:, 0], v[:, 1]) > 8.0)
        if not ok.any():
            continue
        r = rows.setdefault(tr.aircraft, {"ratio": [], "course": [], "dv": []})
        r["ratio"].append(np.hypot(vl[ok, 0], vl[ok, 1]) / np.hypot(v[ok, 0], v[ok, 1]))
        d = np.arctan2(vl[ok, 0], vl[ok, 1]) - np.arctan2(v[ok, 0], v[ok, 1])
        r["course"].append(np.degrees(np.abs((d + np.pi) % (2 * np.pi) - np.pi)))
        r["dv"].append(np.linalg.norm(vl[ok, :2] - v[ok, :2], axis=1))
    return {a: {k: np.concatenate(v) for k, v in r.items()} for a, r in rows.items()}


def check_velocity(summary, real: list) -> None:
    """Скорость по телеметрии есть у всех реальных треков и согласована с координатами (медианы)."""
    have = sum(tr.vel_log is not None for tr in real)
    summary.check("Скорость по телеметрии есть у всех реальных треков", have == len(real),
                  f"{have} из {len(real)}")
    agree = velocity_agreement(real)
    if not agree:
        summary.check("Скорость по телеметрии согласована со скоростью по координатам", False, "нет данных")
        return
    cat = {k: np.concatenate([a[k] for a in agree.values()]) for k in ("ratio", "course", "dv")}
    summary.check("Скорость по телеметрии согласована со скоростью по координатам (медианы)",
                  abs(np.median(cat["ratio"]) - 1.0) <= 0.05 and np.median(cat["course"]) <= 5.0,
                  f"отношение модулей {np.median(cat['ratio']):.3f}, отличие курса {np.median(cat['course']):.1f}°, "
                  f"точек {len(cat['ratio'])}")
    summary.section("Скорость по телеметрии против скорости по координатам (центральная разность), точки быстрее "
                    "8 м/с, медианы", [
                        f"{a:12s} отношение {np.median(r['ratio']):.3f}, курс {np.median(r['course']):4.1f}°, "
                        f"|разность| {np.median(r['dv']):.2f} м/с (90% - {np.quantile(r['dv'], 0.9):.2f}), "
                        f"точек {len(r['ratio'])}" for a, r in sorted(agree.items())])


def check_doppler(summary, tracks: list) -> None:
    """Известный ответ и согласованность доплера в модели РЛС (radar/model.py)."""
    from common.cases import config_override

    t = np.arange(30) * config.SCAN_PERIOD
    radial = np.column_stack([1000.0 + 20.0 * t, np.zeros_like(t), np.zeros_like(t)])
    across = np.column_stack([-300.0 + 20.0 * t, np.full_like(t, 1000.0), np.zeros_like(t)])
    exact = 20.0 * across[:, 0] / np.linalg.norm(across, axis=1)
    e1 = float(np.abs(true_radial_velocity(radial, np.zeros(3))[2:] - 20.0).max())
    e2 = float(np.abs(true_radial_velocity(across, np.zeros(3))[2:] - exact[2:]).max())
    summary.check("Доплер: известный ответ (движение от РЛС и мимо неё)", e1 < 1e-9 and e2 < 1e-9,
                  f"от РЛС 20 м/с: ошибка {e1:.1e}; мимо РЛС: ошибка {e2:.1e} м/с")

    vel = np.tile([20.0, 0.0, 0.0], (len(t), 1))
    truth = np.column_stack([-300.0 + 20.0 * t, np.full_like(t, 1000.0), 5.0 + 0.0 * t])
    a, b = true_radial_velocity(truth, np.zeros(3)), true_radial_velocity(truth, np.zeros(3), vel=vel)
    summary.check("Доплер по заданной скорости: известный ответ (прямолинейное движение мимо РЛС)",
                  float(np.abs(a[2:] - b[2:]).max()) < 1e-9 and np.isfinite(b[0]),
                  f"наибольшее отличие от формулы по траектории {np.abs(a[2:] - b[2:]).max():.1e} м/с")

    sample = tracks[:4]
    with config_override({"DOPPLER_IN_IMM": True, "DOPPLER_IN_NET": True}):
        on = [observe(tr) for tr in sample]
    off = [observe(tr) for tr in sample]
    noise = np.concatenate([rt.vr - true_radial_velocity(rt.truth, rt.radar_pos) for rt in on])
    ratio = float(np.nanstd(noise) / config.RADAR_SIGMA_VR)
    summary.check("Доплер: СКО шума соответствует заданному", abs(ratio - 1.0) <= 0.05,
                  f"{ratio:.3f} заданного ({config.RADAR_SIGMA_VR} м/с), отсчётов {np.isfinite(noise).sum()}")
    same = all(np.array_equal(x.meas, y.meas) and np.array_equal(x.polar, y.polar) for x, y in zip(on, off))
    summary.check("Доплер не меняет отметки положения", same and all(rt.vr is None for rt in off),
                  f"трасс {len(sample)}; без флагов радиальной скорости нет")
