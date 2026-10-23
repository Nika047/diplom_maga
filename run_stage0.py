"""
Этап 0: данные.

1. Проверки на подставных данных с известным ответом (ENU, разметка, очистка).
2. Чтение и очистка логов flight_logs/, разметка участков.
3. Разбиение реальных полётов на выборки (целыми логами).
4. Калибровка пределов генератора по train, генерация синтетики. В варианте данных merged
   (config.DATA_VARIANT) генерируется пул того же состава, но больше в 1 / (1 - VAL_SHARE -
   TEST_SHARE) раз, и он делится на train / val / test в тех же долях, что реальные логи
   (data/splits.split_synth); val и test сохраняются в tracks_synth_eval.pkl.
5. Проверки PASS / FAIL, кэш в cache/stage0/, рисунки и сводка в results/stage0/.

Запуск: .venv\\Scripts\\python.exe run_stage0.py
"""

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import flight_logs, generator, phases, splits as splits_mod  # noqa: E402
from data.geo import geodetic_to_enu, local_step_m  # noqa: E402
from data.track import Track  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from viz import map_html, plots  # noqa: E402

STAGE = "stage0"
OUT_DIR = config.RESULTS_DIR / STAGE
CACHE = config.CACHE_DIR / STAGE


def tracks_hash(tracks: list) -> str:
    h = hashlib.sha256()
    for tr in tracks:
        h.update(tr.flight_id.encode("utf-8"))
        h.update(np.ascontiguousarray(tr.t, dtype=np.float64).tobytes())
        h.update(np.round(np.ascontiguousarray(tr.pos, dtype=np.float64), 6).tobytes())
    return h.hexdigest()[:16]


def tracks_fingerprint(tracks: list) -> str:
    """
    Отпечаток для сверки синтетики между компьютерами. Точная сумма тут не годится:
    интегрирование на разных процессорах и версиях Python расходится в последнем
    разряде (на рабочем и домашнем ПК - 6e-8 м на сумме около 2e8 м).
    """
    total = sum(float(np.abs(tr.pos).sum()) for tr in tracks)
    return f"{len(tracks)} траекторий, {sum(len(tr) for tr in tracks)} точек, сумма |координат| {total:.3f} м"


def hours(tracks: list) -> float:
    return sum(tr.duration_s for tr in tracks) / 3600.0


# ============================================================================
# 1. Подставные данные с известным ответом
# ============================================================================

def self_tests(summary: StageSummary) -> None:
    # ENU: смещение на север даёт y, равный расстоянию по поверхности, и x около нуля
    lat0, lon0 = 59.9, 30.3
    enu = geodetic_to_enu(np.array([lat0, lat0 + 0.009]), np.array([lon0, lon0]),
                          np.array([0.0, 0.0]), lat0, lon0, 0.0)
    dist = float(local_step_m(lat0, lon0, lat0 + 0.009, lon0))
    summary.check("ENU: известный ответ", abs(enu[1, 1] - dist) < 2.0 and abs(enu[1, 0]) < 0.01
                  and np.allclose(enu[0], 0.0),
                  f"y = {enu[1, 1]:.1f} м при ожидаемых {dist:.1f} м, x = {enu[1, 0]:.3f} м")

    # Разметка: прямая -> только straight; прямая + окружность 10 град/с + прямая
    dt, v = 2.0, 20.0
    t = np.arange(0, 120, dt)
    straight = np.stack([np.zeros_like(t), v * t, np.zeros_like(t)], axis=1)
    sp, _, om = phases.compute_kinematics(t, straight)
    ph, onsets = phases.label_phases(t, sp, om)
    summary.check("Разметка: прямая", (ph == phases.PHASE_CODE["straight"]).all() and not onsets,
                  f"участков не-прямая: {(ph != phases.PHASE_CODE['straight']).sum()}")

    w = np.radians(10.0)
    pts, psi, p = [], 0.0, np.zeros(2)
    t3 = np.arange(0, 180 + dt, dt)
    for tk in t3:
        pts.append(p.copy())
        rate = w if 60 <= tk < 120 else 0.0
        for _ in range(20):  # мелкий шаг
            psi += rate * dt / 20
            p += v * np.array([np.sin(psi), np.cos(psi)]) * dt / 20
    pos = np.column_stack([np.array(pts), np.zeros(len(pts))])
    sp, _, om = phases.compute_kinematics(t3, pos)
    ph, onsets = phases.label_phases(t3, sp, om)
    in_circle = (t3 >= 70) & (t3 < 118)
    ok = (len(onsets) == 1 and abs(t3[onsets[0]] - 60) <= 4
          and (ph[in_circle] == phases.PHASE_CODE["turn"]).all()
          and (ph[(t3 >= 62) & (t3 < 66)] == phases.PHASE_CODE["entry"]).all()
          and (ph[t3 < 54] == phases.PHASE_CODE["straight"]).all())
    summary.check("Разметка: окружность 10 град/с между прямыми", ok,
                  f"начало маневра на {t3[onsets[0]] if onsets else None} с (ожидалось 60 с), "
                  f"|w| в середине {np.nanmedian(np.abs(om[in_circle])):.1f} град/с")

    # Очистка: выброс на 5 км убирается, пауза 30 с режет полёт на два куска
    n = 120
    tt = np.arange(n) * 2.0
    lat = 59.9 + np.arange(n) * 0.0002   # около 22 м за 2 с
    lon = np.full(n, 30.3)
    alt = np.full(n, 100.0)
    lat_glitch = lat.copy()
    lat_glitch[50] += 0.045
    raw = flight_logs.RawLog("Test-2025-01-01-000000.csv", "Test", tt, lat_glitch, lon, alt,
                             [""] * n, n, 0)
    tracks, st = flight_logs.clean_log(raw)
    vmax = (float((np.linalg.norm(np.diff(tracks[0].pos[:, :2], axis=0), axis=1) / np.diff(tracks[0].t)).max())
            if len(tracks) == 1 else float("nan"))
    summary.check("Очистка: одиночный выброс", len(tracks) == 1 and st.jumps == 1 and vmax < 15.0,
                  f"кусков {len(tracks)}, выбросов {st.jumps}, макс. скорость {vmax:.1f} м/с (истинная 11.1)")
    tt_gap = tt.copy()
    tt_gap[60:] += 30.0
    raw = flight_logs.RawLog("Test-2025-01-01-000000.csv", "Test", tt_gap, lat, lon, alt,
                             [""] * n, n, 0)
    tracks, st = flight_logs.clean_log(raw)
    summary.check("Очистка: пауза 30 с", len(tracks) == 2 and st.gap_cuts == 1,
                  f"кусков {len(tracks)}, разрезов по паузе {st.gap_cuts}")

    # Перепривязка времени: прямая 20 м/с, лог пишет раз в 2 с последнее значение GPS с
    # нерегулярной задержкой 0-1.5 с. До перепривязки скорость по шагам «пилит», после -
    # почти постоянна.
    v_true = 20.0
    t_log = np.arange(0, 300, 2.0)
    t_gps = t_log - (np.arange(len(t_log)) * 0.37) % 1.5
    pos = np.column_stack([np.zeros_like(t_gps), v_true * t_gps, np.zeros_like(t_gps)])
    v_before = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t_log)
    t_new, pos_new, _ = flight_logs.retime_and_resample(t_log, pos, [])
    v_after = np.linalg.norm(np.diff(pos_new, axis=0), axis=1) / np.diff(t_new)
    inner = v_after[10:-10]
    summary.check("Перепривязка времени: известный ответ",
                  np.std(v_before) / v_true > 0.2 and np.abs(inner - v_true).max() / v_true < 0.10,
                  f"разброс скорости до {np.std(v_before) / v_true:.0%}, после - макс. отклонение "
                  f"{np.abs(inner - v_true).max() / v_true:.1%} от {v_true:.0f} м/с")


# ============================================================================
# 2-4. Данные
# ============================================================================

def load_real(summary: StageSummary, log):
    t0 = time.time()
    tracks, total, per_file = flight_logs.load_all_logs()
    for tr in tracks:
        phases.annotate(tr)
    log.info("Логи прочитаны за %.1f с", time.time() - t0)

    n_files = len(list(config.FLIGHT_LOGS_DIR.glob("*.csv")))
    summary.check("Прочитаны все логи", len(per_file) == n_files == 150,
                  f"файлов {len(per_file)} из {n_files}, ожидалось 150")
    summary.section("Очистка логов (сумма по файлам)", [
        f"строк: {total.rows}, неразборчивых: {total.bad_rows}",
        f"высота вне диапазона: {total.alt_out_of_range}, время не растёт: {total.time_not_increasing}",
        f"зависшая телеметрия (повтор точки): {total.stale}",
        f"выбросов: {total.jumps}, новых кусков после серии выбросов: {total.jump_resets}",
        f"разрезов по паузам > {config.GAP_MAX_S:.0f} с: {total.gap_cuts}",
        f"вырезано стоячих точек: {total.stationary_removed}, в коротких кусках: {total.short_dropped_points}",
        f"осталось точек: {total.kept_points}, кусков: {total.chunks}",
    ])

    max_v = max_vz = max_dt = 0.0
    for tr in tracks:
        d = np.diff(tr.t)
        max_v = max(max_v, float((np.linalg.norm(np.diff(tr.pos[:, :2], axis=0), axis=1) / d).max()))
        max_vz = max(max_vz, float((np.abs(np.diff(tr.pos[:, 2])) / d).max()))
        max_dt = max(max_dt, float(d.max()))
    # ENU и шаговая оценка расстояния слегка расходятся - допуск 1%
    summary.check("После очистки нет скачков", max_v <= config.V_MAX * 1.01 and max_vz <= config.VZ_MAX * 1.01,
                  f"макс. скорость {max_v:.1f} м/с (предел {config.V_MAX}), "
                  f"макс. |vz| {max_vz:.1f} м/с (предел {config.VZ_MAX})")
    summary.check("После очистки нет пауз", max_dt <= config.GAP_MAX_S,
                  f"макс. шаг {max_dt:.1f} с (предел {config.GAP_MAX_S})")

    # До перепривязки времени корреляция соседних приращений скорости около -0.75 («пила»)
    corr = []
    for tr in tracks:
        v = np.linalg.norm(np.diff(tr.pos[:, :2], axis=0), axis=1) / np.diff(tr.t)
        dv = np.diff(v)
        if len(dv) > 10 and dv.std() > 0:
            corr.append(np.corrcoef(dv[:-1], dv[1:])[0, 1])
    med = float(np.median(corr))
    summary.check("Нет «пилы» скорости от рассинхрона GPS", med > -0.3,
                  f"медиана корреляции соседних приращений скорости {med:+.2f} (без перепривязки около -0.75)")
    return tracks


def describe_real(tracks: list, splits: dict, summary: StageSummary) -> None:
    by_type = {}
    for tr in tracks:
        by_type.setdefault(tr.aircraft, []).append(tr)
    lines = [f"{'тип':12s} {'логов':>5s} {'кусков':>6s} {'часов':>6s} {'км':>7s}  прошивка  выборки"]
    for aircraft in sorted(by_type):
        trs = by_type[aircraft]
        logs = sorted({tr.log_file for tr in trs})
        fw = ",".join(sorted({tr.firmware for tr in trs}))
        split_count = {}
        for lf in logs:
            split_count[splits[lf]] = split_count.get(splits[lf], 0) + 1
        sc = " ".join(f"{k}:{v}" for k, v in sorted(split_count.items()))
        lines.append(f"{aircraft:12s} {len(logs):5d} {len(trs):6d} {hours(trs):6.2f} "
                     f"{sum(tr.path_m for tr in trs) / 1000:7.0f}  {fw:9s} {sc}")
    summary.section("Реальные полёты по типам (после очистки)", lines)

    lines = []
    for name in splits_mod.SPLITS:
        trs = [tr for tr in tracks if splits[tr.log_file] == name]
        n_logs = len({tr.log_file for tr in trs})
        lines.append(f"{name:9s} логов {n_logs:3d}, кусков {len(trs):4d}, часов {hours(trs):6.2f}")
    summary.section("Выборки", lines)


def check_splits(tracks: list, splits: dict, summary: StageSummary) -> None:
    logs_used = {tr.log_file for tr in tracks if tr.aircraft in config.USED_AIRCRAFT}
    by_split = {}
    for lf, name in splits.items():
        by_split.setdefault(name, set()).add(lf)
    overlap = sum(len(by_split.get(a, set()) & by_split.get(b, set()))
                  for i, a in enumerate(splits_mod.SPLITS) for b in splits_mod.SPLITS[i + 1:])
    holdout_leak = [lf for lf, name in splits.items()
                    if config.USE_HOLDOUT and name in ("train", "val", "test")
                    and flight_logs.aircraft_from_filename(lf) == config.HOLDOUT_AIRCRAFT]
    sim_leak = [lf for lf, name in splits.items()
                if name in ("val", "test", "holdout")
                and flight_logs.aircraft_from_filename(lf) in config.SIM_AIRCRAFT]
    unassigned = [lf for lf in logs_used if lf not in splits]
    summary.check("Разбиение: выборки не пересекаются, всё распределено",
                  overlap == 0 and not unassigned, f"пересечений {overlap}, без выборки {len(unassigned)}")
    if config.USE_HOLDOUT:
        summary.check("Разбиение: отложенный тип и симулятор не в train/val/test",
                      not holdout_leak and not sim_leak,
                      f"отложенный тип {config.HOLDOUT_AIRCRAFT}: утечек {len(holdout_leak)}, симулятор: {len(sim_leak)}")
    else:
        summary.check("Разбиение: симулятор не в val/test", not sim_leak,
                      f"утечек {len(sim_leak)}; {config.HOLDOUT_AIRCRAFT} в общих данных (вариант merged)")
    for name in ("train", "val", "test") + (("holdout",) if config.USE_HOLDOUT else ()):
        n = len({tr.log_file for tr in tracks if splits[tr.log_file] == name})
        summary.check(f"Разбиение: выборка {name} не пуста", n > 0, f"логов {n}")
    used_h = hours([tr for tr in tracks if tr.aircraft in config.USED_AIRCRAFT])
    lo, hi = config.EXPECTED_USED_HOURS
    summary.check("Объём самолётных полётов в ожидаемом диапазоне", lo <= used_h <= hi,
                  f"{used_h:.1f} ч, ожидалось {lo}-{hi} ч")


def check_synth(synth: list, limits: generator.Limits, summary: StageSummary) -> None:
    worst_v_hi = worst_v_lo = worst_w = worst_vz = worst_step = 0.0
    for tr in synth:
        d = np.diff(tr.t)
        chord_v = np.linalg.norm(np.diff(tr.pos[:, :2], axis=0), axis=1) / d
        worst_v_hi = max(worst_v_hi, float((chord_v / limits.v_max).max()))
        worst_v_lo = max(worst_v_lo, float((limits.v_min / chord_v).max()))
        ratio = np.abs(tr.omega) / limits.omega_max_deg(tr.speed)
        worst_w = max(worst_w, float(np.nanmax(ratio)) if np.isfinite(ratio).any() else 0.0)
        worst_vz = max(worst_vz, float((np.abs(np.diff(tr.pos[:, 2])) / d / limits.vz_max).max()))
        step = np.linalg.norm(np.diff(tr.pos, axis=0), axis=1) / (np.hypot(limits.v_max, limits.vz_max) * d)
        worst_step = max(worst_step, float(step.max()))
    # Хорда короче дуги: при развороте на угол a за шаг скорость по хорде меньше в sin(a/2)/(a/2) раз
    arc = np.radians(float(limits.omega_max_deg(limits.v_min))) * config.SYNTH_DT_OUT
    chord = np.sin(arc / 2) / (arc / 2)
    summary.check("Синтетика: скорость в пределах", worst_v_hi <= 1.001 and worst_v_lo <= 1.0 / (0.98 * chord),
                  f"макс. V/v_max = {worst_v_hi:.3f}, макс. v_min/V = {worst_v_lo:.3f} "
                  f"(допуск {1.0 / (0.98 * chord):.3f} на хорду при развороте за шаг)")
    summary.check("Синтетика: разворот не круче предела", worst_w <= 1.15,
                  f"макс. |w| / min(g tg(phi_max) / V, предел) = {worst_w:.3f} (допуск 1.15 на сглаживание курса)")
    summary.check("Синтетика: вертикальная скорость в пределах", worst_vz <= 1.001,
                  f"макс. |vz| / vz_max = {worst_vz:.3f}")
    summary.check("Синтетика: нет скачков позиции", worst_step <= 1.001,
                  f"макс. шаг / (V_max dt) = {worst_step:.3f}")


def compare_distributions(real_train: list, synth: list, summary: StageSummary) -> None:
    def speeds(trs):
        return np.concatenate([tr.speed[tr.speed >= config.PHASE_V_MIN] for tr in trs])

    turn_w = phases.turn_omegas
    lines = []
    worst_v = worst_w = 0.0
    qs = (0.05, 0.5, 0.95)
    vr, vs = np.quantile(speeds(real_train), qs), np.quantile(speeds(synth), qs)
    for q, a, b in zip(qs, vr, vs):
        rel = abs(b - a) / a
        worst_v = max(worst_v, rel)
        lines.append(f"скорость q{int(q * 100):02d}: логи {a:5.1f}, синтетика {b:5.1f} м/с ({rel:+.0%})")
    qs_w = (0.5, 0.9)
    wr, ws = np.quantile(turn_w(real_train), qs_w), np.quantile(turn_w(synth), qs_w)
    for q, a, b in zip(qs_w, wr, ws):
        rel = abs(b - a) / a
        worst_w = max(worst_w, rel)
        lines.append(f"|w| на маневре q{int(q * 100):02d}: логи {a:5.1f}, синтетика {b:5.1f} град/с ({rel:+.0%})")
    summary.section("Логи (train) против синтетики", lines)
    summary.check("Синтетика: скорости близки к логам", worst_v <= config.CHECK_SPEED_TOL,
                  f"макс. отклонение квантилей {worst_v:.0%} (допуск {config.CHECK_SPEED_TOL:.0%})")
    summary.check("Синтетика: |w| на маневрах близка к логам", worst_w <= config.CHECK_OMEGA_TOL,
                  f"макс. отклонение квантилей {worst_w:.0%} (допуск {config.CHECK_OMEGA_TOL:.0%})")


def shares_table(groups: dict) -> tuple:
    lines = [f"{'группа':12s} " + " ".join(f"{p:>8s}" for p in config.PHASES) + "  маневров/ч"]
    shares = {}
    for name, trs in groups.items():
        sh = phases.phase_shares(trs)
        shares[name] = sh
        n_onsets = sum(len(tr.turn_onsets) for tr in trs)
        lines.append(f"{name:12s} " + " ".join(f"{sh[p]:8.1%}" for p in config.PHASES)
                     + f"  {n_onsets / max(hours(trs), 1e-9):9.1f}")
    return lines, shares


# ============================================================================
# Главная функция
# ============================================================================

def main() -> int:
    log = setup_logging(STAGE)
    summary = StageSummary("Этап 0: данные")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    log.info("1. Проверки на подставных данных")
    self_tests(summary)

    log.info("2. Логи полётов")
    real = load_real(summary, log)

    log.info("3. Разбиение")
    log_aircraft = {tr.log_file: tr.aircraft for tr in real}
    splits = splits_mod.make_splits(log_aircraft, make_rng("splits"))
    splits_mod.save_splits(splits, CACHE / "splits.json")
    describe_real(real, splits, summary)
    check_splits(real, splits, summary)
    real_train = [tr for tr in real if splits[tr.log_file] == "train"]

    log.info("4. Калибровка и генерация синтетики")
    limits, info = generator.calibrate_limits(real_train)
    summary.section("Пределы генератора (квантили логов train)", [
        f"скорость: {limits.v_min:.1f} - {limits.v_max:.1f} м/с (квантили {config.CALIB_V_QUANTILES})",
        f"предельный крен: {limits.phi_max_deg:.1f} град (квантиль {config.CALIB_PHI_QUANTILE})",
        f"вертикальная скорость: {limits.vz_max:.1f} м/с (квантиль {config.CALIB_VZ_QUANTILE})",
        f"предел измеренной |w|: {limits.omega_cap_deg:.1f} град/с (квантиль {config.CALIB_OMEGA_CAP_QUANTILE})",
        f"медиана скорости {np.median(limits.speed_q):.1f} м/с, медиана |w| на маневре "
        f"{np.median(limits.omega_q):.1f} град/с",
        f"точек для оценки: {info}",
    ])
    summary.check("Пределы генератора разумны",
                  0 < limits.v_min < limits.v_max <= config.V_MAX and 15 <= limits.phi_max_deg <= 60
                  and config.OMEGA_TURN_ON < limits.omega_cap_deg < 90,
                  f"v {limits.v_min:.1f}-{limits.v_max:.1f} м/с, крен {limits.phi_max_deg:.1f} град, "
                  f"|vz| {limits.vz_max:.1f} м/с, |w| до {limits.omega_cap_deg:.1f} град/с")
    t0 = time.time()
    target_w = float(np.median(phases.turn_omegas(real_train)))
    pilot_counts = {s: max(1, int(n * config.SYNTH_PILOT_SHARE)) for s, n in config.SYNTH_COUNT_TRAIN.items()}
    lines = []
    for it in range(config.SYNTH_PILOT_ITERATIONS):
        pilot = generator.generate(pilot_counts, limits, make_rng("synth_pilot"), "pilot")
        for tr in pilot:
            phases.annotate(tr)
        measured_w = float(np.median(phases.turn_omegas(pilot)))
        lines.append(f"итерация {it + 1}: множитель {limits.omega_scale:.3f}, медиана |w| "
                     f"синтетики {measured_w:.2f} при цели {target_w:.2f} град/с")
        limits.omega_scale = float(np.clip(limits.omega_scale * target_w / measured_w, *config.OMEGA_SCALE_RANGE))
    lines.append(f"итоговый множитель заданной |w|: {limits.omega_scale:.3f}")
    summary.section("Калибровка |w| через измерение (пробная генерация)", lines)
    if config.USE_HOLDOUT:
        synth_train = generator.generate(config.SYNTH_COUNT_TRAIN, limits, make_rng("synth_train"), "synth")
        synth_eval = {}
    else:
        pool_share = 1.0 - config.VAL_SHARE - config.TEST_SHARE
        pool = generator.generate({s: int(round(n / pool_share)) for s, n in config.SYNTH_COUNT_TRAIN.items()},
                                  limits, make_rng("synth_train"), "synth")
        parts = splits_mod.split_synth(pool, make_rng("synth_split"))
        synth_train, synth_eval = parts["train"], {k: parts[k] for k in ("val", "test")}
    synth_analysis = generator.generate(config.SYNTH_COUNT_ANALYSIS, limits, make_rng("synth_analysis"), "synthA")
    for tr in synth_train + synth_analysis + [tr for v in synth_eval.values() for tr in v]:
        phases.annotate(tr)
    log.info("Синтетика сгенерирована за %.1f с", time.time() - t0)
    lines = []
    for scenario in config.SYNTH_SCENARIOS:
        trs = [tr for tr in synth_train if tr.scenario == scenario]
        lines.append(f"{scenario:11s} траекторий {len(trs):4d}, часов {hours(trs):5.2f}, "
                     f"маневров {sum(len(tr.turn_onsets) for tr in trs):5d}")
    lines.append(f"набор для анализа: {len(synth_analysis)} траекторий, {hours(synth_analysis):.2f} ч")
    for name, trs in synth_eval.items():
        lines.append(f"вариант merged, {name}: {len(trs)} траекторий, {hours(trs):.2f} ч")
    summary.section("Синтетика (train)", lines)
    check_synth(synth_train + synth_analysis + [tr for v in synth_eval.values() for tr in v], limits, summary)
    if synth_eval:
        ids = [tr.flight_id for tr in synth_train] + [tr.flight_id for v in synth_eval.values() for tr in v]
        summary.check("Синтетика merged: train / val / test не пересекаются", len(ids) == len(set(ids)),
                      f"траекторий {len(ids)}, уникальных {len(set(ids))}")
    compare_distributions(real_train, synth_train, summary)

    small = {name: 3 for name in config.SYNTH_SCENARIOS}
    h1 = tracks_hash(generator.generate(small, limits, make_rng("check_determinism"), "det"))
    h2 = tracks_hash(generator.generate(small, limits, make_rng("check_determinism"), "det"))
    summary.check("Синтетика: детерминизм при одном seed", h1 == h2, f"{h1} / {h2}")

    used = [tr for tr in real if tr.aircraft in config.USED_AIRCRAFT]
    groups = {}
    for tr in used:
        groups.setdefault(tr.aircraft, []).append(tr)
    groups["синтетика"] = synth_train
    lines, shares = shares_table(groups)
    summary.section("Доли участков", lines)
    turn_share = sum(phases.phase_shares(used)[p] for p in ("entry", "turn"))
    summary.check("Доля маневров в логах правдоподобна", 0.05 <= turn_share <= 0.7,
                  f"вход + маневр = {turn_share:.0%} времени самолётных полётов")

    log.info("5. Кэш и контрольные суммы")
    with open(CACHE / "tracks_real.pkl", "wb") as fh:
        pickle.dump(real, fh)
    with open(CACHE / "tracks_synth_train.pkl", "wb") as fh:
        pickle.dump(synth_train, fh)
    with open(CACHE / "tracks_synth_analysis.pkl", "wb") as fh:
        pickle.dump(synth_analysis, fh)
    if synth_eval:
        with open(CACHE / "tracks_synth_eval.pkl", "wb") as fh:
            pickle.dump(synth_eval, fh)
    (CACHE / "calibration.json").write_text(json.dumps(limits.as_dict(), indent=1), encoding="utf-8")
    summary.section("Контрольные суммы (сравнивать между запусками и компьютерами)", [
        f"реальные треки: {tracks_hash(real)}",
        f"разбиение: {hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()[:16]}",
        f"синтетика train (отпечаток): {tracks_fingerprint(synth_train)}",
        f"синтетика анализ (отпечаток): {tracks_fingerprint(synth_analysis)}",
        f"синтетика train, точная сумма (может отличаться между компьютерами): {tracks_hash(synth_train)}",
    ])

    log.info("6. Рисунки")
    map_html.write_flights_map(real, splits, OUT_DIR / "map_real_flights.html", "Реальные полёты после очистки")
    picked, seen = [], set()
    for tr in sorted(real_train, key=lambda x: (-len(x.turn_onsets), x.flight_id)):
        if tr.log_file not in seen:
            picked.append(tr)
            seen.add(tr.log_file)
        if len(picked) == 8:
            break
    plots.plot_tracks_phases(picked, OUT_DIR / "real_examples_phases.png",
                             "Реальные полёты (train) с разметкой участков")
    if picked:
        plots.plot_track_timeline(picked[0], OUT_DIR / "real_timeline.png")
    examples = [next(tr for tr in synth_analysis if tr.scenario == s) for s in config.SYNTH_SCENARIOS]
    plots.plot_tracks_phases(examples, OUT_DIR / "synth_examples_phases.png",
                             "Синтетика: по одному примеру каждого сценария")
    plots.plot_track_timeline(next(tr for tr in synth_analysis if tr.scenario == "zigzag"),
                              OUT_DIR / "synth_timeline.png")
    plots.plot_distributions(real_train, synth_train, OUT_DIR / "distributions.png")
    plots.plot_phase_shares(shares, OUT_DIR / "phase_shares.png")

    summary.write(OUT_DIR / "summary.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
