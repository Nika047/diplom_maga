"""
Чтение и очистка логов полётов EdgeTX/OpenTX из flight_logs/.

Из лога берутся только время, GPS и высота (истина) и режим полёта (для анализа). Скорость
GPS-приёмника (GSpd, Hdg, VSpd группы GPS) переносится в Track.vel_log только для доплера по
телеметрии (этап 18); во входы моделей не идёт.
Очистка, по шагам:
  1. строки без GPS, с нулевыми координатами или высотой вне [ALT_MIN, ALT_MAX];
  2. повторы времени и «зависшая» телеметрия (те же координаты и высота подряд);
  3. скачки: горизонтальная скорость от последней принятой точки больше V_MAX или
     вертикальная больше VZ_MAX. После JUMP_RESET_N отбраковок подряд - новый кусок
     с текущей точки (иначе один выброс в начале отбросил бы весь полёт);
  4. разрез по паузам в записи длиннее GAP_MAX_S - без интерполяции через паузу;
  5. стоянки (скорость ниже V_MOVING_MIN дольше STATIONARY_MAX_S) вырезаются,
     в начале и в конце куска стоячие точки отрезаются всегда;
  6. куски короче MIN_CHUNK_POINTS / MIN_CHUNK_DURATION_S отбрасываются.
Каждый кусок переводится в ENU с началом в своей первой точке.

Перепривязка времени (RETIME): GPS обновляется не в моменты записи лога, поэтому
смещения за шаг чередуются (корреляция соседних приращений скорости около -0.75).
Время точки заменяется моментом, когда сглаженная кривая пройденного пути s(t)
достигает пути этой точки; затем трек интерполируется на равномерную сетку с
исходным шагом (между соседними точками, пауз длиннее GAP_MAX_S внутри куска нет).
Форма траектории не меняется, меняется только время точек.

Проверяется на этапе 0: после очистки нет шагов быстрее V_MAX и VZ_MAX, нет пауз
длиннее GAP_MAX_S.
"""

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from data.geo import geodetic_to_enu, local_step_m
from data.track import Track

_NAME_RE = re.compile(r"^(?P<aircraft>.+)-(?P<date>\d{4}-\d{2}-\d{2})-(?P<time>\d{6})\.csv$")


@dataclass
class RawLog:
    log_file: str
    aircraft: str
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    modes: list
    n_rows: int
    n_bad_rows: int
    vel: np.ndarray = None   # (N, 3) м/с ENU по GSpd, Hdg, VSpd; nan - нет в строке


@dataclass
class CleanStats:
    """Счётчики очистки по файлу или суммарно."""
    rows: int = 0
    bad_rows: int = 0
    alt_out_of_range: int = 0
    time_not_increasing: int = 0
    stale: int = 0
    jumps: int = 0
    jump_resets: int = 0
    gap_cuts: int = 0
    stationary_removed: int = 0
    short_dropped_points: int = 0
    kept_points: int = 0
    chunks: int = 0
    extra: Counter = field(default_factory=Counter)

    def add(self, other: "CleanStats") -> None:
        for name in self.__dataclass_fields__:
            if name != "extra":
                setattr(self, name, getattr(self, name) + getattr(other, name))


def aircraft_from_filename(name: str) -> str:
    """Тип аппарата - префикс имени файла до даты."""
    match = _NAME_RE.match(name)
    return match.group("aircraft") if match else "unknown"


def source_of(aircraft: str) -> str:
    return "sim" if aircraft in config.SIM_AIRCRAFT else "real"


def _parse_time(date: str, time: str) -> float:
    text = f"{date.strip()} {time.strip()}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(text)


def _clean_mode(value: str) -> str:
    return value.strip().strip('"').strip().rstrip("*").strip()


def _velocity_columns(header: list) -> tuple:
    """Индексы GSpd, Hdg группы GPS (первые после столбца GPS; у части логов есть вторые Hdg, GSpd
    от полётного контроллера раньше GPS) и VSpd; множитель GSpd в м/с."""
    names = [h.strip() for h in header]
    start = names.index("GPS") if "GPS" in names else 0

    def find(prefix):
        hits = [i for i, n in enumerate(names) if n.startswith(prefix)]
        after = [i for i in hits if i > start]
        return (after or hits or [None])[0]

    i_spd, i_hdg = find("GSpd"), find("Hdg")
    scale = 1.0 / 3.6 if i_spd is not None and "kmh" in names[i_spd] else 1.0
    return i_spd, i_hdg, find("VSpd"), scale


def _float(row: list, i) -> float:
    try:
        return float(row[i]) if i is not None else float("nan")
    except (IndexError, ValueError):
        return float("nan")


def read_log(path: Path) -> RawLog:
    """Читает CSV; строки с неразборчивым временем или GPS пропускаются (считаются в n_bad_rows)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {}
        for i, name in enumerate(header):
            col.setdefault(name.strip(), i)  # при повторе имени берём первое
        i_date, i_time, i_gps = col["Date"], col["Time"], col["GPS"]
        i_alt, i_fm = col["Alt(m)"], col.get("FM")
        i_spd, i_hdg, i_vs, spd_scale = _velocity_columns(header)
        t, lat, lon, alt, modes, vel = [], [], [], [], [], []
        n_rows = n_bad = 0
        for row in reader:
            n_rows += 1
            try:
                parts = row[i_gps].replace(",", " ").split()
                la, lo = float(parts[0]), float(parts[1])
                ts = _parse_time(row[i_date], row[i_time])
                al = float(row[i_alt])
            except (IndexError, ValueError):
                n_bad += 1
                continue
            t.append(ts)
            lat.append(la)
            lon.append(lo)
            alt.append(al)
            modes.append(_clean_mode(row[i_fm]) if i_fm is not None and i_fm < len(row) else "")
            gs, hdg = _float(row, i_spd) * spd_scale, np.radians(_float(row, i_hdg))
            vel.append((gs * np.sin(hdg), gs * np.cos(hdg), _float(row, i_vs)))
    return RawLog(path.name, aircraft_from_filename(path.name), np.array(t), np.array(lat),
                  np.array(lon), np.array(alt), modes, n_rows, n_bad, np.array(vel).reshape(-1, 3))


def detect_firmware(modes: list) -> str:
    codes = Counter(m for m in modes if m)
    inav = sum(n for m, n in codes.items() if m in config.INAV_MODES)
    ardu = sum(n for m, n in codes.items() if m in config.ARDUPILOT_MODES)
    if inav == ardu == 0:
        return "unknown"
    return "inav" if inav >= ardu else "ardupilot"


def _jump_filter(t, lat, lon, alt, stats: CleanStats):
    """Возвращает (маска принятых точек, флаг «новый кусок перед точкой»)."""
    n = len(t)
    keep = np.zeros(n, dtype=bool)
    new_piece = np.zeros(n, dtype=bool)
    if n == 0:
        return keep, new_piece
    keep[0] = new_piece[0] = True
    last = 0
    rejected = 0
    for i in range(1, n):
        dt = t[i] - t[last]
        vh = local_step_m(lat[last], lon[last], lat[i], lon[i]) / dt
        vz = abs(alt[i] - alt[last]) / dt
        if vh <= config.V_MAX and vz <= config.VZ_MAX:
            keep[i] = True
            last, rejected = i, 0
            continue
        rejected += 1
        stats.jumps += 1
        if rejected >= config.JUMP_RESET_N:
            keep[i] = new_piece[i] = True
            last, rejected = i, 0
            stats.jump_resets += 1
            stats.jumps -= 1
    return keep, new_piece


def _split_by_flags(n: int, cut_before: np.ndarray) -> list:
    """Делит range(n) на отрезки индексов; cut_before[i] = True - отрезок начинается с i."""
    starts = list(np.flatnonzero(cut_before))
    if not starts or starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [n]
    return [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _remove_stationary(t, lat, lon, stats: CleanStats) -> list:
    """Делит кусок по длинным стоянкам и отрезает стоячие концы; возвращает отрезки индексов."""
    n = len(t)
    if n < 2:
        return [np.arange(n)]
    speed = local_step_m(lat[:-1], lon[:-1], lat[1:], lon[1:]) / np.diff(t)
    moving = np.empty(n, dtype=bool)
    moving[1:] = speed >= config.V_MOVING_MIN
    moving[0] = moving[1]
    keep = np.ones(n, dtype=bool)
    cut_before = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if moving[i]:
            i += 1
            continue
        j = i
        while j < n and not moving[j]:
            j += 1
        at_edge = i == 0 or j == n
        if at_edge or t[j - 1] - t[i] >= config.STATIONARY_MAX_S:
            keep[i:j] = False
            if j < n:
                cut_before[j] = True
        i = j
    stats.stationary_removed += int((~keep).sum())
    pieces = []
    for seg in _split_by_flags(n, cut_before):
        seg = seg[keep[seg]]
        if len(seg):
            pieces.append(seg)
    return pieces


def retime_and_resample(t: np.ndarray, pos: np.ndarray, extra: list) -> tuple:
    """
    Перепривязка времени по длине пути и ресемплирование на равномерную сетку.

    extra - список массивов (N,) или (N, k), которые интерполируются вместе с pos
    (широта, долгота); для нечисловых данных передаются индексы точек.
    Возвращает (t_new, pos_new, extra_new).
    """
    n = len(t)
    step = float(np.median(np.diff(t)))
    # Путь в 3D: при наборе высоты почти на месте (VTOL) горизонтальный путь не растёт
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    k = max(3, int(round(config.RETIME_SMOOTH_S / step)) | 1)
    pad = k // 2
    if n <= k:
        return t, pos, extra
    # Края продолжаются линейно, чтобы сглаживание не сжимало путь в начале и конце
    head = s[0] - (s[pad] - s[0]) / pad * np.arange(pad, 0, -1)
    tail = s[-1] + (s[-1] - s[-1 - pad]) / pad * np.arange(1, pad + 1)
    s_smooth = np.convolve(np.concatenate([head, s, tail]), np.ones(k) / k, mode="valid")
    s_smooth = np.maximum.accumulate(s_smooth + 1e-9 * np.arange(n))
    t_true = np.interp(s, s_smooth, t)
    # GPS отстаёт от записи не больше чем примерно на шаг лога: на стоянках и
    # зависаниях путь не растёт, и без ограничения время сжималось бы
    t_true = np.clip(t_true, t - config.RETIME_MAX_SHIFT_S, t + config.RETIME_MAX_SHIFT_S)
    t_true = np.maximum.accumulate(t_true + 1e-9 * np.arange(n))
    t_new = np.arange(t_true[0], t_true[-1] + 1e-9, step)
    pos_new = np.column_stack([np.interp(t_new, t_true, pos[:, j]) for j in range(pos.shape[1])])
    extra_new = []
    for arr in extra:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            extra_new.append(np.interp(t_new, t_true, arr))
        else:
            extra_new.append(np.column_stack([np.interp(t_new, t_true, arr[:, j]) for j in range(arr.shape[1])]))
    return t_new - t_new[0], pos_new, extra_new


def clean_log(raw: RawLog) -> tuple:
    """Очистка одного лога -> (список Track, CleanStats)."""
    stats = CleanStats(rows=raw.n_rows, bad_rows=raw.n_bad_rows)
    t, lat, lon, alt = raw.t, raw.lat, raw.lon, raw.alt
    vel = raw.vel if raw.vel is not None else np.full((len(t), 3), np.nan)
    modes = np.array(raw.modes, dtype=object)

    ok = ~((lat == 0) & (lon == 0)) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    in_range = (alt >= config.ALT_MIN) & (alt <= config.ALT_MAX)
    stats.alt_out_of_range = int((ok & ~in_range).sum())
    ok &= in_range
    idx = np.flatnonzero(ok)

    # Время строго растёт, зависшая телеметрия (те же координаты и высота) - выкидывается
    keep_idx = []
    for i in idx:
        if keep_idx:
            p = keep_idx[-1]
            if t[i] <= t[p]:
                stats.time_not_increasing += 1
                continue
            if lat[i] == lat[p] and lon[i] == lon[p] and alt[i] == alt[p]:
                stats.stale += 1
                continue
        keep_idx.append(i)
    idx = np.array(keep_idx, dtype=int)
    t, lat, lon, alt, modes, vel = t[idx], lat[idx], lon[idx], alt[idx], modes[idx], vel[idx]

    keep, new_piece = _jump_filter(t, lat, lon, alt, stats)
    t, lat, lon, alt, modes, new_piece, vel = (a[keep] for a in (t, lat, lon, alt, modes, new_piece, vel))

    cut = new_piece.copy()
    if len(t):
        gaps = np.zeros(len(t), dtype=bool)
        gaps[1:] = np.diff(t) > config.GAP_MAX_S
        stats.gap_cuts = int(gaps.sum())
        cut |= gaps

    tracks = []
    firmware = detect_firmware(raw.modes)
    stem = Path(raw.log_file).stem
    for piece in _split_by_flags(len(t), cut):
        for seg in _remove_stationary(t[piece], lat[piece], lon[piece], stats):
            sel = piece[seg]
            duration = t[sel[-1]] - t[sel[0]]
            if len(sel) < config.MIN_CHUNK_POINTS or duration < config.MIN_CHUNK_DURATION_S:
                stats.short_dropped_points += len(sel)
                continue
            pos = geodetic_to_enu(lat[sel], lon[sel], alt[sel], lat[sel[0]], lon[sel[0]], alt[sel[0]])
            t_sel, lat_sel, lon_sel, mode_sel = t[sel] - t[sel[0]], lat[sel], lon[sel], list(modes[sel])
            vel_sel = vel[sel]
            if config.RETIME:
                t_sel, pos, (lat_sel, lon_sel, idx, vel_sel) = retime_and_resample(
                    t_sel, pos, [lat_sel, lon_sel, np.arange(len(sel)), vel_sel])
                mode_sel = [mode_sel[int(round(i))] for i in idx]
            tracks.append(Track(
                flight_id=f"{stem}#{len(tracks)}",
                source=source_of(raw.aircraft),
                aircraft=raw.aircraft,
                t=t_sel,
                pos=pos,
                log_file=raw.log_file,
                firmware=firmware,
                lat=lat_sel,
                lon=lon_sel,
                flight_mode=mode_sel,
                vel_log=np.asarray(vel_sel, dtype=float),
            ))
            stats.kept_points += len(sel)
    stats.chunks = len(tracks)
    return tracks, stats


def attach_log_velocity(tracks: list, logs_dir: Path = config.FLIGHT_LOGS_DIR) -> tuple:
    """
    Скорость по телеметрии (Track.vel_log) для треков из кэша этапа 0, собранных до её появления:
    логи читаются и очищаются заново, трек сопоставляется по flight_id, совпадение координат
    проверяется. Меняет треки на месте; возвращает (число с vel_log, число несовпадений).
    """
    fresh = {tr.flight_id: tr for tr in load_all_logs(logs_dir)[0]}
    attached = mismatch = 0
    for tr in tracks:
        if tr.source == "synth":
            continue
        new = fresh.get(tr.flight_id)
        if new is None or new.pos.shape != tr.pos.shape or not np.allclose(new.pos, tr.pos, atol=1e-6):
            mismatch += 1
            continue
        tr.vel_log = new.vel_log
        attached += 1
    return attached, mismatch


def load_all_logs(logs_dir: Path = config.FLIGHT_LOGS_DIR) -> tuple:
    """Читает и очищает все CSV папки. Возвращает (треки, суммарные счётчики, счётчики по файлам)."""
    tracks, total, per_file = [], CleanStats(), {}
    for path in sorted(logs_dir.glob("*.csv")):
        raw = read_log(path)
        file_tracks, stats = clean_log(raw)
        tracks.extend(file_tracks)
        total.add(stats)
        per_file[path.name] = stats
    return tracks, total, per_file
