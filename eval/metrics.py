"""
Метрики прогноза - единые для всех этапов. В скрипты этапов не копировать.

Прогноз из обзора k на h обзоров сравнивается с истиной в обзоре k + h. Пара (k, h)
учитывается, если k >= WARMUP_SCANS, k + h есть в трассе и прогноз существует.

Основные метрики - по горизонтали (x, y): маневры горизонтальные, а угол места РЛС
меряет грубо (ошибка высоты ~ дальность * СКО угла места - десятки метров), и в 3D
она заслоняла бы всё остальное. 3D-ошибка считается отдельно (ade_3d).

  MAE  - среднее |ошибки| по компонентам x и y;
  RMSE - корень из среднего квадрата горизонтальной ошибки;
  ADE  - средняя горизонтальная ошибка по всем горизонтам;
  FDE  - средняя горизонтальная ошибка на последнем горизонте H;
  R2   - 1 - сумма квадратов ошибок / сумма квадратов отклонений истинного смещения
         от среднего; смещение - от последней отметки (k) до истины в k + h.

Динамическая ошибка: для каждого начала маневра (обзор onset) берутся прогнозы из
обзоров onset + m, m в DYN_OFFSETS, на горизонте h; кривая - средняя ошибка по m.
  пик - максимум кривой при m >= 0;
  площадь - среднее за первые DYN_AREA_SCANS обзоров после начала маневра;
  восстановление - первый m после пика, где ошибка не выше уровня прямой,
  умноженного на DYN_RECOVERY_FACTOR (уровень прямой - средняя ошибка на этом
  горизонте из обзоров участка "straight").

Проверяется на этапе 1 на подставных данных с известным ответом.
"""

import numpy as np

import config
from data.phases import PHASE_CODE


def forecast_pairs(rt, forecast: np.ndarray, warmup: int = config.WARMUP_SCANS) -> dict:
    """Все пары (k, h) одной трассы: ошибка, смещение, участок в момент k."""
    k_total, horizon, _ = forecast.shape
    rows = {"err": [], "disp": [], "h": [], "k": [], "phase": []}
    for h in range(1, horizon + 1):
        k = np.arange(warmup, k_total - h)
        if len(k) == 0:
            continue
        pred = forecast[k, h - 1]
        ok = ~np.isnan(pred).any(axis=1)
        k = k[ok]
        rows["err"].append(forecast[k, h - 1] - rt.truth[k + h])
        rows["disp"].append(rt.truth[k + h] - rt.meas[k])
        rows["h"].append(np.full(len(k), h))
        rows["k"].append(k)
        rows["phase"].append(rt.phase[k])
    if not rows["err"]:
        return {"err": np.zeros((0, 3)), "disp": np.zeros((0, 3)), "h": np.zeros(0, int),
                "k": np.zeros(0, int), "phase": np.zeros(0, np.int8)}
    return {key: np.concatenate(val) for key, val in rows.items()}


def concat_pairs(parts: list) -> dict:
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def summarize(pairs: dict, horizon: int = config.HORIZON_SCANS) -> dict:
    """Метрики по набору пар (k, h)."""
    err, disp, h = pairs["err"], pairs["disp"], pairs["h"]
    if len(err) == 0:
        return {name: float("nan") for name in ("mae", "rmse", "ade", "fde", "r2", "ade_3d", "n")}
    e2 = err[:, :2]
    dist = np.linalg.norm(e2, axis=1)
    d2 = disp[:, :2]
    sst = float(((d2 - d2.mean(axis=0)) ** 2).sum())
    return {
        "mae": float(np.abs(e2).mean()),
        "rmse": float(np.sqrt((dist ** 2).mean())),
        "ade": float(dist.mean()),
        "fde": float(dist[h == horizon].mean()) if (h == horizon).any() else float("nan"),
        "r2": 1.0 - float((dist ** 2).sum()) / sst if sst > 0 else float("nan"),
        "ade_3d": float(np.linalg.norm(err, axis=1).mean()),
        "n": int(len(err)),
    }


def ade_by_horizon(pairs: dict, horizon: int = config.HORIZON_SCANS) -> np.ndarray:
    dist = np.linalg.norm(pairs["err"][:, :2], axis=1)
    return np.array([dist[pairs["h"] == h].mean() if (pairs["h"] == h).any() else np.nan
                     for h in range(1, horizon + 1)])


def select(pairs: dict, mask: np.ndarray) -> dict:
    return {key: val[mask] for key, val in pairs.items()}


def dynamic_curve(radar_tracks: list, forecasts: list, h: int,
                  offsets=config.DYN_OFFSETS) -> tuple:
    """Кривая динамической ошибки: (m, средняя ошибка, число маневров) по всем трассам."""
    ms = np.arange(offsets[0], offsets[1] + 1)
    sums = np.zeros(len(ms))
    counts = np.zeros(len(ms), dtype=int)
    for rt, fc in zip(radar_tracks, forecasts):
        k_total = len(rt.t)
        for onset in rt.onsets:
            for i, m in enumerate(ms):
                k = onset + m
                if k < config.WARMUP_SCANS or k + h >= k_total or np.isnan(fc[k, h - 1]).any():
                    continue
                sums[i] += np.linalg.norm(fc[k, h - 1, :2] - rt.truth[k + h, :2])
                counts[i] += 1
    with np.errstate(invalid="ignore"):
        return ms, sums / counts, counts


def straight_level(pairs: dict, h: int) -> float:
    sel = (pairs["h"] == h) & (pairs["phase"] == PHASE_CODE["straight"])
    return float(np.linalg.norm(pairs["err"][sel, :2], axis=1).mean()) if sel.any() else float("nan")


def dynamic_stats(ms: np.ndarray, curve: np.ndarray, level: float) -> dict:
    after = ms >= 0
    m_after, c_after = ms[after], curve[after]
    valid = ~np.isnan(c_after)
    if not valid.any():
        return {"peak": float("nan"), "peak_m": float("nan"), "area": float("nan"), "recovery": float("nan")}
    i_peak = int(np.nanargmax(c_after))
    area_sel = (m_after < config.DYN_AREA_SCANS) & valid
    recovery = float("nan")
    for m, c in zip(m_after[i_peak:], c_after[i_peak:]):
        if not np.isnan(c) and c <= level * config.DYN_RECOVERY_FACTOR:
            recovery = float(m)
            break
    return {"peak": float(c_after[i_peak]), "peak_m": float(m_after[i_peak]),
            "area": float(c_after[area_sel].mean()), "recovery": recovery}


# ============================================================================
# Начало маневра (этап 5)
# ============================================================================
# Обнаружитель выдаёт оценку (score) в каждом обзоре с окном; тревога - score >= порога.
# Метка обзора - data/windows.onset_labels. Обнаружители сравниваются при одинаковой
# вероятности ложной тревоги: порог выбирается на val (threshold_at_false_alarm).
# «Спокойный» обзор - участок "straight" вне интервала метки любого начала маневра:
# тревога в нём ложная.

def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Средняя точность - площадь под кривой «точность - полнота» по обзорам, без порога."""
    order = np.argsort(-score, kind="stable")
    lab = label[order] > 0.5
    if not lab.any():
        return float("nan")
    precision = np.cumsum(lab) / np.arange(1, len(lab) + 1)
    return float(precision[lab].sum() / lab.sum())


def binary_scores(alarm: np.ndarray, label: np.ndarray) -> dict:
    """Точность, полнота и F1 тревог по обзорам."""
    lab = label > 0.5
    tp = float((alarm & lab).sum())
    precision = tp / alarm.sum() if alarm.any() else float("nan")
    recall = tp / lab.sum() if lab.any() else float("nan")
    f1 = 2 * tp / (alarm.sum() + lab.sum()) if (alarm.any() or lab.any()) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1}


def _quiet(rt, ks: np.ndarray, ahead: int, behind: int) -> np.ndarray:
    """Спокойные обзоры среди ks: прямая вне интервалов o - ahead .. o + behind."""
    near = np.zeros(len(ks), dtype=bool)
    for o in rt.onsets:
        near |= (ks >= o - ahead) & (ks <= o + behind)
    return (rt.phase[ks] == PHASE_CODE["straight"]) & ~near


def threshold_at_false_alarm(tracks: list, p_fa: float = config.ONSET_P_FA, ahead: int = config.ONSET_AHEAD,
                             behind: int = config.ONSET_BEHIND) -> float:
    """
    Порог, при котором тревога горит не больше чем в доле p_fa спокойных обзоров
    (вероятность ложной тревоги на обзор). tracks - как в onset_events.
    """
    neg = np.sort(np.concatenate([np.asarray(s)[_quiet(rt, np.asarray(ks), ahead, behind)]
                                  for rt, ks, s in tracks]))[::-1]
    if len(neg) == 0:
        return float("nan")
    allowed = int(np.floor(p_fa * len(neg)))
    return float(np.nextafter(neg[min(allowed, len(neg) - 1)], np.inf))  # тревога строго выше этого значения


def _covered_onsets(rt, ks: np.ndarray, ahead: int, behind: int) -> list:
    """Начала маневров, у которых все обзоры o - ahead .. o + behind есть в ks."""
    have = set(int(k) for k in ks)
    return [o for o in rt.onsets if all(k in have for k in range(o - ahead, o + behind + 1))]


def onset_events(tracks: list, threshold: float, ahead: int = config.ONSET_AHEAD,
                 behind: int = config.ONSET_BEHIND, late: int = config.ONSET_DETECT_LATE) -> dict:
    """
    Оценка по событиям. tracks - [(RadarTrack, обзоры ks, score в этих обзорах)].
    Эпизод тревоги - подряд идущие тревожные обзоры; его начало - первый из них.
      recall - доля начал маневров o, у которых эпизод тревоги начался в
             o - ahead .. o + late (late шире интервала метки: видна и поздняя реакция
             классических обнаружителей). Тревога, горевшая ещё до интервала, не считается:
             это не реакция на этот маневр (иначе обнаружитель, который почти всегда
             в тревоге, получал бы полноту 1 и опережение ahead);
      lead - опережение в обзорах: o минус начало первого такого эпизода (больше нуля -
             до начала маневра, меньше - с задержкой);
      p_fa - доля спокойных обзоров с тревогой (вероятность ложной тревоги);
      false_per_hour - эпизоды, начавшиеся в спокойных обзорах, на час спокойного полёта.
    """
    leads, n_onsets, false_eps, false_scans, quiet_scans = [], 0, 0, 0, 0
    for rt, ks, score in tracks:
        ks = np.asarray(ks)
        alarm = score >= threshold
        starts = alarm & ~np.r_[False, alarm[:-1] & (np.diff(ks) == 1)]
        pos = {int(k): i for i, k in enumerate(ks)}
        for o in _covered_onsets(rt, ks, ahead, late):
            n_onsets += 1
            hit = [k for k in range(o - ahead, o + late + 1) if starts[pos[k]]]
            if hit:
                leads.append(o - hit[0])
        quiet = _quiet(rt, ks, ahead, behind)
        quiet_scans += int(quiet.sum())
        false_scans += int((alarm & quiet).sum())
        false_eps += int((starts & quiet).sum())
    leads = np.array(leads, dtype=float)
    hours = quiet_scans * config.SCAN_PERIOD / 3600.0
    return {"n_onsets": n_onsets, "recall": len(leads) / n_onsets if n_onsets else float("nan"),
            "lead_mean": float(leads.mean()) if len(leads) else float("nan"),
            "lead_median": float(np.median(leads)) if len(leads) else float("nan"),
            "early_share": float((leads > 0).mean()) if len(leads) else float("nan"),
            "p_fa": false_scans / quiet_scans if quiet_scans else float("nan"),
            "false_per_hour": false_eps / hours if hours > 0 else float("nan")}


def alarm_response(tracks: list, threshold: float, offsets=config.DYN_OFFSETS) -> tuple:
    """Доля тревог в обзорах o + m относительно начала маневра: (m, доля, число маневров)."""
    ms = np.arange(offsets[0], offsets[1] + 1)
    hits = np.zeros(len(ms))
    counts = np.zeros(len(ms), dtype=int)
    for rt, ks, score in tracks:
        pos = {int(k): i for i, k in enumerate(ks)}
        for o in rt.onsets:
            for i, m in enumerate(ms):
                if o + m in pos:
                    hits[i] += float(score[pos[o + m]] >= threshold)
                    counts[i] += 1
    with np.errstate(invalid="ignore"):
        return ms, hits / counts, counts


# ============================================================================
# Строб - область прогноза (этап 6б)
# ============================================================================
# Область вокруг прогноза на горизонте h: круг радиуса r или эллипс {e: e^T C^-1 e <= q}.
# Конформная калибровка (split conformal): радиус - квантиль меры несоответствия на
# калибровочном наборе (val) уровня ceil((n + 1) * покрытие) / n. При
# перестановочности примеров это даёт покрытие не ниже заданного на новых данных.

def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Конформный квантиль: значение с номером ceil((n + 1) * coverage) по возрастанию."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * coverage))
    return float(s[min(rank, n) - 1]) if rank <= n else float("inf")


def mahalanobis2(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """e^T C^-1 e для err (N, 2) и cov (N, 2, 2)."""
    return np.einsum("ni,nij,nj->n", err, np.linalg.inv(cov), err)


def gate_report(inside: np.ndarray, area: np.ndarray, phase: np.ndarray, h: np.ndarray) -> dict:
    """
    Покрытие (доля истин внутри области) в целом, по участкам и по горизонтам;
    размер - эквивалентный радиус sqrt(площадь / pi), медиана и среднее, м.
    """
    def cov_of(mask):
        return float(inside[mask].mean()) if mask.any() else float("nan")
    radius = np.sqrt(area / np.pi)
    out = {"coverage": cov_of(np.ones(len(inside), bool)),
           "radius_median": float(np.median(radius)) if len(radius) else float("nan"),
           "radius_mean": float(radius.mean()) if len(radius) else float("nan"), "n": int(len(inside))}
    for name in ("straight", "entry", "turn", "exit"):
        out[f"coverage_{name}"] = cov_of(phase == PHASE_CODE[name])
    for hh in np.unique(h):
        out[f"coverage_h{int(hh)}"] = cov_of(h == hh)
        out[f"radius_h{int(hh)}"] = float(np.median(radius[h == hh]))
    return out
