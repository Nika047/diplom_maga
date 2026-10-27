"""
Программа прогнозирования положения воздушного объекта (этап 11 версии 2).

Прогноз - итоговая модель случая доплера: IMM с параметрами генетического подбора даёт прогноз на 1..H обзоров
(до 10 с), сеть - поправку к нему. Три случая (--case): nodop - РЛС без доплера, dop_traj и dop_tele - с доплером
(для доплера по телеметрии источник скорости при моделировании отметок из лога - скорость GPS-приёмника).
Для сравнения рядом - прогноз IMM без поправки. --ensemble - среднее поправок всех seed итоговой модели.

Вход - один из двух:
  --log файл.csv   - лог полёта EdgeTX/OpenTX: очистка как на этапе 0, отметки РЛС по
                     модели этапа 1 (--pd - вероятность обнаружения; в случаях с доплером ещё радиальная
                     скорость), прогноз и ошибки относительно GPS. --save-radar файл.csv - сохранить отметки в
                     формате --radar;
  --radar файл.csv - отметки РЛС: столбцы t (с), range_m, az_deg (от севера по часовой),
                     el_deg и, для случаев с доплером, vr_mps (радиальная скорость, м/с); пустая дальность -
                     пропуск. Обзоры - через SCAN_PERIOD. Положение РЛС - --radar-lat, --radar-lon,
                     --radar-alt. Истины нет, только прогноз.

Выход (--out, по умолчанию results/<прогон>/predict/<имя файла>):
  forecast.csv - прогнозы гибрида и IMM для каждого обзора и горизонта (ENU и широта,
                 долгота);
  report.html  - карта с ползунком по обзорам: отметки, истина, прогнозы гибрида и IMM;
                 таблица ошибок, график ошибки во времени;
  summary.txt  - ошибки (если есть истина), время обработки обзора, проверки.

Модель - пакет этапа 11: --bundle (по умолчанию artifacts/<прогон>/stage11/<случай>). Пакеты в git не хранятся.
--smoke - пакет смоука этапа 11: проверка программы на рабочем ПК, прогноз для диплома не годится.

Проверки (в summary.txt): пересчёт ENU -> широта и долгота совпадает с координатами лога;
прогноз причинный - по началу трассы он тот же, что по всей трассе.
"""
import argparse
import base64
import csv
import dataclasses
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from data import phases  # noqa: E402
from data.flight_logs import clean_log, read_log  # noqa: E402
from data.geo import enu_to_geodetic, geodetic_to_enu  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from pipeline.bundle import FinalPredictor  # noqa: E402
from radar.model import RadarTrack, converted_cov, observe, polar_to_enu, to_polar  # noqa: E402
from common.cases import CASES, config_override  # noqa: E402
from viz.plots import PHASE_COLORS, PHASE_RU  # noqa: E402

PERIOD_TOL = 0.05        # допустимое отклонение шага времени в файле отметок, доля периода
GEO_TOL_M = 1.0          # проверка пересчёта координат лога, м
CAUSAL_TOL_M = 1e-3      # проверка причинности: сеть в float32, итог зависит от состава батча на уровне 1e-6 м


def prefix(rt: RadarTrack, n: int) -> RadarTrack:
    """Первые n обзоров трассы."""
    cut = {f.name: getattr(rt, f.name)[:n] for f in dataclasses.fields(rt)
           if isinstance(getattr(rt, f.name), np.ndarray) and getattr(rt, f.name).ndim and f.name != "radar_pos"}
    return dataclasses.replace(rt, onsets=[o for o in rt.onsets if o < n], **cut)


# ============================================================================
# Вход
# ============================================================================

def from_log(path: Path, pd: float, log, sensor: dict = None) -> list:
    """Куски лога -> [(RadarTrack, точка отсчёта ENU (lat, lon, alt), проверка координат, м)]."""
    raw = read_log(path)
    tracks, stats = clean_log(raw)
    log.info("Лог %s: строк %d, кусков после очистки %d", path.name, raw.n_rows, len(tracks))
    out = []
    for tr in tracks:
        phases.annotate(tr)
        first = np.flatnonzero((raw.lat == tr.lat[0]) & (raw.lon == tr.lon[0]))
        alt0 = float(raw.alt[first[0]]) if len(first) else 0.0
        with config_override({"RADAR_PD": pd, **(sensor or {})}):
            rt = observe(tr)
        la, lo, _ = enu_to_geodetic(tr.pos, tr.lat[0], tr.lon[0], alt0)
        geo_err = float(np.max(np.hypot((la - tr.lat) * 111_320.0, (lo - tr.lon) * 111_320.0 * np.cos(
            np.radians(tr.lat[0])))))
        out.append((rt, (float(tr.lat[0]), float(tr.lon[0]), alt0), geo_err))
    return out


def from_radar(path: Path, lat: float, lon: float, alt: float, doppler: bool = False) -> RadarTrack:
    t, polar, vr = [], [], []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t"]))
            vr.append(float(row["vr_mps"]) if doppler and row.get("vr_mps", "").strip() else np.nan)
            if row["range_m"].strip():
                polar.append([float(row["range_m"]), np.radians(float(row["az_deg"])),
                              np.radians(float(row["el_deg"]))])
            else:
                polar.append([np.nan] * 3)
    t, polar = np.array(t), np.array(polar)
    steps = np.diff(t)
    if len(t) < 3 or np.abs(steps - config.SCAN_PERIOD).max() > PERIOD_TOL * config.SCAN_PERIOD:
        raise ValueError(f"{path}: обзоры должны идти через {config.SCAN_PERIOD} с (пропуск - пустая дальность)")
    detected = ~np.isnan(polar[:, 0])
    nan3 = np.full((len(t), 3), np.nan)
    vr = np.where(detected, np.array(vr), np.nan)
    if doppler and not np.isfinite(vr[detected]).any():
        raise ValueError(f"{path}: для случая с доплером нужен столбец vr_mps")
    return RadarTrack(path.stem, t, nan3, np.zeros(3), polar, polar_to_enu(polar), converted_cov(polar), detected,
                      np.full(len(t), PHASE_CODE["straight"], dtype=np.int8), [], vr=vr if doppler else None)


def save_radar(rt: RadarTrack, origin: tuple, path: Path) -> tuple:
    """
    Отметки в формате --radar; возвращает (широта, долгота, высота) РЛС. Модель РЛС
    задаёт углы в системе ENU начала трассы, а настоящая РЛС меряет их в своей: при
    РЛС в 8 км от начала трассы оси расходятся на 0,07°, и без пересчёта прогноз на
    карте сдвигается до 17 м. Поэтому отметки пересчитываются в систему РЛС.
    """
    la, lo, al = enu_to_geodetic(rt.radar_pos[None], *origin)
    radar = (float(la[0]), float(lo[0]), float(al[0]))
    polar = np.full_like(rt.polar, np.nan)
    det = rt.detected
    glat, glon, galt = enu_to_geodetic(rt.meas[det], *origin)
    polar[det] = to_polar(geodetic_to_enu(glat, glon, galt, *radar))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t", "range_m", "az_deg", "el_deg"] + (["vr_mps"] if rt.vr is not None else []))
        for k in range(len(rt.t)):
            r, az, el = polar[k]
            wr.writerow([f"{rt.t[k] - rt.t[0]:.3f}"] + ([f"{r:.3f}", f"{np.degrees(az) % 360:.7f}",
                                                         f"{np.degrees(el):.7f}"] if det[k] else ["", "", ""])
                        + ([f"{rt.vr[k]:.3f}" if det[k] and np.isfinite(rt.vr[k]) else ""] if rt.vr is not None else []))
    return radar


# ============================================================================
# Выход
# ============================================================================

def errors(rt: RadarTrack, fc: dict) -> dict:
    """ADE, FDE и ADE по участкам гибрида и IMM на одних обзорах (там, где есть прогноз гибрида)."""
    out = {}
    for name in ("imm", "hyb"):
        masked = np.full_like(fc[name], np.nan)
        masked[fc["ks"]] = fc[name][fc["ks"]]
        pairs = metrics.forecast_pairs(rt, masked)
        s = metrics.summarize(pairs)
        out[name] = {"ade": s["ade"], "fde": s["fde"], "n": s["n"]}
        for p in ("straight", "entry", "turn", "exit"):
            out[name][p] = metrics.summarize(metrics.select(pairs, pairs["phase"] == PHASE_CODE[p]))["ade"]
    return out


def to_latlon(xyz: np.ndarray, origin: tuple) -> np.ndarray:
    """(..., 3) ENU -> (..., 2) широта, долгота; nan сохраняется."""
    flat = xyz.reshape(-1, 3)
    out = np.full((len(flat), 2), np.nan)
    ok = ~np.isnan(flat).any(axis=1)
    if ok.any():
        la, lo, _ = enu_to_geodetic(flat[ok], *origin)
        out[ok] = np.stack([la, lo], axis=1)
    return out.reshape(xyz.shape[:-1] + (2,))


def write_csv(path: Path, items: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["кусок", "k", "t_s", "h", "t_прогноза_s", "метод", "east_m", "north_m", "lat", "lon"])
        for it in items:
            rt, fc = it["rt"], it["fc"]
            for name, label in (("hyb", "гибрид"), ("imm", "IMM")):
                ll = to_latlon(fc[name], it["origin"]) if it["origin"] else None
                for k in fc["ks"]:
                    for h in range(config.HORIZON_SCANS):
                        e, n = fc[name][k, h, :2]
                        geo = [f"{ll[k, h, 0]:.7f}", f"{ll[k, h, 1]:.7f}"] if ll is not None else ["", ""]
                        wr.writerow([it["name"], k, f"{rt.t[k] - rt.t[0]:.1f}", h + 1,
                                     f"{rt.t[k] - rt.t[0] + (h + 1) * config.SCAN_PERIOD:.1f}", label,
                                     f"{e:.2f}", f"{n:.2f}"] + geo)


def error_plot(it: dict) -> str:
    """PNG (base64) ошибки на горизонте H по времени с участками - только при известной истине."""
    rt, fc, h = it["rt"], it["fc"], config.HORIZON_SCANS
    ks = fc["ks"][fc["ks"] + h < len(rt.t)]
    if not len(ks) or np.isnan(rt.truth).all():
        return ""
    t = rt.t[ks] - rt.t[0]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for code_name, code in PHASE_CODE.items():
        if code_name in ("entry", "turn"):
            ax.fill_between(t, 0, 1, where=rt.phase[ks] == code, transform=ax.get_xaxis_transform(),
                            color=PHASE_COLORS[code_name], alpha=0.15, step="mid", label=PHASE_RU[code_name])
    for name, label, color in (("imm", "IMM", "#d32f2f"), ("hyb", "гибрид", "#2e7d32")):
        ax.plot(t, np.linalg.norm(fc[name][ks, h - 1, :2] - rt.truth[ks + h, :2], axis=1), lw=0.8, color=color,
                label=label)
    ax.set_xlabel("время, с")
    ax.set_ylabel(f"ошибка через {h * config.SCAN_PERIOD:.0f} с, м")
    ax.legend(fontsize=7, ncol=4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _round(a: np.ndarray):
    return [None if np.isnan(v).any() else [round(float(v[0]), 7), round(float(v[1]), 7)] for v in a]


def write_report(path: Path, title: str, items: list, err_table: str, info: str) -> None:
    chunks = []
    for it in items:
        rt, fc, origin = it["rt"], it["fc"], it["origin"]
        if not origin:
            continue
        has_truth = not np.isnan(rt.truth).all()
        k_total, h = len(rt.t), config.HORIZON_SCANS
        meas = rt.meas.copy()
        meas[~rt.detected] = np.nan
        fut = np.full((k_total, h, 3), np.nan)
        if has_truth:
            for hh in range(1, h + 1):
                fut[:k_total - hh, hh - 1] = rt.truth[hh:]
        err = {name: (np.linalg.norm(fc[name][:, -1, :2] - fut[:, -1, :2], axis=1) if has_truth
                      else np.full(k_total, np.nan)) for name in ("imm", "hyb")}
        chunks.append({
            "name": it["name"], "ks": [int(k) for k in fc["ks"]], "t": [round(float(v - rt.t[0]), 1) for v in rt.t],
            "plots": _round(to_latlon(meas, origin)),
            "truth": _round(to_latlon(rt.truth, origin)) if has_truth else [],
            "imm": [_round(p) for p in to_latlon(fc["imm"], origin)],
            "hyb": [_round(p) for p in to_latlon(fc["hyb"], origin)],
            "fut": [_round(p) for p in to_latlon(fut, origin)] if has_truth else [],
            "phase": [PHASE_RU[config.PHASES[c]] for c in rt.phase] if has_truth else [],
            "err_imm": [None if np.isnan(v) else round(float(v), 1) for v in err["imm"]],
            "err_hyb": [None if np.isnan(v) else round(float(v), 1) for v in err["hyb"]],
            "png": it["png"],
        })
    html = (_TEMPLATE.replace("__TITLE__", title).replace("__TABLE__", err_table).replace("__INFO__", info)
            .replace("__DATA__", json.dumps({"chunks": chunks, "period": config.SCAN_PERIOD}, ensure_ascii=False)))
    path.write_text(html, encoding="utf-8")


_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{color-scheme:light}
body{margin:0;font-family:system-ui,Arial,sans-serif;font-size:14px;color:#222;background:#fff}
header{padding:10px 16px;background:#f3f3f3}
h1{font-size:18px;margin:0 0 4px}
#map{width:100%;height:62vh}
.bar{padding:8px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.bar input[type=range]{flex:1;min-width:200px}
#state{font-variant-numeric:tabular-nums}
section{padding:8px 16px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}
th{background:#f3f3f3}
td:first-child,th:first-child{text-align:left}
.legend span{display:inline-block;margin-right:14px}
.sw{display:inline-block;width:14px;height:4px;vertical-align:middle;margin-right:4px}
img{max-width:100%}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div>__INFO__</div></header>
<div class="bar">
<label>Кусок <select id="chunk"></select></label>
<input type="range" id="k" min="0" max="0" value="0">
<label><input type="checkbox" id="follow" checked> следить за целью</label>
<span id="state"></span>
</div>
<div class="bar legend">
<span><i class="sw" style="background:#2e7d32"></i>прогноз гибрида</span>
<span><i class="sw" style="background:#d32f2f"></i>прогноз IMM</span>
<span id="lg-truth"><i class="sw" style="background:#000"></i>истина через 2-10 с</span>
<span><i class="sw" style="background:#888"></i>отметки РЛС</span>
</div>
<div id="map"></div>
<section><h2>Ошибки прогноза</h2>__TABLE__</section>
<section id="plot"></section>
<script>
const DATA = __DATA__;
const map = L.map("map");
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 19, attribution: "&copy; OpenStreetMap"}).addTo(map);
const layer = L.layerGroup().addTo(map);
const base = L.layerGroup().addTo(map);
const sel = document.getElementById("chunk"), slider = document.getElementById("k");
DATA.chunks.forEach((c, i) => sel.add(new Option(c.name, i)));
const ok = p => p !== null;
function line(pts, opt) { const p = pts.filter(ok); if (p.length > 1) L.polyline(p, opt).addTo(layer); }
function dots(pts, opt) { pts.filter(ok).forEach(p => L.circleMarker(p, opt).addTo(layer)); }
function showChunk() {
  const c = DATA.chunks[sel.value];
  base.clearLayers();
  const all = (c.truth.length ? c.truth : c.plots).filter(ok);
  if (c.truth.length) L.polyline(all, {color: "#000", weight: 1, opacity: 0.25}).addTo(base);
  if (all.length) map.fitBounds(all);
  slider.max = c.ks.length - 1; slider.value = 0;
  document.getElementById("lg-truth").hidden = !c.truth.length;
  document.getElementById("plot").innerHTML = c.png ? "<h2>Ошибка во времени</h2><img src='data:image/png;base64," + c.png + "'>" : "";
  draw();
}
function draw() {
  const c = DATA.chunks[sel.value];
  if (!c.ks.length) return;
  const k = c.ks[slider.value];
  layer.clearLayers();
  dots(c.plots.slice(Math.max(0, k - 20), k + 1), {radius: 2, color: "#888", weight: 1});
  const here = c.plots[k] || c.hyb[k][0];
  if (document.getElementById("follow").checked && here) map.setView(here, Math.max(map.getZoom(), 16), {animate: false});
  if (c.truth.length) {
    line([c.truth[k]].concat(c.fut[k]), {color: "#000", weight: 2, dashArray: "4 4"});
    dots(c.fut[k], {radius: 3, color: "#000", weight: 1});
  }
  line([here].concat(c.imm[k]), {color: "#d32f2f", weight: 2});
  dots(c.imm[k], {radius: 3, color: "#d32f2f", weight: 1});
  line([here].concat(c.hyb[k]), {color: "#2e7d32", weight: 3});
  dots(c.hyb[k], {radius: 4, color: "#2e7d32", weight: 1, fillOpacity: 0.8});
  let s = "обзор " + k + ", t = " + c.t[k] + " с";
  if (c.phase.length) s += ", участок: " + c.phase[k];
  if (c.err_hyb[k] !== null) s += " | ошибка через " + (5 * DATA.period) + " с: гибрид " + c.err_hyb[k] + " м, IMM " + c.err_imm[k] + " м";
  document.getElementById("state").textContent = s;
}
sel.onchange = showChunk; slider.oninput = draw;
// Размер карты известен только после разметки страницы: иначе fitBounds даёт масштаб всего мира
window.addEventListener("load", () => { map.invalidateSize(); showChunk(); });
</script>
</body>
</html>
"""


def error_table(rows: list) -> str:
    if not rows:
        return "<p>Истины нет (отметки реальной РЛС) - ошибки не считаются.</p>"
    cols = [("ade", "ADE"), ("fde", "FDE"), ("straight", "прямая"), ("entry", "вход"), ("turn", "маневр"),
            ("exit", "выход")]
    html = ["<table><tr><th>кусок</th><th>метод</th>" + "".join(f"<th>{c}, м</th>" for _, c in cols)
            + "<th>изм. ADE</th></tr>"]
    for name, e in rows:
        for m, label in (("imm", "IMM"), ("hyb", "гибрид")):
            cells = "".join("<td>-</td>" if np.isnan(e[m][c]) else f"<td>{e[m][c]:.1f}</td>" for c, _ in cols)
            gain = f"{e['hyb']['ade'] / e['imm']['ade'] - 1:+.1%}" if m == "hyb" else ""
            html.append(f"<tr><td>{name}</td><td>{label}</td>{cells}<td>{gain}</td></tr>")
    return "".join(html) + "</table>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогноз положения воздушного объекта: гибрид IMM + нейросеть")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", type=Path, help="лог полёта CSV (GPS - истина, отметки РЛС моделируются)")
    src.add_argument("--radar", type=Path, help="отметки РЛС CSV: t, range_m, az_deg, el_deg")
    parser.add_argument("--radar-lat", type=float, help="широта РЛС (с --radar), градусы")
    parser.add_argument("--radar-lon", type=float, help="долгота РЛС (с --radar), градусы")
    parser.add_argument("--radar-alt", type=float, default=0.0, help="высота РЛС (с --radar), м")
    parser.add_argument("--pd", type=float, default=config.RADAR_PD, help="вероятность обнаружения (с --log)")
    parser.add_argument("--save-radar", type=Path, help="с --log: сохранить отметки первого куска в формате --radar")
    parser.add_argument("--case", choices=CASES, default="dop_traj", help="случай доплера итоговой модели")
    parser.add_argument("--bundle", type=Path, help="пакет итоговой модели (этап 11)")
    parser.add_argument("--ensemble", action="store_true", help="среднее поправок всех seed итоговой модели")
    parser.add_argument("--out", type=Path, help="папка результатов")
    parser.add_argument("--smoke", action="store_true", help="пакет смоука этапа 11 (проверка программы)")
    args = parser.parse_args()

    log = setup_logging("predict" + ("_smoke" if args.smoke else ""))
    src_path = args.log or args.radar
    out_dir = args.out or config.RESULTS_DIR / "predict" / (src_path.stem + ("_smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = StageSummary(f"Прогноз: {src_path.name}" + (" (СМОУК-МОДЕЛЬ: не результат)" if args.smoke else ""))
    root = config.ARTIFACTS_DIR / "_smoke" if args.smoke else config.ARTIFACTS_DIR
    bundle_dir = args.bundle or root / "stage11" / args.case
    if not (bundle_dir / "config.json").exists():
        log.error("Нет пакета %s - нужен результат этапа 11 (домашний ПК) или --smoke", bundle_dir)
        return 2
    device = torch.device("cpu")
    pred = FinalPredictor(bundle_dir, device, ensemble=args.ensemble)
    log.info("Пакет %s (%s), окно %d обзоров, случай %s", bundle_dir, pred.name, pred.window, pred.case)

    if args.log:
        items = [{"name": f"кусок {i}", "rt": rt, "origin": origin, "geo_err": ge}
                 for i, (rt, origin, ge) in enumerate(from_log(args.log, args.pd, log, pred.sensor_override))]
        if not items:
            log.error("В логе не осталось кусков после очистки")
            return 1
        geo = max(it["geo_err"] for it in items)
        summary.check("Пересчёт ENU -> широта и долгота совпадает с координатами лога", geo <= GEO_TOL_M,
                      f"наибольшее расхождение {geo:.3f} м")
        if args.save_radar:
            lat, lon, alt = save_radar(items[0]["rt"], items[0]["origin"], args.save_radar)
            log.info("Отметки первого куска: %s; РЛС: --radar-lat %.8f --radar-lon %.8f --radar-alt %.2f",
                     args.save_radar, lat, lon, alt)
    else:
        if args.radar_lat is None or args.radar_lon is None:
            log.error("С --radar нужны --radar-lat и --radar-lon")
            return 2
        items = [{"name": args.radar.stem, "rt": from_radar(args.radar, args.radar_lat, args.radar_lon,
                                                               args.radar_alt, pred.case != "nodop"),
                  "origin": (args.radar_lat, args.radar_lon, args.radar_alt)}]

    t0, n_scans = time.perf_counter(), 0
    for it in items:
        it["fc"] = pred.run(it["rt"])
        n_scans += len(it["rt"].t)
    per_scan_ms = (time.perf_counter() - t0) / max(n_scans, 1) * 1000

    rt, fc = items[0]["rt"], items[0]["fc"]
    n = len(rt.t) // 2
    part = pred.run(prefix(rt, n))
    ks = fc["ks"][fc["ks"] < n]
    diff = float(np.nanmax(np.abs(part["hyb"][ks] - fc["hyb"][ks]), initial=0.0)) if len(ks) else float("nan")
    summary.check("Прогноз причинный: по началу трассы тот же, что по всей", len(ks) > 0 and diff < CAUSAL_TOL_M,
                  f"обзоров {len(ks)}, наибольшее расхождение {diff:.2e} м")

    rows, lines = [], []
    has_truth = args.log is not None
    for it in items:
        it["png"] = error_plot(it) if has_truth else ""
        if has_truth:
            e = errors(it["rt"], it["fc"])
            rows.append((it["name"], e))
            lines.append(f"{it['name']}: обзоров {len(it['rt'].t)}, пар {e['hyb']['n']}; ADE IMM {e['imm']['ade']:.1f} м, "
                         f"гибрид {e['hyb']['ade']:.1f} м ({e['hyb']['ade'] / e['imm']['ade'] - 1:+.1%}); "
                         f"вход в маневр {e['imm']['entry']:.1f} -> {e['hyb']['entry']:.1f} м")
    if lines:
        summary.section("Ошибки прогноза по горизонтали (все горизонты 2-10 с)", lines)
    det = np.concatenate([it["rt"].detected for it in items])
    info = (f"Модель: {pred.name}, окно {pred.window} обзоров ({pred.window * config.SCAN_PERIOD:.0f} с). "
            f"Обзоров: {n_scans}, пропусков: {1 - det.mean():.1%}. "
            f"Время обработки: {per_scan_ms:.2f} мс на обзор (процессор).")
    if args.smoke:
        info += " <b>Смоук-модель: прогноз для диплома не годится.</b>"
    summary.section("Обработка", [info.replace("<b>", "").replace("</b>", "")])

    write_csv(out_dir / "forecast.csv", items)
    write_report(out_dir / "report.html", f"Прогноз положения: {src_path.name}", items, error_table(rows), info)
    summary.write(out_dir / "summary.txt")
    log.info("Результаты: %s (forecast.csv, report.html, summary.txt)", out_dir)
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
