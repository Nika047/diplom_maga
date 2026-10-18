"""
Интерактивная карта реальных полётов (Leaflet + OpenStreetMap).

Самодостаточный HTML: Leaflet подключается с CDN, тайлы грузятся в браузере при
открытии. Каждый полёт - линия цвета своего типа аппарата, слои по типам можно
включать и выключать, по щелчку - название полёта, выборка, длительность.
Идея взята из visualization/map_html.py старого проекта, без слоя целей.
"""

import json
from pathlib import Path

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
           "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body{margin:0;height:100%;font-family:system-ui,Arial,sans-serif}#map{width:100%;height:100vh}</style>
</head>
<body>
<div id="map"></div>
<script>
const DATA = __DATA__;
const map = L.map("map");
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {maxZoom: 18, attribution: "&copy; OpenStreetMap"}).addTo(map);
const groups = {};
const bounds = [];
for (const f of DATA.flights) {
    if (!groups[f.group]) groups[f.group] = L.layerGroup().addTo(map);
    const line = L.polyline(f.latlon, {color: f.color, weight: 2, opacity: 0.8});
    line.bindPopup(f.popup);
    line.addTo(groups[f.group]);
    for (const p of f.latlon) bounds.push(p);
}
L.control.layers(null, groups, {collapsed: false}).addTo(map);
if (bounds.length) map.fitBounds(bounds); else map.setView([59.9, 30.3], 9);
</script>
</body>
</html>
"""


def write_flights_map(tracks: list, splits: dict, path: Path, title: str) -> None:
    """Карта треков с lat/lon. splits: {имя лога: выборка}."""
    groups = sorted({tr.aircraft for tr in tracks})
    color = {g: _COLORS[i % len(_COLORS)] for i, g in enumerate(groups)}
    flights = []
    for tr in tracks:
        if tr.lat is None:
            continue
        split = splits.get(tr.log_file, "-")
        popup = (f"<b>{tr.flight_id}</b><br>тип: {tr.aircraft}<br>выборка: {split}"
                 f"<br>длительность: {tr.duration_s / 60:.1f} мин<br>путь: {tr.path_m / 1000:.1f} км")
        latlon = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(tr.lat, tr.lon)]
        flights.append({"group": f"{tr.aircraft}", "color": color[tr.aircraft],
                        "popup": popup, "latlon": latlon})
    html = _TEMPLATE.replace("__TITLE__", title).replace(
        "__DATA__", json.dumps({"flights": flights}, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
