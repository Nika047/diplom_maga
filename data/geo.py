"""
Перевод координат WGS84 (широта, долгота, высота) в локальную систему ENU.

Строгий пересчёт через геоцентрические координаты ECEF, без приближения
плоской Земли. Проверка: точка на 1 км севернее начала даёт y около 1000 м.
"""

import numpy as np

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt_m) * np.cos(lat) * np.cos(lon)
    y = (n + alt_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + alt_m) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def geodetic_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray,
                    lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """(N,) градусы и метры -> (N, 3) ENU в метрах относительно (lat0, lon0, alt0)."""
    ecef = geodetic_to_ecef(np.asarray(lat_deg, float), np.asarray(lon_deg, float),
                            np.asarray(alt_m, float))
    ref = geodetic_to_ecef(np.array(lat0), np.array(lon0), np.array(alt0))
    d = ecef - ref
    la, lo = np.radians(lat0), np.radians(lon0)
    east = -np.sin(lo) * d[:, 0] + np.cos(lo) * d[:, 1]
    north = (-np.sin(la) * np.cos(lo) * d[:, 0] - np.sin(la) * np.sin(lo) * d[:, 1]
             + np.cos(la) * d[:, 2])
    up = (np.cos(la) * np.cos(lo) * d[:, 0] + np.cos(la) * np.sin(lo) * d[:, 1]
          + np.sin(la) * d[:, 2])
    return np.stack([east, north, up], axis=1)


def local_step_m(lat1, lon1, lat2, lon2):
    """Горизонтальное расстояние между соседними точками (м), приближение для малых шагов."""
    r = _WGS84_A
    lat_mid = np.radians((np.asarray(lat1) + np.asarray(lat2)) / 2.0)
    dx = r * np.cos(lat_mid) * np.radians(np.asarray(lon2) - np.asarray(lon1))
    dy = r * np.radians(np.asarray(lat2) - np.asarray(lat1))
    return np.hypot(dx, dy)
