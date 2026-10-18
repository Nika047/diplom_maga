"""
Набор данных корректора: отметки РЛС -> прогон IMM -> окна (data/windows.py).

Прогон IMM по всей выборке train занимает минуты, поэтому результат кэшируется в
cache/stage2/<имя>.npz. Рядом - <имя>.json с параметрами, от которых зависят
признаки (РЛС, IMM, окно, список трасс). При расхождении кэш пересобирается сам:
устаревший кэш после смены параметров IMM (например, на этапе 4) не используется.

Нормализация - по train: признаки - вычитание среднего и деление на СКО; цель -
только деление на СКО, без сдвига, чтобы нулевой выход сети оставался нулевой
поправкой (гибрид = IMM).
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from common.logging_setup import get_logger
from data.windows import build_windows
from filters.imm import IMM
from radar.model import observe

FEATURES_VERSION = 1  # увеличить при изменении состава признаков в data/windows.py


@dataclass
class WindowSet:
    x: np.ndarray        # (N, W, F)
    y: np.ndarray        # (N, H, 2)
    track: np.ndarray    # (N,) индекс трассы в flight_ids
    k: np.ndarray        # (N,) обзор
    psi: np.ndarray      # (N,) курс IMM в обзоре k
    flight_ids: list

    def __len__(self) -> int:
        return len(self.x)


def signature(flight_ids: list) -> dict:
    keys = ("SCAN_PERIOD", "RADAR_SIGMA_R", "RADAR_SIGMA_AZ_DEG", "RADAR_SIGMA_EL_DEG", "RADAR_PD",
            "RADAR_DIST_RANGE", "RADAR_BELOW_MIN_Z", "HORIZON_SCANS", "WARMUP_SCANS",
            "FILTER_SIGMA_A_CV", "FILTER_SIGMA_J_CA", "FILTER_SIGMA_A_CT", "FILTER_SIGMA_W_CT",
            "FILTER_SIGMA_A_Z", "IMM_MODELS", "IMM_P_STAY", "IMM_MU0", "INIT_SIGMA_A", "INIT_SIGMA_W",
            "WINDOW_SCANS", "RANDOM_SEED")
    sig = {key: getattr(config, key) for key in keys}
    sig = json.loads(json.dumps(sig))  # кортежи -> списки, как после чтения из файла
    sig["FEATURES_VERSION"] = FEATURES_VERSION
    sig["tracks"] = hashlib.sha256("\n".join(flight_ids).encode("utf-8")).hexdigest()[:16]
    sig["n_tracks"] = len(flight_ids)
    return sig


def build(tracks: list, imm: IMM = None) -> tuple:
    """Окна по списку треков. Возвращает (WindowSet, список RadarTrack, список FilterRun)."""
    imm = imm or IMM()
    xs, ys, tr_idx, ks, psis, rts, runs = [], [], [], [], [], [], []
    for i, tr in enumerate(tracks):
        rt = observe(tr)
        run = imm.run(rt)
        w = build_windows(rt, run)
        rts.append(rt)
        runs.append(run)
        xs.append(w["x"])
        ys.append(w["y"])
        ks.append(w["k"])
        psis.append(w["psi"])
        tr_idx.append(np.full(len(w["k"]), i))
    ws = WindowSet(np.concatenate(xs), np.concatenate(ys), np.concatenate(tr_idx), np.concatenate(ks),
                   np.concatenate(psis), [tr.flight_id for tr in tracks])
    return ws, rts, runs


def load_or_build(name: str, tracks: list, cache_dir: Path, rebuild: bool = False) -> WindowSet:
    """Окна из кэша, если параметры совпадают, иначе сборка и запись кэша."""
    log = get_logger()
    flight_ids = [tr.flight_id for tr in tracks]
    sig = signature(flight_ids)
    npz, meta = cache_dir / f"{name}.npz", cache_dir / f"{name}.json"
    if not rebuild and npz.exists() and meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")) == sig:
            data = np.load(npz)
            log.info("Окна %s: из кэша %s (%d примеров)", name, npz, len(data["x"]))
            return WindowSet(data["x"], data["y"], data["track"], data["k"], data["psi"], flight_ids)
        log.info("Окна %s: параметры изменились, кэш пересобирается", name)
    log.info("Окна %s: прогон РЛС и IMM по %d трассам...", name, len(tracks))
    ws, _, _ = build(tracks)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz, x=ws.x, y=ws.y, track=ws.track, k=ws.k, psi=ws.psi)
    meta.write_text(json.dumps(sig, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Окна %s: %d примеров, записаны в %s", name, len(ws), npz)
    return ws


@dataclass
class Normalizer:
    x_mean: np.ndarray   # (F,)
    x_std: np.ndarray    # (F,)
    y_std: np.ndarray    # (2,)

    @classmethod
    def fit(cls, ws: WindowSet) -> "Normalizer":
        x = ws.x.reshape(-1, ws.x.shape[-1]).astype(np.float64)
        y = ws.y.reshape(-1, 2).astype(np.float64)
        x_std = x.std(axis=0)
        x_std[x_std < 1e-6] = 1.0
        return cls(x.mean(axis=0).astype(np.float32), x_std.astype(np.float32),
                   np.sqrt((y ** 2).mean(axis=0)).astype(np.float32))

    def x(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def y(self, y: np.ndarray) -> np.ndarray:
        return (y / self.y_std).astype(np.float32)

    def y_inv(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std

    def as_dict(self) -> dict:
        return {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(), "y_std": self.y_std.tolist()}
