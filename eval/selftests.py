"""
Самопроверки на подставных данных с известным ответом (этап 1 версии 2): метрики, модели движения,
линейная экстраполяция, пересчёт отметок РЛС, бутстреп по полётам.
"""

import numpy as np

import config
from eval import bootstrap, metrics
from filters.extrapolation import cv_extrapolate
from filters.models import CA, CT, CV, VX, VY, W, X, Y
from radar.model import RadarTrack, converted_cov, polar_to_enu, to_polar


def _fake_radar_track(truth: np.ndarray) -> RadarTrack:
    k = len(truth)
    return RadarTrack("test", np.arange(k) * config.SCAN_PERIOD, truth, np.zeros(3), np.zeros((k, 3)),
                      truth.copy(), np.tile(np.eye(3), (k, 1, 1)), np.ones(k, bool),
                      np.zeros(k, np.int8), [])


def filters_selftest(summary) -> None:
    hz = config.HORIZON_SCANS
    k = 40
    truth = np.column_stack([np.arange(k) * 20.0, np.zeros(k), np.zeros(k)])
    rt = _fake_radar_track(truth)
    fc = np.zeros((k, hz, 3))
    for h in range(1, hz + 1):
        fc[: k - h, h - 1] = truth[h:] + np.array([3.0, 4.0, 0.0])
    s = metrics.summarize(metrics.forecast_pairs(rt, fc))
    summary.check("Метрики: известный ответ (ошибка 3, 4 м)",
                  abs(s["ade"] - 5) < 1e-9 and abs(s["rmse"] - 5) < 1e-9 and abs(s["mae"] - 3.5) < 1e-9
                  and abs(s["fde"] - 5) < 1e-9,
                  f"ADE {s['ade']:.3f}, RMSE {s['rmse']:.3f}, MAE {s['mae']:.3f}, FDE {s['fde']:.3f} "
                  f"(ожидалось 5, 5, 3.5, 5)")
    for h in range(1, hz + 1):
        fc[: k - h, h - 1] = truth[h:]
    s = metrics.summarize(metrics.forecast_pairs(rt, fc))
    summary.check("Метрики: точный прогноз", s["ade"] == 0 and abs(s["r2"] - 1) < 1e-12,
                  f"ADE {s['ade']}, R2 {s['r2']}")

    # Модели движения: CT на окружности, CV и CA на прямой - точный прогноз
    v, w, t = 20.0, 0.1, config.SCAN_PERIOD
    x = np.zeros(9)
    x[VX], x[W] = v, w
    model = CT()
    for _ in range(10):
        x = model.f(x, t)
    th = w * 10 * t
    exact = np.array([v / w * np.sin(th), v / w * (1 - np.cos(th))])
    err_ct = float(np.linalg.norm(x[[X, Y]] - exact))
    x = np.zeros(9)
    x[VX], x[VY] = 10.0, -5.0
    err_cv = float(np.linalg.norm(CV().f(x, 4.0)[[X, Y]] - [40.0, -20.0]))
    x = np.zeros(9)
    x[VX], x[6] = 10.0, 1.0
    err_ca = float(abs(CA().f(x, 4.0)[X] - (40.0 + 0.5 * 16.0)))
    summary.check("Модели движения: известный ответ", max(err_ct, err_cv, err_ca) < 1e-6,
                  f"ошибка CT на окружности {err_ct:.2e} м, CV {err_cv:.2e} м, CA {err_ca:.2e} м")

    # Линейная экстраполяция на прямой без шума
    rt = _fake_radar_track(truth)
    fc = cv_extrapolate(rt)
    s = metrics.summarize(metrics.forecast_pairs(rt, fc))
    summary.check("Линейная экстраполяция: прямая без шума", s["ade"] < 1e-6, f"ADE {s['ade']:.2e} м")

    # Пересчёт отметок: туда-обратно точно; ковариация согласована с шумом (NEES около 3)
    rng = np.random.default_rng(0)
    rel = np.array([[2000.0, 1500.0, 300.0]])
    polar = to_polar(rel)
    back = float(np.abs(polar_to_enu(polar) - rel).max())
    sig = np.array([config.RADAR_SIGMA_R, np.radians(config.RADAR_SIGMA_AZ_DEG),
                    np.radians(config.RADAR_SIGMA_EL_DEG)])
    noisy = polar + rng.standard_normal((20000, 3)) * sig
    d = polar_to_enu(noisy) - rel
    cov = converted_cov(noisy)
    nees = np.einsum("ni,nij,nj->n", d, np.linalg.inv(cov), d)
    summary.check("Пересчёт отметок: точность и NEES", back < 1e-6 and 2.85 <= nees.mean() <= 3.15,
                  f"ошибка туда-обратно {back:.1e} м, средний NEES {nees.mean():.3f} (ожидалось около 3)")


def bootstrap_selftest(summary) -> None:
    """Бутстреп по полётам: постоянная разность -> вырожденный интервал; шум нулевого среднего -> накрывает 0."""
    from common.rng import make_rng

    rng = np.random.default_rng(0)
    n = 60
    counts = rng.integers(200, 400, n).astype(float)
    base_mean = rng.normal(40, 8, n)
    sums_b = base_mean * counts
    res = bootstrap.paired_difference((base_mean - 2.0) * counts, counts, sums_b, counts, n_boot=500,
                                      rng=make_rng("bootstrap"))
    noise = rng.normal(0, 3, n)
    res0 = bootstrap.paired_difference((base_mean + noise) * counts, counts, sums_b, counts, n_boot=500,
                                       rng=make_rng("bootstrap"))
    ok = (abs(res["diff"] + 2.0) < 1e-9 and res["better"] and abs(res["hi"] + 2.0) < 1e-6
          and res0["lo"] < 0 < res0["hi"] and not bootstrap.selection_rule(res0))
    summary.check("Бутстреп по полётам: известный ответ", ok,
                  f"выигрыш 2 м во всех полётах: разность {res['diff']:.3f}, интервал [{res['lo']:.3f}, {res['hi']:.3f}]; "
                  f"шум: интервал [{res0['lo']:.2f}, {res0['hi']:.2f}] накрывает 0")
