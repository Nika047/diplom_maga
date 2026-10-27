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


def window_selftest(summary, tracks: list, cd) -> None:
    """
    Окна и корректор (случай cd): поворот в систему курса, причинность окон (признаки по началу трассы =
    по полной, в том числе с доплером), нулевая поправка = IMM, смоук модели (формы, нулевой выход, градиенты).
    """
    import dataclasses

    import torch

    from common.cases import config_override
    from data.windows import apply_correction, build_windows, n_features, to_global, to_local
    from filters.imm import IMM
    from models import build_corrector
    from radar.model import observe

    v = np.array([[1.0, 0.0]])
    loc = to_local(v, np.array([np.pi / 2]))
    back = to_global(loc, np.array([np.pi / 2]))
    summary.check("Система курса: известный ответ", np.allclose(loc, [[0.0, 1.0]]) and np.allclose(back, v),
                  f"восток при курсе 90 град -> {loc.round(6).tolist()}, обратно {back.round(6).tolist()}")
    with config_override(cd.override()):
        tr = max(tracks, key=len)
        rt = observe(tr)
        imm = IMM()
        run = imm.run(rt)
        full = build_windows(rt, run, with_target=False)
        k0 = int(full["k"][len(full["k"]) // 2])
        cut = dataclasses.replace(rt, t=rt.t[:k0 + 1], truth=rt.truth[:k0 + 1], polar=rt.polar[:k0 + 1],
                                  meas=rt.meas[:k0 + 1], meas_cov=rt.meas_cov[:k0 + 1],
                                  detected=rt.detected[:k0 + 1], phase=rt.phase[:k0 + 1],
                                  onsets=[k for k in rt.onsets if k <= k0],
                                  vr=None if rt.vr is None else rt.vr[:k0 + 1])
        part = build_windows(cut, imm.run(cut), with_target=False)
        common = np.intersect1d(part["k"], full["k"])
        same = np.array_equal(part["x"][np.isin(part["k"], common)], full["x"][np.isin(full["k"], common)])
        summary.check("Окна причинные: признаки по началу трассы = по полной", same and len(common) > 0,
                      f"случай {cd.case}, трасса {tr.flight_id}, обрезка на обзоре {k0}, общих окон {len(common)}, "
                      f"признаков {full['x'].shape[-1]}")
        w = build_windows(rt, run)
        hyb = apply_correction(run.forecast, w["k"], w["psi"], np.zeros_like(w["y"]))
        a = metrics.summarize(metrics.forecast_pairs(rt, run.forecast))
        b = metrics.summarize(metrics.forecast_pairs(rt, hyb))
        summary.check("Нулевая поправка = IMM", np.array_equal(hyb, run.forecast, equal_nan=True) and a == b,
                      f"ADE IMM {a['ade']:.3f} м, гибрида с нулевой поправкой {b['ade']:.3f} м")
        nf = full["x"].shape[-1]
    torch.manual_seed(config.RANDOM_SEED)
    model = build_corrector("GRU", nf, config.HORIZON_SCANS)
    x = torch.randn(8, config.WINDOW_SCANS, nf)
    out = model(x)
    ((out - torch.ones_like(out)) ** 2).mean().backward()
    grads = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    summary.check("Модель: формы, нулевой выход до обучения, градиенты",
                  tuple(out.shape) == (8, config.HORIZON_SCANS, 2) and float(out.abs().max()) == 0.0 and grads > 0,
                  f"выход {tuple(out.shape)}, max|выход| {float(out.abs().max())}, сумма |градиентов| {grads:.3f}, "
                  f"параметров {sum(p.numel() for p in model.parameters())}, признаков {nf} "
                  f"(n_features без доплера {n_features(len(config.IMM_MODELS))})")
