"""
Этап 9 (версия 2): анализ итоговых моделей случая.

Итоговая модель - состояние final_<случай> (этап 7), ансамбль - этап 8. Разделы (можно выбрать --only):
  gaps   - пропуски отметок: P_d из PD_LEVELS без переобучения (CV, KF, IMM, итоговая модель, ансамбль её seed);
  gate   - строб с покрытием GATE_COVERAGE: эллипс хи-квадрат IMM, конформные IMM, круг и эллипс вокруг
           прогноза сети, круг по разбросу ансамбля (радиус пропорционален разбросу участников);
  noise  - шум РЛС: NOISE_SCALES; IMM, итоговая модель при штатном шуме и корректор основы, переобученный
           при этом шуме; разложение ошибки на шум РЛС, метод и непредсказуемость (оракул, filters/oracle.py);
  doppler- точность доплера 0,5 / 1 / 2 м/с (только случаи с доплером): IMM и корректор основы, переобученный;
  synth  - отдельный сгенерированный test (не входит в обучение и разбиение), тот же состав сценариев;
  react  - быстрота реакции на маневр: вероятность модели CT в первых обзорах после начала и восстановление
           динамической ошибки.
Переобучение (шум, доплер) - корректор основы с гиперпараметрами основы, 1 seed: итоговая модель с гипотезами
не переобучается (оговорка в тексте). Сети берутся из хранилища моделей, повторный запуск продолжает.

  .venv\\Scripts\\python.exe run_stage9.py [--case ...] [--only gaps,gate,noise,doppler,synth,react] [--smoke]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from common import cases  # noqa: E402
from common.cases import config_override  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import generator  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from eval import metrics  # noqa: E402
from eval.compare import mask_forecast  # noqa: E402
from filters.imm import IMM  # noqa: E402
from filters.oracle import best_of, oracle_forecast  # noqa: E402
from pipeline import data as pdata  # noqa: E402
from pipeline import evaluation, hypotheses as hyp, predictors  # noqa: E402
from pipeline.registry import Registry  # noqa: E402
from pipeline.runs import fit_and_eval  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from viz.filter_plots import method_color  # noqa: E402
from viz.plots import _save  # noqa: E402

STAGE = "stage9"
SECTIONS = ("gaps", "gate", "noise", "doppler", "synth", "react")
SIGMAS_VR = (0.5, 1.0, 2.0)
REGIONS = ("IMM, эллипс хи-квадрат", "IMM, эллипс конформный", "сеть, круг конформный",
           "сеть, эллипс IMM конформный", "ансамбль, круг по разбросу")


def add_args(parser) -> None:
    parser.add_argument("--only", default=",".join(SECTIONS))


def noise_override(s: float) -> dict:
    return {"RADAR_SIGMA_R": config.RADAR_SIGMA_R * s, "RADAR_SIGMA_AZ_DEG": config.RADAR_SIGMA_AZ_DEG * s,
            "RADAR_SIGMA_EL_DEG": config.RADAR_SIGMA_EL_DEG * s}


def cell(vals) -> str:
    return evaluation.cell(vals)


class Final:
    """Итоговая модель случая: окна, сети по seed, поправки на любом варианте данных."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        fin = ctx.state("final")
        self.method = fin["method"]
        ex = ctx.registry.data[self.method][str(ctx.registry.seeds(self.method)[0])]["extra"]
        self.bc = hyp.BaseConfig(ex["corrector"], ex["window"], ex["net_input"], ex["hp"], ex.get("params", {}))
        self.base_method = fin["base_method"]
        self.seeds = ctx.registry.seeds(self.method)
        self.cd = ctx.casedata(window=self.bc.window, net_input=self.bc.net_input)

    def corr(self, cd, names=None) -> list:
        return [predictors.corrections(self.ctx, self.method, s, cd, names) for s in self.seeds]

    def methods(self, cd, names) -> dict:
        """{"сеть": [отчёты по seed], "ансамбль seed": отчёт} на наборах names в cd."""
        corrs = self.corr(cd, names)
        per_seed = [{n: evaluation.summarize_method(cd.eval(n)[1], evaluation.hybrid_forecasts(cd, n, c[n]))
                     for n in names} for c in corrs]
        ens = predictors.mean_corr(corrs)
        ens_rep = {n: evaluation.summarize_method(cd.eval(n)[1], evaluation.hybrid_forecasts(cd, n, ens[n]))
                   for n in names}
        return {"net": per_seed, "ens": ens_rep, "corrs": corrs, "ens_corr": ens}


def classic_reports(cd, names) -> dict:
    out = {}
    for n in names:
        fcs = evaluation.classic_forecasts(cd, n)
        out[n] = {m: evaluation.summarize_method(cd.eval(n)[1], fc) for m, fc in fcs.items()}
    return out


# ============================================================================
# Пропуски отметок
# ============================================================================

def section_gaps(ctx: Ctx, fin: Final) -> None:
    names = config.EVAL_SETS
    levels = (1.0, 0.8) if ctx.smoke else config.PD_LEVELS
    table, lines = {}, []
    for pd_ in levels:
        cdp = fin.cd.variant(extra={"RADAR_PD": pd_})
        cl, nets = classic_reports(cdp, names), fin.methods(cdp, names)
        table[pd_] = {n: {**cl[n], "сеть": {k: np.mean([r[n][k] for r in nets["net"]]) for k in ("ade", "entry", "turn")},
                          "ансамбль seed": nets["ens"][n]} for n in names}
        if pd_ == levels[0]:
            miss = float(1.0 - np.mean([rt.detected.mean() for rt in cdp.eval("test")[1]]))
            ctx.summary.check("P_d = 1: пропусков нет", miss == 0.0, f"доля пропусков {miss:.3f}")
        elif pd_ < 1.0:
            miss = float(1.0 - np.mean([rt.detected.mean() for rt in cdp.eval("test")[1]]))
            ctx.summary.check(f"P_d = {pd_}: доля пропусков соответствует", abs(miss - (1 - pd_)) <= config.PD_MISS_TOL,
                              f"доля пропусков {miss:.3f} при ожидаемой {1 - pd_:.3f}")
    for n in names:
        lines = [f"{'P_d':>5s} " + "".join(f"{m:>16s}" for m in ("IMM", "KF", "сеть", "ансамбль seed"))]
        for pd_ in levels:
            t = table[pd_][n]
            lines.append(f"{pd_:5.1f} " + "".join(f"{t[m]['ade']:16.1f}" for m in ("IMM", "KF", "сеть", "ансамбль seed")))
        ctx.summary.section(f"{n}: ADE (м) при пропусках отметок; сети не переобучались", lines)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ("IMM", "KF", "сеть", "ансамбль seed"):
        ax.plot(levels, [table[p]["test"][m]["ade"] for p in levels], "-o", ms=4, color=method_color(m if m != "сеть" else "IMM+GRU"), label=m)
    ax.set_xlabel("вероятность обнаружения P_d")
    ax.set_ylabel("ADE test, м")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.set_title(f"Пропуски отметок: {cases.CASE_TITLES[ctx.case]}", fontsize=10)
    _save(fig, ctx.out_dir / "gaps.png")


# ============================================================================
# Строб
# ============================================================================

def gate_pairs(rts: list, fcs: list, runs_cov: list, spread: list = None) -> dict:
    """Пары (k, h): ошибка (N, 2), ковариация IMM (N, 2, 2), разброс ансамбля (N,), горизонт, участок."""
    rows = {"err": [], "cov": [], "h": [], "phase": [], "spread": []}
    for i, (rt, fc, rc) in enumerate(zip(rts, fcs, runs_cov)):
        for h in range(1, config.HORIZON_SCANS + 1):
            k = np.arange(config.WARMUP_SCANS, len(rt.t) - h)
            k = k[~np.isnan(fc[k, h - 1, 0])]
            if not len(k):
                continue
            rows["err"].append(fc[k, h - 1, :2] - rt.truth[k + h, :2])
            rows["cov"].append(rc.forecast_cov[k, h - 1][:, :2, :2])
            rows["h"].append(np.full(len(k), h))
            rows["phase"].append(rt.phase[k])
            rows["spread"].append(spread[i][k, h - 1] if spread is not None else np.ones(len(k)))
    return {key: np.concatenate(val) for key, val in rows.items()}


def region_params(p_imm: dict, p_net: dict, p_ens: dict, coverage: float) -> dict:
    """Пороги по горизонтам, калибровка на val (split conformal)."""
    out = {r: {} for r in REGIONS}
    for h in range(1, config.HORIZON_SCANS + 1):
        si, sn, se = p_imm["h"] == h, p_net["h"] == h, p_ens["h"] == h
        out[REGIONS[0]][h] = 5.991 if coverage == 0.95 else float("nan")
        out[REGIONS[1]][h] = metrics.conformal_quantile(metrics.mahalanobis2(p_imm["err"][si], p_imm["cov"][si]), coverage)
        out[REGIONS[2]][h] = metrics.conformal_quantile(np.linalg.norm(p_net["err"][sn], axis=1), coverage)
        out[REGIONS[3]][h] = metrics.conformal_quantile(metrics.mahalanobis2(p_net["err"][sn], p_net["cov"][sn]), coverage)
        out[REGIONS[4]][h] = metrics.conformal_quantile(
            np.linalg.norm(p_ens["err"][se], axis=1) / (p_ens["spread"][se] + 1.0), coverage)
    return out


def apply_region(name: str, thr: dict, p: dict) -> tuple:
    q = np.array([thr[name][int(h)] for h in p["h"]])
    if name == REGIONS[2]:
        return np.linalg.norm(p["err"], axis=1) <= q, np.pi * q ** 2
    if name == REGIONS[4]:
        r = q * (p["spread"] + 1.0)
        return np.linalg.norm(p["err"], axis=1) <= r, np.pi * r ** 2
    det = np.linalg.det(p["cov"])
    return metrics.mahalanobis2(p["err"], p["cov"]) <= q, np.pi * q * np.sqrt(det)


def section_gate(ctx: Ctx, fin: Final) -> None:
    from pipeline.predictors import forecast_lists
    from run_stage8 import spread as ens_spread

    cd, coverage = fin.cd, config.GATE_COVERAGE
    names = config.EVAL_SETS
    ens_state = ctx.state("ens")
    runs_cov = {}
    with config_override(cd.override()):
        imm = IMM()
        for n in names:
            runs_cov[n] = [imm.run(rt, with_cov=True) for rt in cd.eval(n)[1]]
    same = max(float(np.nanmax(np.abs(a.forecast - b.forecast), initial=0.0))
               for n in names for a, b in zip(cd.eval(n)[2], runs_cov[n]))
    ctx.summary.check("Ковариация не меняет прогноз IMM", same == 0.0, f"наибольшее расхождение {same:.3g} м")
    q = np.array([1.0, 2.0, 3.0])
    ctx.summary.check("Конформный квантиль: известный ответ", metrics.conformal_quantile(q, 0.5) == 2.0, "ранг ceil((n+1)p)")
    imm_fc = {n: evaluation.imm_forecasts(cd, n) for n in names}
    net_fc = forecast_lists(cd, fin.corr(cd)[0])
    ens_members = [tuple(m) for m in ens_state["members"]] if ens_state else [(fin.method, s) for s in fin.seeds]
    ens_fc = {n: [np.mean(x, axis=0) for x in zip(*[ctx.registry.forecasts(m, s)[n] for m, s in ens_members])] for n in names}
    spread = {n: ens_spread(ens_members, n, ctx.registry) for n in names}
    pi = {n: gate_pairs(cd.eval(n)[1], imm_fc[n], runs_cov[n]) for n in names}
    pn = {n: gate_pairs(cd.eval(n)[1], net_fc[n], runs_cov[n]) for n in names}
    pe = {n: gate_pairs(cd.eval(n)[1], ens_fc[n], runs_cov[n], spread[n]) for n in names}
    eig = min(float(np.linalg.eigvalsh(p["cov"]).min()) for p in pi.values())
    ctx.summary.check("Ковариации прогноза IMM положительно определены", eig > 0.0, f"наименьшее собственное {eig:.3g} м^2")
    thr = region_params(pi["val"], pn["val"], pe["val"], coverage)
    reports = {}
    for n in names:
        reports[n] = {}
        for region in REGIONS:
            p = pi[n] if region.startswith("IMM,") else (pe[n] if region.startswith("ансамбль") else pn[n])
            inside, area = apply_region(region, thr, p)
            reports[n][region] = metrics.gate_report(inside, area, p["phase"], p["h"])
        lines = [f"{'строб':30s} {'покрытие':>9s} {'прямая':>8s} {'вход':>8s} {'маневр':>8s} {'выход':>8s} {'радиус, м':>10s}"]
        for region in REGIONS:
            r = reports[n][region]
            lines.append(f"{region:30s} {r['coverage']:9.3f} {r['coverage_straight']:8.3f} {r['coverage_entry']:8.3f} "
                         f"{r['coverage_turn']:8.3f} {r['coverage_exit']:8.3f} {r['radius_median']:10.1f}")
        ctx.summary.section(f"{n}: покрытие и размер строба (заданное {coverage:.0%})", lines)
    low = min(reports["val"][r]["coverage"] for r in REGIONS[1:])
    ctx.summary.check("На калибровочном наборе (val) покрытие не ниже заданного", low >= coverage - 1e-9,
                      f"наименьшее покрытие конформных стробов {low:.4f}")
    # разброс ансамбля предсказывает ошибку
    e = np.linalg.norm(pe["test"]["err"], axis=1)
    sp = pe["test"]["spread"]
    top = sp >= np.quantile(sp, 0.75)
    ctx.summary.section("Разброс ансамбля как оценка неопределённости (test)", [
        f"корреляция Спирмена |ошибка| - разброс: {float(np.corrcoef(np.argsort(np.argsort(e)), np.argsort(np.argsort(sp)))[0, 1]):.3f}",
        f"ошибка при разбросе в верхней четверти {e[top].mean():.1f} м, в остальных {e[~top].mean():.1f} м"])
    hs = np.arange(1, config.HORIZON_SCANS + 1)
    for key, ylabel, fname in (("coverage", "покрытие", "gate_coverage.png"), ("radius", "радиус, м", "gate_radius.png")):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for region in REGIONS:
            ax.plot(hs * config.SCAN_PERIOD, [metrics_h(reports["test"][region], key, h) for h in hs], "-o", ms=4,
                    label=region, color=method_color(region))
        if key == "coverage":
            ax.axhline(coverage, color="k", lw=0.8, ls="--")
        ax.set_xlabel("горизонт прогноза, с")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.set_title(f"Строб, test: {cases.CASE_TITLES[ctx.case]}", fontsize=10)
        _save(fig, ctx.out_dir / fname)


def metrics_h(report: dict, key: str, h: int) -> float:
    return report[f"{key}_h{h}"]


# ============================================================================
# Шум РЛС, разложение ошибки
# ============================================================================

def retrain(ctx: Ctx, fin: Final, cdv, tag: str, tracks_tr: list) -> dict:
    """Корректор основы, переобученный на вариантe данных cdv; 1 seed. Возвращает поправки по наборам оценки."""
    bc = fin.bc
    base_ex = ctx.registry.data[fin.base_method][str(ctx.registry.seeds(fin.base_method)[0])]["extra"]
    ws_tr = cdv.windows(f"main_{tag}", tracks_tr)
    out = fit_and_eval(ctx, cdv, STAGE, f"{fin.base_method.split('/', 1)[1]} {tag}", bc.corrector, "hybrid", ws_tr,
                       {k: v for k, v in base_ex["params"].items() if k in ("lr", "batch", "weight_decay")},
                       bc.hp, bc.window, seeds=ctx.st.seeds[:1], overfit=False)
    return out[0]


def section_noise(ctx: Ctx, fin: Final) -> None:
    names = config.EVAL_SETS
    scales = (1.0, 0.25) if ctx.smoke else config.NOISE_SCALES
    tracks_tr = pdata.train_tracks(ctx.sets)
    rows, curves = {}, {}
    for s in scales:
        cds = fin.cd.variant(extra=noise_override(s)) if s != 1.0 else fin.cd
        cl = classic_reports(cds, names)
        nets = fin.methods(cds, names)
        row = {n: {"IMM": cl[n]["IMM"], "итоговая (штатный шум)": {k: float(np.mean([r[n][k] for r in nets["net"]]))
                                                                   for k in ("ade", "entry", "turn")}} for n in names}
        if s != 1.0:
            res = retrain(ctx, fin, cds, f"шум x{s:g}", tracks_tr)
            for n in names:
                row[n]["корректор основы, переобучен"] = evaluation.summarize_method(cds.eval(n)[1], res["fcs"][n])
        else:
            base_rep = {n: {k: ctx.registry.metric(fin.base_method, n, k)[0] for k in ("ade", "entry", "turn")}
                        for n in names}
            for n in names:
                row[n]["корректор основы, переобучен"] = base_rep[n]
        rows[s] = row
        # оракул на тех же трассах: лучший из CV и CT по точной истине
        for n in names:
            rts = cds.eval(n)[1]
            ora = [mask_forecast(best_of(rt.truth, [oracle_forecast(rt.truth, False), oracle_forecast(rt.truth, True)]), k)
                   for rt, k in zip(rts, cds.ks(n))]
            row[n]["оракул: лучший из CV и CT"] = evaluation.summarize_method(rts, ora)
    for n in names:
        methods = ["IMM", "итоговая (штатный шум)", "корректор основы, переобучен", "оракул: лучший из CV и CT"]
        lines = [f"{'шум':>6s} " + "".join(f"{m[:26]:>28s}" for m in methods)]
        for s in scales:
            lines.append(f"x{s:<5g} " + "".join(f"{rows[s][n][m]['ade']:28.1f}" for m in methods))
        ctx.summary.section(f"{n}: ADE (м) от множителя СКО шума РЛС", lines)
    nom, low = rows[scales[0]]["test"], rows[scales[-1]]["test"]
    a, b, c = nom["корректор основы, переобучен"]["ade"], low["корректор основы, переобучен"]["ade"], low["оракул: лучший из CV и CT"]["ade"]
    ctx.summary.section("Разложение ошибки корректора основы на test (доли от ошибки при штатном шуме)", [
        f"штатный {a:.1f} м -> шум x{scales[-1]:g} {b:.1f} м -> оракул {c:.1f} м: шум РЛС {(a - b) / a:.0%}, метод "
        f"{(b - c) / a:.0%} (< 0 - сеть почти без шума точнее оракула), непредсказуемость {c / a:.0%}"])
    ctx.summary.check("Лучший из оракулов не хуже каждого (test, штатный шум)",
                      nom["оракул: лучший из CV и CT"]["ade"] <= nom["IMM"]["ade"] + 1e-6, "оракул против IMM")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in ("IMM", "итоговая (штатный шум)", "корректор основы, переобучен", "оракул: лучший из CV и CT"):
        ax.plot(scales, [rows[s]["test"][m]["ade"] for s in scales], "-o", ms=4, label=m, color=method_color(m))
    ax.set_xscale("log")
    ax.set_xlabel("множитель СКО шума РЛС")
    ax.set_ylabel("ADE test, м")
    ax.legend(fontsize=7)
    ax.set_title(f"Вклад шума отметок: {cases.CASE_TITLES[ctx.case]}", fontsize=10)
    _save(fig, ctx.out_dir / "noise.png")


# ============================================================================
# Точность доплера
# ============================================================================

def section_doppler(ctx: Ctx, fin: Final) -> None:
    if ctx.case == "nodop":
        ctx.summary.section("Точность доплера", ["случай без доплера: раздел не применяется"])
        return
    names = config.EVAL_SETS
    sigmas = (0.5, 2.0) if ctx.smoke else SIGMAS_VR
    tracks_tr = pdata.train_tracks(ctx.sets)
    rows = {}
    for v in sigmas:
        over = {"RADAR_SIGMA_VR": v, "FILTER_SIGMA_VR": v}
        cdv = fin.cd.variant(extra=over)
        cl = classic_reports(cdv, names)
        res = retrain(ctx, fin, cdv, f"доплер СКО {v:g}", tracks_tr)
        rows[v] = {n: {"IMM с доплером": cl[n]["IMM"],
                       "корректор основы, переобучен": evaluation.summarize_method(cdv.eval(n)[1], res["fcs"][n])}
                   for n in names}
    ref = Registry(ctx.paths, "nodop")
    nodop_imm = ref.metric("stage1/IMM", "test")[0] if ref.has("stage1/IMM") else float("nan")
    for n in names:
        lines = [f"{'СКО доплера, м/с':>17s} {'IMM с доплером':>16s} {'корректор переобучен':>22s}"]
        for v in sigmas:
            lines.append(f"{v:17g} {rows[v][n]['IMM с доплером']['ade']:16.1f} "
                         f"{rows[v][n]['корректор основы, переобучен']['ade']:22.1f}")
        ctx.summary.section(f"{n}: ADE (м) от точности доплера; IMM без доплера на test: {nodop_imm:.1f} м", lines)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ("IMM с доплером", "корректор основы, переобучен"):
        ax.plot(sigmas, [rows[v]["test"][m]["ade"] for v in sigmas], "-o", ms=4, label=m, color=method_color(m))
    ax.axhline(nodop_imm, color="gray", ls="--", lw=0.8, label="IMM без доплера")
    ax.set_xlabel("СКО радиальной скорости, м/с")
    ax.set_ylabel("ADE test, м")
    ax.legend(fontsize=8)
    ax.set_title(f"Выигрыш доплера от его точности: {cases.CASE_TITLES[ctx.case]}", fontsize=10)
    _save(fig, ctx.out_dir / "doppler_accuracy.png")


# ============================================================================
# Сгенерированный test
# ============================================================================

def section_synth(ctx: Ctx, fin: Final) -> None:
    with open(ctx.paths.stage0 / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    share = 0.02 if ctx.smoke else config.SYNTH_TEST_SHARE
    counts = {s: max(1, int(round(n * share))) for s, n in config.SYNTH_COUNT_TRAIN.items()}
    tracks = generator.generate(counts, limits, make_rng("synth_test"), "synthT")
    from data import phases

    for tr in tracks:
        phases.annotate(tr)
    clash = {tr.flight_id for tr in tracks} & {tr.flight_id for tr in ctx.sets["synth"]}
    ctx.summary.check("Сгенерированный test не пересекается с обучением по именам", not clash,
                      f"траекторий {len(tracks)}, совпадений имён {len(clash)}")
    sets2 = {**ctx.sets, "synth_test": tracks}
    cd = fin.cd.variant()
    cd.sets = sets2
    cl = classic_reports(cd, ("synth_test",))
    nets = fin.methods(cd, ("synth_test",))
    real_test = {m: ctx.registry.report(m, "test")["ade"] for m in ("stage1/IMM", fin.method) if ctx.registry.has(m)}
    lines = [f"{'метод':34s} {'ADE':>8s} {'вход':>8s} {'маневр':>8s}",
             f"{'IMM':34s} {cl['synth_test']['IMM']['ade']:8.1f} {cl['synth_test']['IMM']['entry']:8.1f} {cl['synth_test']['IMM']['turn']:8.1f}",
             f"{'итоговая (среднее по seed)':34s} {np.mean([r['synth_test']['ade'] for r in nets['net']]):8.1f} "
             f"{np.mean([r['synth_test']['entry'] for r in nets['net']]):8.1f} {np.mean([r['synth_test']['turn'] for r in nets['net']]):8.1f}",
             f"{'ансамбль seed':34s} {nets['ens']['synth_test']['ade']:8.1f} {nets['ens']['synth_test']['entry']:8.1f} {nets['ens']['synth_test']['turn']:8.1f}"]
    ctx.summary.section("Отдельный сгенерированный test (не входил в обучение и разбиение), ADE (м)", lines + [
        "для сравнения test этого прогона: " + ", ".join(f"{m} {v:.1f}" for m, v in real_test.items())])
    ctx.save_state("synth_test", {"imm": cl["synth_test"]["IMM"], "net": {k: float(np.mean([r["synth_test"][k] for r in nets["net"]]))
                                                                          for k in ("ade", "entry", "turn")},
                                  "ens": nets["ens"]["synth_test"]})


# ============================================================================
# Реакция на маневр
# ============================================================================

def section_react(ctx: Ctx, fin: Final) -> None:
    cd = fin.cd
    j = config.IMM_MODELS.index("CT")
    lines = []
    for n in ("val", "test"):
        _, rts, runs = cd.eval(n)
        vals = [[] for _ in range(6)]
        for rt, run in zip(rts, runs):
            for o in rt.onsets:
                for m in range(6):
                    if o + m < len(rt.t) and not np.isnan(run.mu[o + m, j]):
                        vals[m].append(run.mu[o + m, j])
        curve = [float(np.mean(v)) if v else float("nan") for v in vals]
        lines.append(f"{n}: средняя вероятность CT в обзорах 0..5 от начала маневра: " + " ".join(f"{c:.2f}" for c in curve))
        corr = fin.corr(cd, (n,))[0]
        rep = evaluation.dynamic_summary(rts, evaluation.hybrid_forecasts(cd, n, corr[n]))
        imm = evaluation.dynamic_summary(rts, evaluation.imm_forecasts(cd, n))
        lines.append(f"{n}: восстановление динамической ошибки (обзоров): IMM {imm['dyn_recovery']:.0f}, итоговая "
                     f"{rep['dyn_recovery']:.0f}; пик на h=5: IMM {imm['dyn_peak']:.0f} м, итоговая {rep['dyn_peak']:.0f} м")
    ctx.summary.section("Быстрота реакции на маневр", lines)


def run_case(ctx: Ctx) -> bool:
    if ctx.state("final") is None:
        ctx.summary.check("Итоговая модель случая выбрана на этапе 7", False, "нет состояния final")
        return False
    only = {s.strip() for s in ctx.args.only.split(",")}
    fin = Final(ctx)
    ctx.summary.section("Итоговая модель", [f"{fin.method} (основа: {fin.base_method}), seed {fin.seeds}"])
    for name, fn in (("gaps", section_gaps), ("gate", section_gate), ("noise", section_noise),
                     ("doppler", section_doppler), ("synth", section_synth), ("react", section_react)):
        if name in only:
            ctx.log.info("Раздел %s", name)
            fn(ctx, fin)
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 9: анализ итоговых моделей", run_case, needs_gpu=True, add_args=add_args)


if __name__ == "__main__":
    sys.exit(main())
