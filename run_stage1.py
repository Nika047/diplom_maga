"""
Этап 1 (версия 2): классические методы и подбор их параметров в каждом случае доплера.

Для случая (без доплера / доплер по траектории / доплер по телеметрии, common/cases.py):
  1. самопроверки на подставных данных с известным ответом (метрики, модели движения, бутстреп);
  2. генетический подбор (Optuna, NSGA-II) параметров IMM по ADE на реальных трассах train (каждая
     TUNE_IMM_TRACK_STEP-я); в случаях с доплером в геном входит доверие фильтра к доплеру
     (FILTER_SIGMA_VR из DOPPLER_FILTER_SIGMAS);
  3. тот же бюджет для фильтра Калмана (СКО шума ускорения, и доверие к доплеру); линейная
     экстраполяция - полный перебор числа отметок;
  4. итог на val и test (holdout): CV, KF, IMM с исходными и подобранными параметрами, на одних
     обзорах; метрики, участки, динамическая ошибка, время шага фильтра.

Результат для следующих этапов - состояние imm_<случай> (параметры IMM и доверие к доплеру) и записи
реестра stage1/<метод>. Попытки подбора хранятся в cache/<прогон>/stage1/<случай>/optuna.db: прерванный
запуск продолжается (та же команда). Нейросетей нет, GPU не нужен.

  .venv\\Scripts\\python.exe run_stage1.py [--case nodop|dop_traj|dop_tele|all] [--smoke]
"""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from common import cases  # noqa: E402
from common.cases import config_override  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from eval import metrics, selftests  # noqa: E402
from eval.compare import compare  # noqa: E402
from filters.extrapolation import cv_extrapolate  # noqa: E402
from filters.imm import IMM  # noqa: E402
from filters.models import X, Y  # noqa: E402
from pipeline import evaluation, latency  # noqa: E402
from pipeline.casedata import CaseData  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from radar.model import observe  # noqa: E402
from tuning import search  # noqa: E402
from viz.tuning_plots import plot_convergence  # noqa: E402

STAGE = "stage1"
SIGMAS = tuple(config.DOPPLER_FILTER_SIGMAS)


def _ade(rts: list, forecasts: list) -> float:
    return metrics.summarize(metrics.concat_pairs([metrics.forecast_pairs(rt, fc)
                                                   for rt, fc in zip(rts, forecasts)]))["ade"]


def imm_objective(rts: list, case: str):
    def objective(trial):
        values = {key: trial.suggest_float(key, lo, hi, log=True) for key, (lo, hi) in config.TUNE_IMM_SPACE.items()}
        values["IMM_P_STAY"] = trial.suggest_float("IMM_P_STAY", *config.TUNE_IMM_P_STAY)
        sigma = trial.suggest_categorical("FILTER_SIGMA_VR", list(SIGMAS)) if case != "nodop" else None
        with config_override({**cases.overrides(case, False, sigma), **values}):
            imm = IMM()
            pairs = metrics.concat_pairs([metrics.forecast_pairs(rt, imm.run(rt).forecast) for rt in rts])
        s = metrics.summarize(pairs)
        turn = metrics.summarize(metrics.select(pairs, np.isin(pairs["phase"], (PHASE_CODE["entry"], PHASE_CODE["turn"]))))
        trial.set_user_attr("maneuver_ade", turn["ade"])
        return s["ade"]
    return objective


def kf_objective(rts: list, case: str):
    def objective(trial):
        values = {key: trial.suggest_float(key, lo, hi, log=True) for key, (lo, hi) in config.TUNE_KF_SPACE.items()}
        sigma = trial.suggest_categorical("FILTER_SIGMA_VR", list(SIGMAS)) if case != "nodop" else None
        with config_override({**cases.overrides(case, False, sigma), **values}):
            kf = IMM(("CV",), mu0=(1.0,))
            return _ade(rts, [kf.run(rt).forecast for rt in rts])
    return objective


def tune(ctx: Ctx, name: str, objective, default: dict, n_trials: int):
    study = search.open_study(f"{name}{'_smoke' if ctx.smoke else ''}", ctx.cache_dir / "optuna.db")
    t0 = time.time()
    search.run_budget(study, objective, n_trials, default)
    done = search.completed(study)
    ctx.summary.check(f"{name.upper()}: бюджет выполнен", len(done) >= n_trials, f"завершено {len(done)} из {n_trials}")
    best, base = study.best_trial, search.default_trial(study)
    ctx.summary.check(f"{name.upper()}: лучшее не хуже исходного", best.value <= base.value + 1e-9,
                      f"ADE train: исходные {base.value:.2f} м, лучшие {best.value:.2f} м")
    return best, base, done, time.time() - t0


def check_methods(ctx: Ctx, imm_params: dict, sigma_vr, analysis: list) -> None:
    """Проверки на синтетике анализа: KF сглаживает шум на прямых, у IMM на маневре растёт вероятность CT."""
    over = {**cases.overrides(ctx.case, False, sigma_vr), **imm_params}
    with config_override(over):
        rts = [observe(tr) for tr in analysis]
        kf_runs, imm_runs = [IMM(("CV",), mu0=(1.0,)).run(rt) for rt in rts], [IMM().run(rt) for rt in rts]
    est_err, meas_err = [], []
    for rt, run in zip(rts, kf_runs):
        if not rt.flight_id.startswith("synthA_straight"):
            continue
        k = np.arange(config.WARMUP_SCANS, len(rt.t))
        est_err.append(np.linalg.norm(run.est[k][:, [X, Y]] - rt.truth[k, :2], axis=1))
        meas_err.append(np.linalg.norm(rt.meas[k, :2] - rt.truth[k, :2], axis=1))
    e, m = np.concatenate(est_err), np.concatenate(meas_err)
    ctx.summary.check("Фильтр Калмана на прямой точнее отметок", e.mean() < m.mean(),
                      f"средняя горизонтальная ошибка оценки {e.mean():.1f} м против отметок {m.mean():.1f} м")
    i_ct = config.IMM_MODELS.index("CT")
    mu_turn, mu_straight = [], []
    for rt, run in zip(rts, imm_runs):
        ok = ~np.isnan(run.mu[:, i_ct])
        mu_turn.append(run.mu[ok & (rt.phase == PHASE_CODE["turn"]), i_ct])
        mu_straight.append(run.mu[ok & (rt.phase == PHASE_CODE["straight"]), i_ct])
    a, b = np.concatenate(mu_turn).mean(), np.concatenate(mu_straight).mean()
    ctx.summary.check("IMM: на маневре растёт вероятность CT", a > b + 0.2,
                      f"средняя вероятность CT на маневре {a:.2f}, на прямой {b:.2f}")


def run_case(ctx: Ctx) -> bool:
    st, case, summary, log = ctx.st, ctx.case, ctx.summary, ctx.log
    selftests.filters_selftest(summary)
    selftests.bootstrap_selftest(summary)
    sets = ctx.sets
    t0 = time.time()

    # --- 1. подбор на реальных трассах train ---
    tracks = sets["train"][::config.TUNE_IMM_TRACK_STEP]
    with config_override(cases.overrides(case, False, None)):
        rts = [observe(tr) for tr in tracks]
    log.info("Подбор: %d трасс train, %d обзоров, бюджет IMM / KF %d попыток", len(rts), sum(len(r.t) for r in rts),
             st.imm_trials)
    sigma0 = {"FILTER_SIGMA_VR": config.RADAR_SIGMA_VR} if case != "nodop" else {}
    imm_best, imm_base, imm_done, imm_s = tune(ctx, "imm", imm_objective(rts, case),
                                               {**search.default_imm_params(), **sigma0}, st.imm_trials)
    kf_best, kf_base, kf_done, kf_s = tune(ctx, "kf", kf_objective(rts, case),
                                           {**{k: getattr(config, k) for k in config.TUNE_KF_SPACE}, **sigma0},
                                           st.imm_trials)
    imm_params = {k: imm_best.params[k] for k in search.IMM_KEYS}
    sigma_vr = imm_best.params.get("FILTER_SIGMA_VR")
    kf_params = {k: kf_best.params[k] for k in config.TUNE_KF_SPACE}
    kf_sigma = kf_best.params.get("FILTER_SIGMA_VR")

    cv_ade = {n: _ade(rts, [cv_extrapolate(rt, n_points=n) for rt in rts]) for n in config.TUNE_CV_POINTS}
    cv_best = min(cv_ade, key=cv_ade.get)
    summary.check("CV: лучшее не хуже исходного", cv_ade[cv_best] <= cv_ade[config.CV_LS_POINTS] + 1e-9,
                  f"ADE train: {config.CV_LS_POINTS} отметок {cv_ade[config.CV_LS_POINTS]:.2f} м, "
                  f"{cv_best} отметок {cv_ade[cv_best]:.2f} м")
    summary.section("Подбор на train (ADE, м)", [
        f"трасс {len(rts)}, попыток IMM {len(imm_done)} ({imm_s:.0f} с), KF {len(kf_done)} ({kf_s:.0f} с)",
        f"IMM: исходные {imm_base.value:.2f} -> лучшие {imm_best.value:.2f} ({imm_best.value / imm_base.value - 1:+.1%}), "
        f"на маневре {imm_best.user_attrs.get('maneuver_ade', float('nan')):.1f} м; "
        f"{ {k: round(v, 4) for k, v in imm_best.params.items()} }",
        f"KF: исходные {kf_base.value:.2f} -> лучшие {kf_best.value:.2f} ({kf_best.value / kf_base.value - 1:+.1%}); "
        f"{ {k: round(v, 4) for k, v in kf_best.params.items()} }",
        "CV: " + ", ".join(f"{n} отм. {v:.2f}" for n, v in list(cv_ade.items())[::6]) + f"; лучшее - {cv_best}",
    ])
    plot_convergence({"IMM": [t.value for t in imm_done], "KF": [t.value for t in kf_done]},
                     ctx.out_dir / "convergence.png", f"Генетический подбор IMM и KF: {cases.CASE_TITLES[case]}")
    ctx.save_state("imm", {"params": imm_params, "sigma_vr": sigma_vr, "ade_train": imm_best.value,
                           "ade_train_default": imm_base.value})
    ctx.save_state("classic", {"kf": kf_params, "kf_sigma_vr": kf_sigma, "cv_points": cv_best})

    # --- 2. итог на val, test (holdout) ---
    cd = CaseData(ctx.paths, sets, case, imm=imm_params, sigma_vr=sigma_vr)
    imm_ms = latency.imm_ms_per_scan(cd, "val")
    for name in config.EVAL_SETS:
        _, rts_e, _ = cd.eval(name)
        fcs = evaluation.classic_forecasts(cd, name, imm_default=search.default_imm_params())
        fcs = {"CV": cd.mask(name, [cv_extrapolate(rt, n_points=cv_best) for rt in rts_e]), "KF (исходный)": fcs["KF"],
               "IMM (исходный)": fcs["IMM (исходный)"], "IMM": fcs["IMM"],
               "CV (исходный)": cd.mask(name, [cv_extrapolate(rt) for rt in rts_e])}
        with config_override({**cd.override(), **kf_params, **cases.overrides(case, False, kf_sigma)}):
            kf = IMM(("CV",), mu0=(1.0,))
            fcs["KF"] = cd.mask(name, [kf.run(rt).forecast for rt in rts_e])
        order = ["CV (исходный)", "CV", "KF (исходный)", "KF", "IMM (исходный)", "IMM"]
        fcs = {m: fcs[m] for m in order}
        for method, fc in fcs.items():
            ctx.registry.set_report(f"{STAGE}/{method}", 0, name, evaluation.report(cd, name, fc),
                                    {"imm_ms": imm_ms, "total_cpu_ms": imm_ms})
        lines = [f"{'метод':16s} {'ADE':>6s} {'FDE':>6s} {'R2':>6s} {'прямая':>7s} {'вход':>7s} {'маневр':>7s}"]
        for method in order:
            r = ctx.registry.report(f"{STAGE}/{method}", name)
            lines.append(f"{method:16s} {r['ade']:6.1f} {r['fde']:6.1f} {r['r2']:6.3f} {r['straight']:7.1f} "
                         f"{r['entry']:7.1f} {r['turn']:7.1f}")
        summary.section(f"{name}: итог (м), одни и те же обзоры", lines)
        compare(rts_e, {m: fcs[m] for m in ("CV", "KF", "IMM")}, summary, f"{name}", ctx.out_dir, name)
    ctx.registry.flush()

    val = {m: ctx.registry.report(f"{STAGE}/{m}", "val") for m in ("KF", "IMM")}
    summary.check("val: IMM на входе в маневр и маневре точнее фильтра Калмана", val["IMM"]["turn"] < val["KF"]["turn"],
                  f"ADE на маневре: IMM {val['IMM']['turn']:.1f} м, KF {val['KF']['turn']:.1f} м")
    with open(ctx.paths.stage0 / "tracks_synth_analysis.pkl", "rb") as fh:
        analysis = pickle.load(fh)
    check_methods(ctx, imm_params, sigma_vr, analysis if not ctx.smoke else analysis[::5])
    summary.section("Время шага фильтра (процессор, мс на обзор)", [f"IMM случая «{cases.CASE_TITLES[case]}»: {imm_ms:.2f}"])
    clean = {m: {n: {k: v for k, v in ctx.registry.report(f"{STAGE}/{m}", n).items() if k != "flights"}
                 for n in config.EVAL_SETS}
             for m in ("CV (исходный)", "CV", "KF (исходный)", "KF", "IMM (исходный)", "IMM")}
    (ctx.out_dir / "metrics.json").write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 1: классические методы и подбор IMM", run_case)


if __name__ == "__main__":
    sys.exit(main())
