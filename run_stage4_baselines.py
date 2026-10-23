"""
Этап 4, дополнение: генетический подбор параметров базовых методов.

На этапе 4 подбирался только IMM, линейная экстраполяция и фильтр Калмана остались с
исходными параметрами. Здесь они подбираются так же, как IMM: та же цель (ADE
прогноза по горизонтали на реальных трассах train, каждая TUNE_IMM_TRACK_STEP-я),
тот же бюджет и тот же генетический алгоритм (Optuna, NSGA-II):
  KF - СКО шума ускорения модели CV (TUNE_KF_SPACE), бюджет TUNE_IMM_TRIALS;
  CV - число отметок для прямой (CV_LS_POINTS): целое из короткого диапазона, поэтому
       полный перебор TUNE_CV_POINTS вместо генетического алгоритма.
Итог - val, test и holdout: CV, KF и IMM с исходными и подобранными параметрами, на
тех же обзорах, что в итоге этапа 4 (окно TUNE_WINDOWS[0]). Подобранный IMM берётся
из cache/stage4/imm_best.json (этап 4, шаг imm); если его нет, строки IMM без него.

Нейросетей нет, GPU не нужен: полный запуск допустим и на рабочем ПК.
  .venv\\Scripts\\python.exe run_stage4_baselines.py
Смоук: .venv\\Scripts\\python.exe run_stage4_baselines.py --smoke

Проверки: бюджет выполнен, подобранное не хуже исходного на train, лучшее число
отметок CV не на краю диапазона перебора.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from filters.extrapolation import cv_extrapolate  # noqa: E402
from filters.imm import IMM  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import CACHE, SMOKE, load_data, plot_convergence, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from tuning import search  # noqa: E402

STAGE = "stage4_baselines"
OUT_DIR = config.RESULTS_DIR / STAGE


def _kf() -> IMM:
    return IMM(("CV",), mu0=(1.0,))


def _ade(rts: list, forecasts: list) -> float:
    return metrics.summarize(metrics.concat_pairs([metrics.forecast_pairs(rt, fc)
                                                   for rt, fc in zip(rts, forecasts)]))["ade"]


def kf_objective(rts: list):
    def objective(trial) -> float:
        values = {key: trial.suggest_float(key, lo, hi, log=True) for key, (lo, hi) in config.TUNE_KF_SPACE.items()}
        with search.config_override(values):
            kf = _kf()
            return _ade(rts, [kf.run(rt).forecast for rt in rts])
    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 4: подбор базовых методов")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 4, подбор базовых методов" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.CACHE_DIR / "stage4_smoke" if args.smoke else CACHE
    n_trials = SMOKE["imm_trials"] if args.smoke else config.TUNE_IMM_TRIALS
    t0 = time.time()

    sets = load_data(args.smoke)
    tracks = sets["train"][::config.TUNE_IMM_TRACK_STEP]
    rts = [observe(tr) for tr in tracks]
    log.info("Подбор базовых методов: %d трасс train, %d обзоров", len(rts), sum(len(r.t) for r in rts))

    # Фильтр Калмана - генетический алгоритм
    default_kf = {key: getattr(config, key) for key in config.TUNE_KF_SPACE}
    study = search.open_study(f"kf{tag}", cache / "optuna.db")
    search.run_budget(study, kf_objective(rts), n_trials, default_kf)
    done = search.completed(study)
    kf_default, kf_best = search.default_trial(study).value, study.best_trial
    summary.check("KF: бюджет выполнен", len(done) >= n_trials, f"завершено {len(done)} из {n_trials}")
    summary.check("KF: лучшее не хуже исходного", kf_best.value <= kf_default + 1e-9,
                  f"ADE train: исходные {kf_default:.2f} м, лучшие {kf_best.value:.2f} м")
    plot_convergence({"KF": [t.value for t in done]}, OUT_DIR / f"convergence_kf{tag}.png",
                     "Генетический подбор фильтра Калмана")

    # Линейная экстраполяция - полный перебор
    cv_ade = {n: _ade(rts, [cv_extrapolate(rt, n_points=n) for rt in rts]) for n in config.TUNE_CV_POINTS}
    cv_best = min(cv_ade, key=cv_ade.get)
    summary.check("CV: лучшее не хуже исходного", cv_ade[cv_best] <= cv_ade[config.CV_LS_POINTS] + 1e-9,
                  f"ADE train: {config.CV_LS_POINTS} отметок {cv_ade[config.CV_LS_POINTS]:.2f} м, "
                  f"{cv_best} отметок {cv_ade[cv_best]:.2f} м")
    summary.check("CV: лучшее внутри диапазона перебора", min(cv_ade) < cv_best < max(cv_ade),
                  f"диапазон {min(cv_ade)}-{max(cv_ade)}, лучшее {cv_best}")

    imm_path = cache / "imm_best.json"
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"] if imm_path.exists() else None
    summary.section("Подбор на train (ADE, м)", [
        f"трасс {len(rts)}, время {time.time() - t0:.0f} с",
        f"KF: исходные {default_kf} -> {kf_default:.2f}; лучшие "
        f"{ {k: round(v, 4) for k, v in kf_best.params.items()} } -> {kf_best.value:.2f} "
        f"({kf_best.value / kf_default - 1:+.1%}), попыток {len(done)}",
        "CV: " + ", ".join(f"{n} отм. {v:.2f}" for n, v in cv_ade.items()) + f"; лучшее - {cv_best}",
        "IMM: " + (f"подобранные параметры из {imm_path}" if imm else
                   "подобранного нет (этап 4, шаг imm) - в таблицах только исходный"),
    ])

    # Итог на тех же обзорах, что у итога этапа 4
    w = config.TUNE_WINDOWS[0]
    for name in config.EVAL_SETS:
        with search.config_override(imm or {}):
            ws, rts_e, runs = dataset.build(sets[name], window=w)
        ks = [np.array(sorted(ws.k[ws.track == i]), dtype=int) for i in range(len(rts_e))]
        fcs = {
            "CV (исходный)": [cv_extrapolate(rt) for rt in rts_e],
            "CV": [cv_extrapolate(rt, n_points=cv_best) for rt in rts_e],
            "KF (исходный)": [_kf().run(rt).forecast for rt in rts_e],
        }
        with search.config_override(kf_best.params):
            kf = _kf()
            fcs["KF"] = [kf.run(rt).forecast for rt in rts_e]
        with search.config_override(search.default_imm_params()):
            imm_default = IMM()
            fcs["IMM (исходный)"] = [imm_default.run(rt).forecast for rt in rts_e]
        if imm:
            fcs["IMM"] = [r.forecast for r in runs]
        fcs = {m: [mask_forecast(fc, k) for fc, k in zip(f, ks)] for m, f in fcs.items()}
        compare(rts_e, fcs, summary, name, OUT_DIR, f"{name}{tag}")
        lines = [f"{'метод':16s} {'ADE':>6s} {'FDE':>6s} {'R2':>6s} {'прямая':>7s} {'вход':>7s} {'маневр':>7s}"]
        for method, f in fcs.items():
            r = summarize_method(rts_e, f)
            lines.append(f"{method:16s} {r['ade']:6.1f} {r['fde']:6.1f} {r['r2']:6.3f} {r['straight']:7.1f} "
                         f"{r['entry']:7.1f} {r['turn']:7.1f}")
        summary.section(f"{name}: итог (м), обзоры как в итоге этапа 4", lines)

    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
