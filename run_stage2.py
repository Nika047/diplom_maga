"""
Этап 2 (версия 2): корректоры IMM + LSTM / GRU / Transformer / Mamba и сети без IMM (GRU, Mamba).

Для случая доплера (без доплера / по траектории / по телеметрии):
  tune  - двухцелевой генетический подбор (Optuna, NSGA-II) каждой модели: ADE на val и время отклика
          сети (мс на обзор); в случаях с доплером в геном входит «доплер во входе сети да / нет».
          Попытки хранятся в cache/<прогон>/stage2/<случай>/optuna.db: прерванный запуск продолжается;
  final - две конфигурации с фронта Парето (лучшая по ADE и самая быстрая из не хуже лучшей больше
          чем на ARCH_NEAR_BEST), обучение по FINAL_SEEDS, оценка на val, test (holdout) вместе с
          IMM этапа 1. Обученные сети лежат в model_store/<прогон>/stage2/<случай>/ и при повторе не
          обучаются заново; прерванное обучение продолжается с последней эпохи.
Данные обучения - реальные полёты train + их расширение + синтетика.

  .venv\\Scripts\\python.exe run_stage2.py [--case ...] [--phase tune|final|all] [--only GRU,Mamba] [--smoke]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from common import cases  # noqa: E402
from eval import selftests  # noqa: E402
from eval.compare import compare  # noqa: E402
from pipeline import data as pdata  # noqa: E402
from pipeline import evaluation  # noqa: E402
from pipeline.runs import fit_and_eval  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from pipeline.tables import method_table  # noqa: E402
from tuning import corrector_search, search  # noqa: E402
from viz.tuning_plots import plot_convergence, plot_pareto  # noqa: E402

STAGE = "stage2"
EXPERIMENTS = [("IMM+LSTM", "LSTM", "hybrid"), ("IMM+GRU", "GRU", "hybrid"),
               ("IMM+Transformer", "Transformer", "hybrid"), ("IMM+Mamba", "Mamba", "hybrid"),
               ("GRU (без IMM)", "GRU", "e2e"), ("Mamba (без IMM)", "Mamba", "e2e")]
FAST = " (быстрая)"


def add_args(parser) -> None:
    parser.add_argument("--phase", choices=("tune", "final", "all"), default="all")
    parser.add_argument("--only", default="", help="только эти корректоры: GRU,Mamba")
    parser.add_argument("--trials", type=int, default=None, help="бюджет подбора на модель")


def casedata_for(ctx: Ctx, net_input: bool):
    return ctx.casedata(net_input=net_input)


def phase_tune(ctx: Ctx, experiments: list, n_trials: int, data: dict) -> bool:
    st, summary, log = ctx.st, ctx.summary, ctx.log
    state = ctx.state("corr", {})
    curves, points, lines, counts = {}, {}, [], []
    for label, corrector, mode in experiments:
        study = search.open_study(f"{label}{'_smoke' if ctx.smoke else ''}", ctx.cache_dir / "optuna.db",
                                  n_objectives=2)
        objective = corrector_search.CorrectorObjective(corrector, mode, data, ctx.device, epochs=st.tune_epochs,
                                                       patience=st.tune_patience)
        t0 = time.time()
        log.info("Подбор %s: бюджет %d попыток", label, n_trials)
        search.run_budget(study, objective, n_trials, corrector_search.default_params(objective.allow_net_input))
        done = search.completed(study)
        counts.append(len(done))
        best, fast = corrector_search.pick(done)
        default = search.default_trial(study)
        front = {t.number for t in study.best_trials}
        state[label] = {"best": best.params, "fast": fast.params, "ade_val": best.values[0], "ms_best": best.values[1],
                        "ade_val_fast": fast.values[0], "ms_fast": fast.values[1],
                        "ade_val_default": default.values[0], "trials": len(done)}
        summary.check(f"{label}: лучшее не хуже исходного", best.values[0] <= default.values[0] + 1e-9,
                      f"ADE val: исходные {default.values[0]:.2f} м, лучшие {best.values[0]:.2f} м")
        lines.append(f"{label:18s} попыток {len(done):3d}, исходные {default.values[0]:.2f} м -> лучшие "
                     f"{best.values[0]:.2f} м ({best.values[0] / default.values[0] - 1:+.1%}, {best.values[1]:.2f} мс); "
                     f"быстрая {fast.values[0]:.2f} м ({fast.values[1]:.2f} мс); "
                     f"{ {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best.params.items()} }")
        curves[label] = [t.values[0] for t in done]
        points[label] = [(t.values[1], t.values[0], t.number in front) for t in done]
        ctx.save_state("corr", state)
        log.info("Подбор %s: %.1f мин", label, (time.time() - t0) / 60)
    summary.check("Бюджет подбора одинаков для всех моделей", len(set(counts)) == 1 and counts[0] >= n_trials,
                  f"завершённых попыток: {counts} (бюджет {n_trials})")
    summary.section("Подбор гиперпараметров: ADE val (м) и время сети (мс на обзор)", lines)
    plot_convergence(curves, ctx.out_dir / "convergence.png", f"Генетический подбор корректоров: {cases.CASE_TITLES[ctx.case]}")
    plot_pareto(points, ctx.out_dir / "pareto.png", f"Точность и время отклика: {cases.CASE_TITLES[ctx.case]}",
                xlabel="время сети, мс на обзор")
    return True


def phase_final(ctx: Ctx, experiments: list, tracks_tr: list) -> bool:
    st, summary, log = ctx.st, ctx.summary, ctx.log
    state = ctx.state("corr", {})
    missing = [label for label, _, _ in experiments if label not in state]
    if missing:
        summary.check("Результаты подбора найдены", False, f"нет: {missing}; сначала --phase tune")
        return False
    sets = ctx.sets
    ids = {name: {tr.log_file or tr.flight_id for tr in sets[name]} for name in config.EVAL_SETS}
    train_ids = {tr.log_file or tr.flight_id for tr in tracks_tr}
    leak = train_ids & set().union(*ids.values())
    summary.check("val, test (holdout) не пересекаются с обучением", not leak and all(ids.values()),
                  ", ".join(f"{n} {len(v)}" for n, v in ids.items()) + f"; пересечений {len(leak)}")

    cds = {net: ctx.casedata(net_input=net) for net in ((True, False) if ctx.case != "nodop" else (True,))}
    ws_tr = {net: cd.windows("main", tracks_tr) for net, cd in cds.items()}
    forecasts = {name: {} for name in config.EVAL_SETS}
    methods = []
    for label, corrector, mode in experiments:
        variants = [(label, state[label]["best"])]
        if state[label]["fast"] != state[label]["best"]:
            variants.append((label + FAST, state[label]["fast"]))
        for name_, params in variants:
            window, net, hp, train_params = corrector_search.split(params)
            net = net if mode == "hybrid" else True
            cd = cds.get(net, cds[True])
            out = fit_and_eval(ctx, cd, STAGE, name_, corrector, mode, ws_tr[net if net in ws_tr else True],
                               train_params, hp, window)
            methods.append(name_)
            for name in config.EVAL_SETS:
                forecasts[name][name_] = out[0]["fcs"][name]
            got, ga = out[0]["reports"]["val"]["ade"], state[label]["ade_val" if name_ == label else "ade_val_fast"]
            summary.check(f"{name_}: лучшая конфигурация воспроизводится", abs(got / ga - 1) <= 0.15,
                          f"ADE val при подборе {ga:.2f} м, при итоговом обучении (seed {out[0]['seed']}, до "
                          f"{st.epochs} эпох) {got:.2f} м")
    cd0 = cds[True]
    for name in config.EVAL_SETS:
        _, rts, _ = cd0.eval(name)
        summary.section(f"{name}: ADE (м), среднее ± СКО по {len(st.seeds)} seed; время - шаг IMM + сеть",
                        method_table(ctx.registry, ["stage1/IMM"] + [f"{STAGE}/{m}" for m in methods], name,
                                     ref="stage1/IMM"))
        shown = {"IMM": evaluation.imm_forecasts(cd0, name)}
        shown.update({m: forecasts[name][m] for m in methods if not m.endswith(FAST)})
        compare(rts, shown, summary, f"{name} (seed {st.seeds[0]})", ctx.out_dir, name)
    (ctx.out_dir / "metrics.json").write_text(json.dumps(
        {m: {n: {k: v for k, v in ctx.registry.report(f"{STAGE}/{m}", n).items() if k != "flights"}
             for n in config.EVAL_SETS} for m in methods}, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def run_case(ctx: Ctx) -> bool:
    st, args = ctx.st, ctx.args
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    experiments = [e for e in EXPERIMENTS if not only or e[1] in only]
    tracks_tr = pdata.train_tracks(ctx.sets)
    cd = ctx.casedata(net_input=True)
    selftests.window_selftest(ctx.summary, ctx.sets["val"], cd)
    ok = True
    if args.phase in ("tune", "all"):
        nets = (True, False) if ctx.case != "nodop" else (True,)
        data = {}
        for net in nets:
            cdn = ctx.casedata(net_input=net)
            data[(config.TUNE_WINDOWS[0], net)] = (cdn.windows("main", tracks_tr), cdn.eval("val")[0])
        n_trials = args.trials or st.trials
        ok &= phase_tune(ctx, experiments, n_trials, data)
        del data
    if ok and args.phase in ("final", "all"):
        ok &= phase_final(ctx, experiments, tracks_tr)
    return ok


def main() -> int:
    return run_stage(STAGE, "Этап 2: корректоры", run_case, needs_gpu=True, add_args=add_args)


if __name__ == "__main__":
    sys.exit(main())
