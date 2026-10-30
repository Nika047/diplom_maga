"""
Этап 4 (версия 2): эволюционный поиск архитектуры корректора.

Геном попытки (tuning/arch_search.py): стопка из 1..ARCH_MAX_BLOCKS блоков GRU / LSTM / Mamba / Attention,
ширина, dropout, параметры обучения, отбор групп признаков окна (в случаях с доплером среди групп есть
«доплер»). Цели NSGA-II: ADE на val и время прохода сети (мс на обзор). Итог - две архитектуры с фронта
(лучшая по ADE и самая быстрая из не хуже лучшей больше чем на ARCH_NEAR_BEST, если быстрее на FAST_MIN_GAIN):
обучение по FINAL_SEEDS и
оценка на val, test (holdout) рядом с IMM и IMM + GRU этапа 2.

  search - поиск, попытки в cache/<прогон>/stage4/<случай>/optuna.db (прерванный запуск продолжается);
  final  - итоговое обучение, сети в model_store/<прогон>/stage4/<случай>/.

  .venv\\Scripts\\python.exe run_stage4.py [--case ...] [--phase search|final|all] [--smoke]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from common import cases  # noqa: E402
from eval.compare import compare  # noqa: E402
from pipeline import data as pdata  # noqa: E402
from pipeline import evaluation  # noqa: E402
from pipeline.runs import fit_and_eval  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from pipeline.tables import method_table  # noqa: E402
from tuning import arch_search, corrector_search, search  # noqa: E402
from viz.tuning_plots import plot_pareto  # noqa: E402

STAGE = "stage4"


def add_args(parser) -> None:
    parser.add_argument("--phase", choices=("search", "final", "all"), default="all")
    parser.add_argument("--trials", type=int, default=None)


def arch_label(params: dict, n_models: int, doppler: bool) -> str:
    arch = arch_search.decode(params, n_models, doppler)
    return f"IMM+стопка {'+'.join(arch['blocks'])} x{arch['hidden']}"


def phase_search(ctx: Ctx, cd, ws_tr, ws_val, n_trials: int) -> bool:
    st, summary = ctx.st, ctx.summary
    n_models, doppler = len(config.IMM_MODELS), ctx.case != "nodop"
    w = config.TUNE_WINDOWS[0]
    study = search.open_study(f"arch{'_smoke' if ctx.smoke else ''}", ctx.cache_dir / "optuna.db", n_objectives=2,
                              population=4 if ctx.smoke else config.ARCH_POPULATION)
    objective = arch_search.ArchObjective(ws_tr, ws_val, ctx.device, n_models, w, epochs=st.tune_epochs,
                                          patience=st.tune_patience, doppler=doppler)
    t0 = time.time()
    ctx.log.info("Поиск архитектуры: бюджет %d попыток", n_trials)
    search.run_budget(study, objective, n_trials, arch_search.default_params(n_models, doppler))
    done = search.completed(study)
    front = study.best_trials
    default = search.default_trial(study)
    summary.check("Бюджет поиска выполнен", len(done) >= n_trials, f"завершено {len(done)} из {n_trials}")
    summary.check("Фронт Парето не пуст, исходная архитектура оценена", len(front) > 0 and default is not None,
                  f"точек фронта {len(front)}" + (f", исходная: ADE {default.values[0]:.2f} м, {default.values[1]:.2f} мс"
                                                  if default else ", исходной нет"))
    lines = [f"{'ADE, м':>7s} {'мс':>7s} {'парам.':>8s}  архитектура / группы признаков"]
    for t in sorted(front, key=lambda t: t.values[0]):
        lines.append(f"{t.values[0]:7.2f} {t.values[1]:7.2f} {t.user_attrs.get('n_params', 0):8d}  "
                     f"{'+'.join(t.user_attrs.get('blocks', []))} x{t.params['hidden']}; "
                     f"{', '.join(t.user_attrs.get('groups', []))}")
    summary.section(f"Фронт Парето ({len(front)} точек из {len(done)} попыток, {(time.time() - t0) / 60:.0f} мин)", lines)
    rows = arch_search.feature_importance(done, n_models, doppler=doppler)
    summary.section("Важность групп признаков (по всем попыткам)", [
        f"{'группа':22s} {'ADE с ней':>10s} {'ADE без':>9s} {'доля в лучших 20%':>18s}"] + [
        f"{r['group']:22s} {r['ade_with']:10.2f} {r['ade_without']:9.2f} {r['share_in_top']:18.0%}" for r in rows])
    rows_b = arch_search.block_statistics(done)
    summary.section("Типы блоков (по всем попыткам)", [
        f"{'блок':10s} {'ADE с ним':>10s} {'ADE без':>9s} {'доля среди блоков лучших 20%':>30s}"] + [
        f"{r['block']:10s} {r['ade_with']:10.2f} {r['ade_without']:9.2f} {r['share_in_top_blocks']:30.0%}"
        for r in rows_b])

    best, fast = corrector_search.pick(done)
    out = {"best": {"params": best.params, "ade_val": best.values[0], "ms": best.values[1]},
           "fast": {"params": fast.params, "ade_val": fast.values[0], "ms": fast.values[1]},
           "default": {"params": default.params, "ade_val": default.values[0], "ms": default.values[1]},
           "importance": {"groups": rows, "blocks": rows_b}}
    ctx.save_state("arch", out)
    summary.section("Кандидаты для итога", [
        f"{k}: ADE {v['ade_val']:.2f} м, {v['ms']:.2f} мс, {arch_label(v['params'], n_models, doppler)}"
        for k, v in out.items() if k in ("best", "fast", "default")])
    front_numbers = {t.number for t in front}
    plot_pareto({"стопки": [(t.values[1], t.values[0], t.number in front_numbers) for t in done]},
                ctx.out_dir / "pareto.png", f"Поиск архитектуры: {cases.CASE_TITLES[ctx.case]}",
                xlabel="время сети, мс на обзор")
    return True


def phase_final(ctx: Ctx, cd, ws_tr) -> bool:
    st, summary = ctx.st, ctx.summary
    found = ctx.state("arch")
    if found is None:
        summary.check("Результаты поиска найдены", False, "сначала --phase search")
        return False
    n_models, doppler = len(config.IMM_MODELS), ctx.case != "nodop"
    w = config.TUNE_WINDOWS[0]
    methods, forecasts = [], {name: {} for name in config.EVAL_SETS}
    cands = [("best", "")] + ([("fast", " (быстрая)")] if found["fast"]["params"] != found["best"]["params"] else [])
    for key, suffix in cands:
        info = found[key]
        arch = arch_search.decode(info["params"], n_models, doppler)
        label = arch_label(info["params"], n_models, doppler) + suffix
        hp = {"hidden": arch["hidden"], "dropout": arch["dropout"], "blocks": arch["blocks"], "columns": arch["columns"]}
        out = fit_and_eval(ctx, cd, STAGE, label, "Stack", "hybrid", ws_tr, arch["train"], hp, w)
        methods.append(label)
        for name in config.EVAL_SETS:
            forecasts[name][label] = out[0]["fcs"][name]
        got = out[0]["reports"]["val"]["ade"]
        summary.check(f"{label}: результат поиска воспроизводится", abs(got / info["ade_val"] - 1) <= 0.15,
                      f"ADE val при поиске {info['ade_val']:.2f} м, при итоговом обучении {got:.2f} м")
    ref = [m for m in ("stage2/IMM+GRU",) if ctx.registry.has(m)]
    for name in config.EVAL_SETS:
        _, rts, _ = cd.eval(name)
        summary.section(f"{name}: ADE (м), среднее ± СКО по {len(st.seeds)} seed", method_table(
            ctx.registry, ["stage1/IMM"] + ref + [f"{STAGE}/{m}" for m in methods], name, ref="stage1/IMM"))
        shown = {"IMM": evaluation.imm_forecasts(cd, name), **{m: forecasts[name][m] for m in methods}}
        compare(rts, shown, summary, f"{name} (seed {st.seeds[0]})", ctx.out_dir, name)
    (ctx.out_dir / "metrics.json").write_text(json.dumps(
        {m: {n: {k: v for k, v in ctx.registry.report(f"{STAGE}/{m}", n).items() if k != "flights"}
             for n in config.EVAL_SETS} for m in methods}, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def run_case(ctx: Ctx) -> bool:
    args = ctx.args
    cd = ctx.casedata(net_input=True)
    ws_tr = cd.windows("main", pdata.train_tracks(ctx.sets))
    ok = True
    if args.phase in ("search", "all"):
        ok &= phase_search(ctx, cd, ws_tr, cd.eval("val")[0], args.trials or ctx.st.arch_trials)
    if ok and args.phase in ("final", "all"):
        ok &= phase_final(ctx, cd, ws_tr)
    return ok


def main() -> int:
    return run_stage(STAGE, "Этап 4: поиск архитектуры", run_case, needs_gpu=True, add_args=add_args)


if __name__ == "__main__":
    sys.exit(main())
