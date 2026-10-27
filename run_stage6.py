"""
Этап 6 (версия 2): гипотезы по одной против основы.

Основа - корректор, выбранный этапом 5 (состояние base_<случай>, "corrector_method"; если лучшая схема не
корректор после IMM, берётся лучший корректор этапов 2 и 4). Гипотезы (pipeline/hypotheses.py):
  вес ошибкам на входе в маневр (W из ENTRY_WEIGHTS, берётся лучший по val), три гипотезы прогноза,
  прогноз начала маневра, модульная схема. Для гипотез, меняющих устройство сети (три гипотезы, начало
  маневра, модульная схема), гиперпараметры подбираются генетически (NSGA-II, первая попытка - параметры
  основы); вес меняет только функцию потерь, параметры берутся с основы. 3 seed.

Отбор по val: гипотеза проходит, если разность ADE с основой отрицательна и её 95% интервал бутстрепа по
полётам лежит левее нуля. По test не отбираем. Выжившие идут в этап 7 (сочетания).
Метрики входа в маневр, угадывание класса гипотез, AP начала маневра, точность распознавателя - в сводке.

  .venv\\Scripts\\python.exe run_stage6.py [--case ...] [--smoke]
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from common import cases  # noqa: E402
from common.cases import config_override  # noqa: E402
from pipeline import data as pdata  # noqa: E402
from pipeline import hypotheses as hyp  # noqa: E402
from pipeline import selection  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from pipeline.tables import method_table  # noqa: E402
from radar.model import observe  # noqa: E402
from training import dataset  # noqa: E402
from viz.tuning_plots import plot_bars_ci, plot_tradeoff  # noqa: E402

STAGE = "stage6"


def prepare(ctx: Ctx, bc: hyp.BaseConfig) -> tuple:
    """Данные основы: окна и цели обучения и val, нормализатор."""
    cd = ctx.casedata(window=bc.window, net_input=bc.net_input)
    tracks_tr = pdata.train_tracks(ctx.sets)
    ws_tr = cd.windows("main", tracks_tr)
    ws_val, rts_val, _ = cd.eval("val")
    with config_override(cd.override()):
        rts_tr = [observe(tr) for tr in tracks_tr]
    data = {"ws_tr": ws_tr, "tg_tr": hyp.targets_for(ws_tr, rts_tr), "ws_val": ws_val,
            "tg_val": hyp.targets_for(ws_val, rts_val), "norm": dataset.Normalizer.fit(ws_tr)}
    return cd, data


def alias(reg, src: str, dst: str) -> None:
    """Метод dst - копия записей метода src (лучший вариант веса)."""
    reg.data[dst] = copy.deepcopy(reg.data[src])
    for seed in reg.seeds(src):
        if reg.has_forecasts(src, seed):
            reg.save_forecasts(dst, seed, reg.forecasts(src, seed))
    reg.flush()


def run_case(ctx: Ctx) -> bool:
    st, summary, reg = ctx.st, ctx.summary, ctx.registry
    base = ctx.state("base")
    if base is None:
        summary.check("Основа выбрана на этапе 5", False, "нет состояния base; сначала run_stage5.py")
        return False
    method = base["corrector_method"]
    bc = hyp.BaseConfig.from_registry(ctx, method)
    cd, data = prepare(ctx, bc)
    state = ctx.state("hyp", {})
    tuned = state.get("tuned", {})
    methods = {}

    def run(spec, label, w_value, hp, params) -> str:
        name = f"{STAGE}/{label}"
        hyp.evaluate_spec(ctx, cd, spec, bc, hp, params, data["ws_tr"], data["tg_tr"], data["norm"], w_value, STAGE, label)
        return name

    # вес ошибкам: оба W, лучший по val
    weight_runs = {w: run({"weight"}, f"вес x{w:g}", w, bc.hp, bc.params) for w in config.ENTRY_WEIGHTS}
    best_w = min(weight_runs, key=lambda w: reg.metric(weight_runs[w], "val")[0])
    alias(reg, weight_runs[best_w], f"{STAGE}/{hyp.SHORT['weight']}")
    methods["weight"] = f"{STAGE}/{hyp.SHORT['weight']}"
    state["weight_w"] = best_w
    # гипотезы, меняющие устройство сети: подбор, затем 3 seed
    for h in hyp.STRUCTURAL:
        spec = frozenset({h})
        key = hyp.spec_name(spec)
        if key not in tuned:
            tuned[key] = hyp.tune_spec(ctx, spec, bc, {**data}, 1.0, st.hyp_trials)
            state["tuned"] = tuned
            ctx.save_state("hyp", state)
        hp, params = hyp.hp_for(bc, tuned[key])
        methods[h] = run(spec, key, 1.0, hp, params)
    state["tuned"] = tuned

    # отбор по val
    rows, survivors = [], []
    for h, m in methods.items():
        d = selection.compare_methods(reg, m, method, "val", st.boot_n)
        d_test = selection.compare_methods(reg, m, method, "test", st.boot_n)
        ok = selection.selection_rule(d)
        rows.append({"hyp": h, "method": m, "diff": d["diff"], "lo": d["lo"], "hi": d["hi"], "rel": d["rel"],
                     "test_diff": d_test["diff"], "survived": ok})
        if ok:
            survivors.append(h)
        summary.check(f"{hyp.TITLES[h]}: решение принято по правилу (интервал разности ADE с основой на val)", True,
                      f"разность {d['diff']:+.2f} м ({d['rel']:+.1%}), интервал [{d['lo']:+.2f}, {d['hi']:+.2f}] -> "
                      f"{'проходит' if ok else 'не проходит'}")
    state.update({"base_method": method, "survivors": survivors, "rows": rows})
    ctx.save_state("hyp", state)

    names = ["stage1/IMM", method] + list(methods.values())
    for n in config.EVAL_SETS:
        summary.section(f"{n}: ADE (м), среднее ± СКО по {len(st.seeds)} seed; к основе", method_table(reg, names, n, ref=method))
    summary.section("Отбор гипотез по val (разность с основой, 95% интервал по полётам)", [
        f"{hyp.TITLES[r['hyp']]:34s} разность {r['diff']:+7.2f} м ({r['rel']:+6.1%}) [{r['lo']:+6.2f}, {r['hi']:+6.2f}], "
        f"test {r['test_diff']:+7.2f} м -> {'ПРОХОДИТ' if r['survived'] else 'отброшена'}" for r in rows]
        + [f"выжившие: {', '.join(hyp.TITLES[h] for h in survivors) or 'нет'}; вес выбран x{best_w:g}"])
    extra_lines = []
    for h, m in methods.items():
        ex = reg.data[m][str(reg.seeds(m)[0])]["extra"]
        keys = [k for k in ex if k.endswith(("class_acc", "class_acc_entry", "best_of_k_ade", "onset_ap",
                                              "recognizer_acc", "recognizer_acc_entry"))]
        if keys:
            extra_lines.append(f"{hyp.TITLES[h]}: " + ", ".join(f"{k} {ex[k]:.3f}" for k in sorted(keys)))
    summary.section("Прочие показатели гипотез (первый seed)", extra_lines or ["нет"])
    summary.section("Подбор для гипотез (ADE val по поправкам, м; основа - первая попытка)", [
        f"{k}: основа {v['ade_val_base']:.2f} -> лучшее {v['ade_val']:.2f}; hp {v['hp']}; обучение {v['params']}"
        for k, v in tuned.items()])
    plot_bars_ci([(m.split("/", 1)[1],) + selection.ade_ci(reg, m, "test", st.boot_n) for m in [method] + list(methods.values())],
                 ctx.out_dir / "test_ade.png", f"ADE test, гипотезы против основы: {cases.CASE_TITLES[ctx.case]}")
    plot_tradeoff({m.split("/", 1)[1]: (reg.metric(m, "test")[0], reg.metric(m, "test", "entry")[0])
                   for m in [method] + list(methods.values())}, ctx.out_dir / "tradeoff.png",
                  f"Средняя ошибка и ошибка на входе в маневр, test: {cases.CASE_TITLES[ctx.case]}")
    (ctx.out_dir / "metrics.json").write_text(json.dumps(
        {m: {n: {k: v for k, v in reg.report(m, n).items() if k != "flights"} for n in config.EVAL_SETS}
         for m in [method] + list(methods.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 6: гипотезы против основы", run_case, needs_gpu=True)


if __name__ == "__main__":
    sys.exit(main())
