"""
Этап 7 (версия 2): сочетания выживших гипотез.

Берутся гипотезы, прошедшие отбор этапа 6 (состояние hyp_<случай>), и все их сочетания из двух и более без
несовместимой пары (три гипотезы + модульная схема). Каждое сочетание - одна сеть (или распознаватель с
экспертами) на корректоре основы; гиперпараметры подбираются генетически (NSGA-II, первая попытка -
параметры основы), затем 3 seed. Итог случая - модель для следующих этапов (состояние final_<случай>):
лучшая по ADE на val среди основы, выживших одиночных гипотез и сочетаний; если она значимо не лучше основы
(правило отбора этапа 6), остаётся основа; среди не значимо худших лучшей выбирается самая быстрая.

  .venv\\Scripts\\python.exe run_stage7.py [--case ...] [--survivors weight,onset] [--smoke]
--survivors - только для проверки конвейера в смоуке: подменяет список выживших.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from common import cases  # noqa: E402
from pipeline import hypotheses as hyp  # noqa: E402
from pipeline import selection  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from pipeline.tables import method_table  # noqa: E402
from run_stage6 import prepare  # noqa: E402
from viz.tuning_plots import plot_bars_ci  # noqa: E402

STAGE = "stage7"


def add_args(parser) -> None:
    parser.add_argument("--survivors", default="", help="проверка конвейера: weight,multi,onset,modular")


def run_case(ctx: Ctx) -> bool:
    st, summary, reg = ctx.st, ctx.summary, ctx.registry
    hs = ctx.state("hyp")
    if hs is None:
        summary.check("Результаты этапа 6 найдены", False, "нет состояния hyp; сначала run_stage6.py")
        return False
    method = hs["base_method"]
    survivors = [h.strip() for h in ctx.args.survivors.split(",") if h.strip()] or list(hs["survivors"])
    bc = hyp.BaseConfig.from_registry(ctx, method)
    cd, data = prepare(ctx, bc)
    w_value = hs.get("weight_w", max(config.ENTRY_WEIGHTS))
    combos = hyp.combinations(survivors)
    state = ctx.state("final_tuned", {})
    singles = {h: f"stage6/{hyp.spec_name({h})}" for h in survivors}
    methods = []
    for spec in combos:
        key = hyp.spec_name(spec)
        if key not in state:
            state[key] = hyp.tune_spec(ctx, spec, bc, {**data}, w_value, st.hyp_trials)
            ctx.save_state("final_tuned", state)
        hp, params = hyp.hp_for(bc, state[key])
        hyp.evaluate_spec(ctx, cd, spec, bc, hp, params, data["ws_tr"], data["tg_tr"], data["norm"], w_value, STAGE, key)
        methods.append(f"{STAGE}/{key}")
    summary.check("Сочетания не содержат несовместимой пары",
                  all(hyp.compatible(c) for c in combos), f"выжившие: {survivors or 'нет'}; сочетаний {len(combos)}")

    candidates = [method] + list(singles.values()) + methods
    pick = selection.choose_base(reg, candidates, "val", st.boot_n)
    final = pick["base"]
    if final != method and not selection.selection_rule(selection.compare_methods(reg, final, method, "val", st.boot_n)):
        final = method              # значимого выигрыша у основы нет - остаётся основа
    rows = sorted(pick["rows"], key=lambda r: r["ade"])
    for n in config.EVAL_SETS:
        summary.section(f"{n}: ADE (м), среднее ± СКО по {len(st.seeds)} seed; к основе",
                        method_table(reg, ["stage1/IMM"] + candidates, n, ref=method))
    summary.section("Итог случая по val (разность с лучшей, 95% интервал по полётам)", [
        f"{r['method']:56s} ADE val {r['ade']:7.2f}, разность {r['diff']:+6.2f} [{r['lo']:+6.2f}, {r['hi']:+6.2f}], "
        f"{r['ms']:5.2f} мс" for r in rows] + [f"лучшая по ADE: {pick['best']}; итоговая модель случая: {final}"
                                               + (" (основа: значимого выигрыша нет)" if final == method else "")])
    summary.check("Итоговая модель случая выбрана", final in candidates, final)
    spec_final = reg.data[final][str(reg.seeds(final)[0])]["extra"].get("spec", [])
    ctx.save_state("final", {"method": final, "base_method": method, "spec": spec_final, "rows": rows,
                             "survivors": survivors, "tuned": state})
    if len(candidates) > 1:
        plot_bars_ci([(m.split("/", 1)[1],) + selection.ade_ci(reg, m, "test", st.boot_n) for m in candidates],
                     ctx.out_dir / "test_ade.png", f"ADE test, основа и сочетания: {cases.CASE_TITLES[ctx.case]}")
    (ctx.out_dir / "metrics.json").write_text(json.dumps(
        {m: {n: {k: v for k, v in reg.report(m, n).items() if k != "flights"} for n in config.EVAL_SETS}
         for m in candidates}, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 7: сочетания гипотез", run_case, needs_gpu=True, add_args=add_args)


if __name__ == "__main__":
    sys.exit(main())
