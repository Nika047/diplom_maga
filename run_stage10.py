"""
Этап 10 (версия 2): доверительные интервалы для главных сравнений.

Бутстреп по полётам (eval/bootstrap.py, BOOTSTRAP_N перестановок, уровень BOOTSTRAP_LEVEL): целый полёт - единица
пересэмплирования, сравнение парное (одни и те же полёты в каждой выборке). Для случая:
  - ADE и 95% интервал: IMM, итоговая модель (этап 7), ансамбли (этап 8), корректоры этапов 2 и 4 - val и test;
  - парные разности: итоговая против IMM, итоговая против основы, ансамбль против итоговой;
  - между случаями (одни и те же полёты и обзоры): итоговая и IMM случая против случая без доплера; доплер по
    телеметрии против доплера по траектории.
Разность называется результатом, если её интервал не содержит 0. Итог по всем случаям - final_models.json в
папке этапа: итоговые модели, ADE test с интервалами, время отклика.

  .venv\\Scripts\\python.exe run_stage10.py [--case ...] [--smoke]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from common import cases  # noqa: E402
from pipeline import selection  # noqa: E402
from pipeline.registry import Registry  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from viz.tuning_plots import plot_bars_ci  # noqa: E402

STAGE = "stage10"


def fmt(d: dict) -> str:
    return f"{d['diff']:+7.2f} м ({d['rel']:+6.1%}) [{d['lo']:+6.2f}, {d['hi']:+6.2f}] -> " + (
        "значимо лучше" if d["hi"] < 0 else "значимо хуже" if d["lo"] > 0 else "различие не доказано")


def run_case(ctx: Ctx) -> bool:
    st, summary, reg = ctx.st, ctx.summary, ctx.registry
    fin, ens = ctx.state("final"), ctx.state("ens")
    if fin is None or ens is None:
        summary.check("Итоговая модель и ансамбль выбраны (этапы 7, 8)", False, "нет состояний final / ens")
        return False
    final, base = fin["method"], fin["base_method"]
    mains = ["stage1/IMM", final, ens["method"], ens["seed_ensemble"]]
    if base != final:
        mains.insert(2, base)
    corr = [m for m in reg.methods() if m.split("/", 1)[0] in ("stage2", "stage4") and m not in mains]
    for n in config.EVAL_SETS:
        rows = [(m, *selection.ade_ci(reg, m, n, st.boot_n)) for m in mains + corr]
        summary.section(f"{n}: ADE и 95% интервал по полётам (м)", [
            f"{m:56s} {a:7.2f}  [{lo:7.2f}, {hi:7.2f}]" for m, a, lo, hi in rows])
        plot_bars_ci([(m.split("/", 1)[1], a, lo, hi) for m, a, lo, hi in rows[:len(mains)]],
                     ctx.out_dir / f"ade_ci_{n}.png", f"ADE {n} с 95% интервалом: {cases.CASE_TITLES[ctx.case]}")
    pairs = [("итоговая против IMM", final, "stage1/IMM"), ("ансамбль (жадный) против итоговой", ens["method"], final),
             ("ансамбль seed против итоговой", ens["seed_ensemble"], final)]
    if base != final:
        pairs.append(("итоговая против основы", final, base))
    for n in ("val", "test"):
        lines = []
        for title, a, b in pairs:
            d = selection.compare_methods(reg, a, b, n, st.boot_n)
            lines.append(f"{title:36s} {fmt(d)}")
        summary.section(f"{n}: парные разности ADE по полётам", lines)
    # между случаями
    cross, lines = {}, []
    if ctx.case != "nodop":
        ref = Registry(ctx.paths, "nodop")
        fin0 = ctx.paths.load_state("final_nodop")
        if fin0 and ref.methods():
            for n in ("val", "test"):
                for title, a, ra, b in (("итоговая против итоговой без доплера", final, ref, fin0["method"]),
                                        ("IMM против IMM без доплера", "stage1/IMM", ref, "stage1/IMM")):
                    d = selection.compare_across(reg, a, ra, b, n, st.boot_n)
                    cross[f"{n}: {title}"] = d
                    lines.append(f"{n}: {title:42s} {fmt(d)}")
    if ctx.case == "dop_tele":
        ref = Registry(ctx.paths, "dop_traj")
        fin1 = ctx.paths.load_state("final_dop_traj")
        if fin1 and ref.methods():
            for n in ("val", "test"):
                for title, a, b in (("итоговая против итоговой с доплером по траектории", final, fin1["method"]),
                                    ("IMM против IMM с доплером по траектории", "stage1/IMM", "stage1/IMM")):
                    d = selection.compare_across(reg, a, ref, b, n, st.boot_n)
                    cross[f"{n}: {title}"] = d
                    lines.append(f"{n}: {title:50s} {fmt(d)}")
    summary.section("Между случаями доплера (одни и те же полёты и обзоры)", lines or ["сравнивать не с чем"])

    # итог по случаям
    models = {}
    for case in cases.CASES:
        f = ctx.paths.load_state(f"final_{case}")
        if not f:
            continue
        r = ctx.registry if case == ctx.case else Registry(ctx.paths, case)
        e = ctx.paths.load_state(f"ens_{case}")
        entry = {"final": f["method"], "base": f["base_method"], "spec": f["spec"]}
        for key, m in (("imm", "stage1/IMM"), ("final", f["method"]), ("ensemble", e["method"] if e else None)):
            if m and r.has(m):
                a, lo, hi = selection.ade_ci(r, m, "test", st.boot_n)
                ex = r.data[m][str(r.seeds(m)[0])]["extra"]
                entry[f"{key}_test"] = {"method": m, "ade": a, "lo": lo, "hi": hi,
                                        "total_cpu_ms": ex.get("total_cpu_ms"), "total_gpu_ms": ex.get("total_gpu_ms")}
        models[case] = entry
    ctx.paths.results(STAGE).joinpath("final_models.json").write_text(
        json.dumps({"run": config.RUN_NAME, "cases": models, "cross": {k: v for k, v in cross.items()}},
                   ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    summary.check("Интервалы посчитаны", True, f"BOOTSTRAP_N = {st.boot_n}, уровень {config.BOOTSTRAP_LEVEL}")
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 10: доверительные интервалы", run_case)


if __name__ == "__main__":
    sys.exit(main())
