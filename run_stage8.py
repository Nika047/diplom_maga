"""
Этап 8 (версия 2): ансамбли - последний шаг на итоговую модель случая.

Жадный отбор по val (ensemble selection, Caruana и др., 2004): из всех обученных сетей случая (этапы 2, 4, 5,
6, 7; каждая сеть и seed - отдельный участник) по одному добавляется участник (с повторением), при котором ADE
среднего прогноза на val ниже всего; до ENSEMBLE_MAX участников. Размер выбирается как наименьший, у которого
val не хуже лучшего больше чем на ENSEMBLE_TOL. Рядом - ансамбль seed итоговой модели этапа 7. Время отклика
ансамбля = шаг IMM + сумма проходов сетей. Разброс участников - оценка неопределённости (строб, этап 9).
Ансамбль считается «выигрышем», если по правилу отбора (интервал разности ADE с итоговой моделью на val
левее нуля) он лучше неё.

  .venv\\Scripts\\python.exe run_stage8.py [--case ...] [--smoke]
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from common import cases  # noqa: E402
from eval import metrics  # noqa: E402
from pipeline import evaluation, selection  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from pipeline.tables import method_table  # noqa: E402
from viz.plots import _save  # noqa: E402

STAGE = "stage8"
POOL_STAGES = ("stage2", "stage4", "stage5", "stage6", "stage7")


def pair_errors(cd, name: str, fcs: list) -> np.ndarray:
    """Горизонтальные ошибки прогноза (P, 2) по всем парам (k, h) набора; ошибка ансамбля = среднее ошибок участников."""
    _, rts, _ = cd.eval(name)
    return metrics.concat_pairs([metrics.forecast_pairs(rt, fc) for rt, fc in zip(rts, fcs)])["err"][:, :2]


def greedy(errors: list, max_size: int) -> tuple:
    """Жадный отбор с повторением. errors - ошибки участников на val (P, 2). Возвращает (индексы, ADE по размерам)."""
    chosen, total, curve = [], np.zeros_like(errors[0]), []
    for _ in range(max_size):
        best, best_ade = None, np.inf
        for j, e in enumerate(errors):
            ade = float(np.linalg.norm((total + e) / (len(chosen) + 1), axis=1).mean())
            if ade < best_ade:
                best, best_ade = j, ade
        chosen.append(best)
        total += errors[best]
        curve.append(best_ade)
    return chosen, curve


def mean_forecasts(members: list, name: str, reg) -> list:
    """Среднее прогнозов участников [(метод, seed)] на наборе, по трассам."""
    per = [reg.forecasts(m, s)[name] for m, s in members]
    return [np.mean([p[i] for p in per], axis=0) for i in range(len(per[0]))]


def spread(members: list, name: str, reg) -> list:
    """Разброс участников (СКО горизонтального положения, м) по трассам и обзорам: (K, H)."""
    per = [reg.forecasts(m, s)[name] for m, s in members]
    return [np.std([p[i][..., :2] for p in per], axis=0).mean(axis=-1) for i in range(len(per[0]))]


def run_case(ctx: Ctx) -> bool:
    st, summary, reg = ctx.st, ctx.summary, ctx.registry
    fin = ctx.state("final")
    if fin is None:
        summary.check("Итоговая модель случая выбрана на этапе 7", False, "нет состояния final")
        return False
    final = fin["method"]
    cd = ctx.casedata()
    members_all = [(m, s) for m in reg.methods() if m.split("/", 1)[0] in POOL_STAGES
                   for s in reg.seeds(m) if reg.has_forecasts(m, s)]
    err_val = {}
    ref_n = None
    for m, s in members_all:
        e = pair_errors(cd, "val", reg.forecasts(m, s)["val"])
        ref_n = ref_n or len(e)
        if len(e) == ref_n:
            err_val[(m, s)] = e
    members = list(err_val)
    summary.check("Участники ансамбля сравнимы (одни и те же пары прогноз-истина)", len(members) >= 1,
                  f"участников {len(members)} из {len(members_all)}, пар на val {ref_n}")
    max_size = min(config.ENSEMBLE_MAX, 2 if ctx.smoke else config.ENSEMBLE_MAX)
    chosen, curve = greedy([err_val[k] for k in members], max_size)
    best = min(curve)
    size = next(i + 1 for i, v in enumerate(curve) if v <= best * (1.0 + config.ENSEMBLE_TOL))
    picked = [members[j] for j in chosen[:size]]

    def costs(ms: list) -> dict:
        ex = [reg.data[m][str(s)]["extra"] for m, s in ms]
        imm = max(e.get("imm_ms", 0.0) for e in ex)
        net = sum(e.get("net_cpu_ms", e.get("total_cpu_ms", 0.0) - e.get("imm_ms", 0.0)) for e in ex)
        out = {"imm_ms": imm, "net_cpu_ms": net, "total_cpu_ms": imm + net, "n_params": sum(e.get("n_params", 0) for e in ex),
               "members": [list(x) for x in ms]}
        gpu = [e.get("net_gpu_ms") for e in ex]
        if all(g is not None for g in gpu):
            out["total_gpu_ms"] = imm + sum(gpu)
        return out

    def add(label: str, ms: list) -> str:
        fcs = {n: mean_forecasts(ms, n, reg) for n in config.EVAL_SETS}
        reg.record(f"{STAGE}/{label}", 0, evaluation.reports(cd, fcs), costs(ms), fcs=fcs)
        return f"{STAGE}/{label}"

    ens = add(f"ансамбль жадный из {size}", picked)
    seeds_final = [(final, s) for s in reg.seeds(final)]
    ens_seeds = add(f"ансамбль {len(seeds_final)} seed итоговой", seeds_final)
    summary.check("Ансамбль не хуже лучшего участника (неравенство треугольника) на val",
                  reg.metric(ens, "val")[0] <= min(float(np.linalg.norm(err_val[k], axis=1).mean()) for k in picked) + 1e-9,
                  f"ансамбль {reg.metric(ens, 'val')[0]:.2f} м")
    d = selection.compare_methods(reg, ens, final, "val", st.boot_n)
    wins = selection.selection_rule(d)
    lines = [f"размер {i + 1}: ADE val {v:.2f} м" for i, v in enumerate(curve)]
    lines += ["выбран размер %d (в пределах %.0f%% от лучшего): %s" % (size, 100 * config.ENSEMBLE_TOL,
                                                                     "; ".join(f"{m} (seed {s})" for m, s in picked)),
              f"против итоговой модели {final}: разность {d['diff']:+.2f} м ({d['rel']:+.1%}) "
              f"[{d['lo']:+.2f}, {d['hi']:+.2f}] -> {'выигрыш' if wins else 'значимого выигрыша нет'}"]
    summary.section("Жадный отбор по val", lines)
    for n in config.EVAL_SETS:
        summary.section(f"{n}: ADE (м); ансамбли и итоговая модель", method_table(
            reg, ["stage1/IMM", final, ens_seeds, ens], n, ref="stage1/IMM"))
    sp = {n: spread(picked, n, reg) for n in config.EVAL_SETS}
    with open(ctx.paths.cache("state") / f"spread_{ctx.case}.pkl", "wb") as fh:
        pickle.dump(sp, fh)
    ctx.save_state("ens", {"members": [list(x) for x in picked], "size": size, "curve": curve, "final": final,
                           "method": ens, "seed_ensemble": ens_seeds, "wins": bool(wins), "diff": d})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(curve) + 1), curve, "-o")
    ax.axvline(size, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("участников в ансамбле")
    ax.set_ylabel("ADE val, м")
    ax.set_title(f"Жадный отбор ансамбля: {cases.CASE_TITLES[ctx.case]}", fontsize=10)
    _save(fig, ctx.out_dir / "greedy.png")
    (ctx.out_dir / "metrics.json").write_text(json.dumps(
        {m: {n: {k: v for k, v in reg.report(m, n).items() if k != "flights"} for n in config.EVAL_SETS}
         for m in (final, ens_seeds, ens)}, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 8: ансамбли", run_case)


if __name__ == "__main__":
    sys.exit(main())
