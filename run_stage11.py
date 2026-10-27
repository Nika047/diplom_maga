"""
Этап 11 (версия 2): программа - пакеты итоговых моделей для predict.py.

Для случая пишет artifacts/<прогон>/stage11/<случай>/ (pipeline/bundle.py): подмены config, окно, сеть и веса
всех seed итоговой модели этапа 7. Проверка: предсказатель из пакета (FinalPredictor), прогнавший трассы val
по одной, даёт ту же ADE, что запись реестра (3 seed и среднее поправок), и обрабатывает обзор за считанные
миллисекунды (процессор). Пакет - вход predict.py:
  python predict.py --log flight_logs/<файл>.csv --case dop_traj [--ensemble]

  .venv\\Scripts\\python.exe run_stage11.py [--case ...] [--smoke]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common import cases  # noqa: E402
from eval import metrics  # noqa: E402
from pipeline import bundle, hypotheses as hyp  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402

STAGE = "stage11"


def ade_of(pred, rts: list) -> float:
    pairs = []
    for rt in rts:
        out = pred.run(rt)
        masked = np.full_like(out["hyb"], np.nan)
        masked[out["ks"]] = out["hyb"][out["ks"]]
        pairs.append(metrics.forecast_pairs(rt, masked))
    return metrics.summarize(metrics.concat_pairs(pairs))["ade"]


def run_case(ctx: Ctx) -> bool:
    summary, reg = ctx.summary, ctx.registry
    fin = ctx.state("final")
    if fin is None:
        summary.check("Итоговая модель случая выбрана на этапе 7", False, "нет состояния final")
        return False
    method = fin["method"]
    seeds = reg.seeds(method)
    ex = reg.data[method][str(seeds[0])]["extra"]
    cd = ctx.casedata(window=ex["window"], net_input=ex["net_input"])
    out = ctx.paths.artifacts(STAGE, ctx.case)
    members = [bundle.export_member(ctx, method, s, out) for s in seeds]
    label = method.split("/", 1)[1]
    cfg = {"case": ctx.case, "method": method, "label": label, "window": ex["window"], "override": cd.override(),
           "members": members, "spec": ex.get("spec", []), "corrector": ex["corrector"], "hp": ex["hp"],
           "response_ms": {k: ex.get(k) for k in ("imm_ms", "net_cpu_ms", "net_gpu_ms", "total_cpu_ms", "total_gpu_ms")},
           "run": config.RUN_NAME}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    summary.check("Пакет итоговой модели записан", (out / "config.json").exists() and all((out / m["file"]).exists() for m in members),
                  f"{out}, участников {len(members)}")

    rts = cd.eval("val")[1]
    p1 = bundle.FinalPredictor(out, torch.device("cpu"), ensemble=False)
    got = ade_of(p1, rts)
    want = reg.report(method, "val", seeds[0])["ade"]
    summary.check("Предсказатель из пакета (1 seed) повторяет результат этапа 7 на val", abs(got / want - 1) <= 0.005,
                  f"ADE val реестра {want:.2f} м, пакета {got:.2f} м")
    pe = bundle.FinalPredictor(out, torch.device("cpu"), ensemble=True)
    got_e = ade_of(pe, rts)
    summary.section("Пакет", [f"{p1.name}: ADE val {got:.2f} м; {pe.name}: ADE val {got_e:.2f} м",
                              f"подмены config: {cfg['override']}"])
    scans = sum(len(rt.t) for rt in rts[:3])
    t0 = time.perf_counter()
    for rt in rts[:3]:
        p1.run(rt)
    ms = (time.perf_counter() - t0) / scans * 1000
    summary.section("Время обработки пакетом (процессор, весь трек за один проход)", [
        f"{ms:.2f} мс на обзор (ориентир: время отклика реестра {cfg['response_ms'].get('total_cpu_ms')} мс)"])
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 11: пакеты итоговых моделей для программы", run_case)


if __name__ == "__main__":
    sys.exit(main())
