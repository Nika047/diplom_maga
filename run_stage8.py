"""
Этап 8: проверка на сгенерированном test - насколько ошибки на сгенерированных траекториях
отличаются от ошибок на реальных полётах.

Новый компонент - сгенерированный test: новые траектории генератора этапа 0 (та же
калибровка, тот же состав сценариев, доля SYNTH_TEST_SHARE от SYNTH_COUNT_TRAIN, поток
"synth_test"), которых не было в обучении. Модели не переобучаются: берутся веса итога
этапа 4 (artifacts/stage4/, 3 seed) и подобранный IMM. На реальном test и на сгенерированном
test считаются одни и те же методы: CV, KF, IMM исходный и подобранный, 4 гибрида, 2 сети
без IMM. Отдельно - доли участков (сгенерированные траектории могут маневрировать чаще) и
ADE по участкам, чтобы разница не объяснялась только составом.

По результату пользователь решает, объединять ли сгенерированные и реальные данные и делить
их на выборки после генерации (PLAN.md, «Этапы 8+»).

Запуск (домашний ПК, веса этапа 4 - только там): .venv\\Scripts\\python.exe run_stage8.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage8.py --smoke

Проверки: сгенерированный test физически согласован и распределён как логи (проверки этапа
0), не пересекается по именам с синтетикой обучения; на реальном test модели повторяют
итог этапа 4.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import generator, phases  # noqa: E402
from data.windows import apply_correction, apply_e2e  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from filters.extrapolation import cv_extrapolate  # noqa: E402
from filters.imm import IMM  # noqa: E402
from models import build_corrector  # noqa: E402
from run_stage0 import check_synth, compare_distributions, hours  # noqa: E402
from run_stage4 import EXPERIMENTS, load_data, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import slug  # noqa: E402
from training.trainer import pick_device, predict  # noqa: E402
from tuning import search  # noqa: E402

SMOKE = {"seeds": (42, 43), "share": 0.04}

STAGE = "stage8"
OUT_DIR = config.RESULTS_DIR / STAGE
SETS = ("реальный test", "сгенерированный test")
PHASES = ("straight", "entry", "turn", "exit")


def load_model(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_feat = len(ck["normalizer"]["x_mean"])
    model = build_corrector(ck["corrector"], n_feat, config.HORIZON_SCANS, ck["window"], **ck["model_kwargs"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), dataset.Normalizer.from_dict(ck["normalizer"]), ck["mode"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 8: проверка на сгенерированном test")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 8: проверка на сгенерированном test" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s", device)
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    art4 = config.ARTIFACTS_DIR / "stage4"
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    share = SMOKE["share"] if args.smoke else config.SYNTH_TEST_SHARE
    t0 = time.time()

    imm_path = cache4 / "imm_best.json"
    if not imm_path.exists():
        summary.check("Подобранный IMM этапа 4 найден", False, f"нет {imm_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    paths = {label: [art4 / f"{slug(f'{label}, seed {s}')}{tag}.pt" for s in seeds] for label, _, _ in EXPERIMENTS}
    found = {label: ps for label, ps in paths.items() if all(p.exists() for p in ps)}
    missing = sorted(set(paths) - set(found))
    summary.check("Веса моделей этапа 4 найдены", bool(found) and (args.smoke or not missing),
                  f"найдены {sorted(found)}" + (f", нет {missing}" if missing else ""))
    if not found:
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1

    # Сгенерированный test
    sets = load_data(args.smoke)
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    counts = {s: max(1, int(round(n * share))) for s, n in config.SYNTH_COUNT_TRAIN.items()}
    synth_test = generator.generate(counts, limits, make_rng("synth_test"), "synthT")
    for tr in synth_test:
        phases.annotate(tr)
    check_synth(synth_test, limits, summary)
    compare_distributions(sets["train"], synth_test, summary)
    clash = {tr.flight_id for tr in synth_test} & {tr.flight_id for tr in sets["synth"]}
    summary.check("Сгенерированный test не пересекается с синтетикой обучения", not clash,
                  f"траекторий {len(synth_test)}, {hours(synth_test):.1f} ч; совпадений имён {len(clash)}")
    # В варианте merged test уже содержит сгенерированные траектории - «реальный test» берётся без них
    tracks = {SETS[0]: [tr for tr in sets["test"] if tr.source != "synth"], SETS[1]: synth_test}

    # Окна (для сетей) и базовые методы на обоих наборах
    window = config.TUNE_WINDOWS[0]
    with search.config_override(imm):
        built = {name: dataset.build(trs, window=window) for name, trs in tracks.items()}
    ks = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
          for name, b in built.items()}
    reports = {name: {} for name in SETS}
    forecasts = {name: {} for name in SETS}
    kf = IMM(("CV",), mu0=(1.0,))
    for name in SETS:
        _, rts, runs = built[name]
        with search.config_override(search.default_imm_params()):
            imm0 = IMM()
            base = {"CV": [cv_extrapolate(rt) for rt in rts], "KF": [kf.run(rt).forecast for rt in rts],
                    "IMM (исходный)": [imm0.run(rt).forecast for rt in rts], "IMM": [r.forecast for r in runs]}
        for method, fcs in base.items():
            fcs = [mask_forecast(f, k) for f, k in zip(fcs, ks[name])]
            forecasts[name][method] = fcs
            reports[name][method] = [summarize_method(rts, fcs)]

    # Сети этапа 4
    for label, ps in found.items():
        for s_i, path in enumerate(ps):
            model, norm, mode = load_model(path, device)
            for name in SETS:
                ws, rts, runs = built[name]
                pred = norm.y_inv(predict(model, norm.x(ws.view(mode).x), device))
                fcs = []
                for i, (rt, run) in enumerate(zip(rts, runs)):
                    sel = ws.track == i
                    fc = (apply_correction(run.forecast, ws.k[sel], ws.psi[sel], pred[sel]) if mode == "hybrid"
                          else apply_e2e(rt, ws.k[sel], ws.psi_e2e[sel], pred[sel]))
                    fcs.append(mask_forecast(fc, ks[name][i]))
                reports[name].setdefault(label, []).append(summarize_method(rts, fcs))
                if s_i == 0:
                    forecasts[name][label] = fcs

    # Повторение этапа 4 на реальном test
    s4 = config.RESULTS_DIR / "stage4" / "final_metrics.json"
    if s4.exists() and config.USE_HOLDOUT:      # в merged test этапа 4 включает сгенерированные траектории
        m4 = json.loads(s4.read_text(encoding="utf-8"))["test"]
        diffs = [abs(np.mean([r["ade"] for r in reports[SETS[0]][lb]]) - np.mean([r["ade"] for r in m4[lb]]))
                 for lb in found if lb in m4]
        summary.check("На реальном test модели повторяют итог этапа 4", bool(diffs) and max(diffs) < 0.05,
                      f"наибольшее расхождение средней ADE {max(diffs) if diffs else float('nan'):.3f} м")

    def mean(name, method, key):
        return float(np.mean([r[key] for r in reports[name][method]]))

    order = [m for m in ("CV", "KF", "IMM (исходный)", "IMM") + tuple(found)]
    lines = [f"{'метод':18s} {'реальный':>9s} {'сгенер.':>9s} {'сгенер./реальн.':>16s} "
             f"{'к IMM реальн.':>14s} {'к IMM сгенер.':>14s}"]
    for m in order:
        a, b = mean(SETS[0], m, "ade"), mean(SETS[1], m, "ade")
        ra, rb = a / mean(SETS[0], "IMM", "ade") - 1, b / mean(SETS[1], "IMM", "ade") - 1
        lines.append(f"{m:18s} {a:9.1f} {b:9.1f} {b / a:16.2f} {ra:+14.1%} {rb:+14.1%}")
    summary.section(f"ADE (м), сети - среднее по {len(seeds)} seed; «к IMM» - к подобранному IMM на том же наборе",
                    lines)
    for name in SETS:
        lines = [f"{'метод':18s}" + "".join(f" {p:>9s}" for p in PHASES)]
        for m in order:
            lines.append(f"{m:18s}" + "".join(f" {mean(name, m, p):9.1f}" for p in PHASES))
        summary.section(f"{name}: ADE по участкам (м)", lines)
    shares = {name: phases.phase_shares(trs) for name, trs in tracks.items()}
    summary.section("Состав наборов: доля точек по участкам", [
        f"{name:22s} " + ", ".join(f"{p} {shares[name][p]:.1%}" for p in PHASES) + f"; {hours(tracks[name]):.1f} ч"
        for name in SETS])
    for name in SETS:
        compare(built[name][1], forecasts[name], summary, f"{name} (seed {seeds[0]})", OUT_DIR,
                f"{'real' if name == SETS[0] else 'synth'}{tag}")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "synth_test.pkl", "wb") as fh:
        pickle.dump(synth_test, fh)
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
