"""
Этап 19: архитектуры этапа 4 на данных с расширением (решение пользователя 26.10.2026).

Основные данные обучения работы - реальные полёты train + их расширение (этап 9) + синтетика;
архитектуры корректора сравнивались на этапе 4, раньше расширения. Новый компонент - тот же
итог этапа 4 на этих данных: IMM+LSTM, IMM+Transformer, IMM+Mamba, GRU и Mamba без IMM обучаются
заново с параметрами генетического подбора этапа 4, 3 seed. IMM+GRU с расширением уже есть
(веса этапа 9). Рядом - те же модели без расширения (веса этапа 4), на тех же обзорах (общие
обзоры - по самому длинному окну, как в итоге этапа 4).

Схемы встраивания сети в фильтр на тех же данных - отдельными запусками этапов 6в и 6г с --aug
(run_stage6c.py --aug, run_stage6d.py --aug).

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage19.py
Смоук (рабочий ПК, нужны смоуки этапов 4 и 9): .venv\\Scripts\\python.exe run_stage19.py --smoke
(в смоуке подобраны только IMM+GRU и GRU без IMM - остальные модели пропускаются).

Проверки: веса этапов 4 и 9 повторяют свои результаты на val; окна обучения - базовые плюс
расширение; проверки обучения из training/experiment.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from data.windows import apply_correction, apply_e2e  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from run_stage4 import EXPERIMENTS, load_data, summarize_method  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage9 import augmented  # noqa: E402
from run_stage11 import reference_ade  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment, slug  # noqa: E402
from training.trainer import pick_device, predict  # noqa: E402
from tuning import search  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage19"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
AUG = "+ все расширения"


def stage4_val(label: str, s_i: int, tag: str):
    """ADE val сети этапа 4 (seed с номером s_i) из итоговых метрик."""
    path = config.RESULTS_DIR / "stage4" / f"final_metrics{tag}.json"
    if not path.exists():
        path = config.RESULTS_DIR / "stage4" / "final_metrics.json"
    if not path.exists():
        return None
    reps = json.loads(path.read_text(encoding="utf-8"))["val"].get(label, [])
    return reps[s_i]["ade"] if s_i < len(reps) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 19: архитектуры на данных с расширением")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 19: архитектуры на данных с расширением" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 19 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    t0 = time.time()

    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    experiments = [e for e in EXPERIMENTS if e[0] in best_all]
    skipped = [e[0] for e in EXPERIMENTS if e[0] not in best_all]
    if not imm_path.exists() or HYB not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    summary.check("Подобраны параметры всех моделей этапа 4", args.smoke or not skipped,
                  f"есть {[e[0] for e in experiments]}" + (f"; нет (пропущены) {skipped}" if skipped else ""))
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]

    sets = load_data(args.smoke)
    windows = sorted({best_all[label]["params"]["window"] for label, _, _ in experiments})
    w_max = windows[-1]
    with search.config_override(imm):
        base_ws, aug_ws = {}, {}
        for w in windows:
            base_ws[w] = dataset.load_or_build(f"train{tag}_w{w}", sets["train"] + sets["synth"], cache4, window=w)
            copies, aug_ws[w] = augmented(sets["train"], w, tag)
        built = {(name, w): dataset.build(sets[name], window=w) for name in EVAL_SETS for w in windows}
    n_aug = {w: sum(len(ws) for ws in aug_ws[w]) for w in windows}
    summary.check("Окна обучения: базовые плюс расширение", all(n_aug[w] > 0 for w in windows),
                  "; ".join(f"окно {w}: базовых {len(base_ws[w])}, расширения {n_aug[w]} ({len(copies)} копий трасс)"
                            for w in windows))

    common = {name: [np.array(sorted(built[(name, w_max)][0].k[built[(name, w_max)][0].track == i]), dtype=int)
                     for i in range(len(sets[name]))] for name in EVAL_SETS}
    rts = {name: built[(name, w_max)][1] for name in EVAL_SETS}
    table = {name: {"IMM": [summarize_method(rts[name], [mask_forecast(r.forecast, k) for r, k in
                                                         zip(built[(name, w_max)][2], common[name])])]}
             for name in EVAL_SETS}
    forecasts = {name: {"IMM": [mask_forecast(r.forecast, k) for r, k in zip(built[(name, w_max)][2], common[name])]}
                 for name in EVAL_SETS}

    def record(method, s_i, window, mode, preds):
        for name in EVAL_SETS:
            ws, rts_w, runs = built[(name, window)]
            fcs = []
            for i, (rt, run) in enumerate(zip(rts_w, runs)):
                sel = ws.track == i
                fc = (apply_correction(run.forecast, ws.k[sel], ws.psi[sel], preds[name][sel]) if mode == "hybrid"
                      else apply_e2e(rt, ws.k[sel], ws.psi_e2e[sel], preds[name][sel]))
                fcs.append(mask_forecast(fc, common[name][i]))
            table[name].setdefault(method, []).append(summarize_method(rts[name], fcs))
            if s_i == 0:
                forecasts[name][method] = fcs

    for label, corrector, mode in experiments:
        window, hp, train_params = search.split_params(best_all[label]["params"])
        params = {**train_params, "epochs": epochs, "patience": patience}
        # Без расширения - веса этапа 4; IMM+GRU с расширением - веса этапа 9
        saved = [(label, "stage4", label)] + ([(f"{label}, {AUG}", "stage9", AUG)] if label == HYB else [])
        for method, art, exp_name in saved:
            ps = [config.ARTIFACTS_DIR / art / f"{slug(f'{exp_name}, seed {s}')}{tag}.pt" for s in seeds]
            if not all(p.exists() for p in ps):
                summary.check(f"{method}: веса этапа {art[5:]} найдены", False, f"нет {[str(p) for p in ps if not p.exists()]}")
                continue
            for s_i, path in enumerate(ps):
                model, norm, _ = load_model(path, device)
                record(method, s_i, window, mode,
                       {name: norm.y_inv(predict(model, norm.x(built[(name, window)][0].view(mode).x), device))
                        for name in EVAL_SETS})
                ref = stage4_val(label, s_i, tag) if art == "stage4" else (
                    reference_ade("stage9", AUG, tag, best_all) if s_i == 0 else None)
                if art == "stage9" and s_i > 0:
                    continue
                got = table["val"][method][-1]["ade"]
                summary.check(f"{method}, seed {seeds[s_i]}: веса повторяют свой результат на val",
                              ref is not None and abs(got / ref - 1) <= 0.005,
                              f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь {got:.2f} м")
        if label == HYB:
            continue
        ws_tr = dataset.concat([base_ws[window], *aug_ws[window]])
        for s_i, seed in enumerate(seeds):
            res = run_experiment(f"{label}, {AUG}, seed {seed}", corrector, mode, ws_tr, built[("val", window)][0],
                                 params, device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                 model_kwargs=hp, overfit=s_i == 0,
                                 extra_sets={n: built[(n, window)][0] for n in EVAL_SETS[1:]})
            record(f"{label}, {AUG}", s_i, window, mode, {"val": res["pred"], **res["extra"]})
        del ws_tr

    def mean(reps, key):
        return float(np.mean([r[key] for r in reps]))

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    for name in EVAL_SETS:
        lines = [f"{'модель':18s} {'без расширения':>15s} {'с расширением':>15s} {'разница':>9s} "
                 f"{'вход: без':>10s} {'вход: с':>9s}"]
        for label, _, _ in experiments:
            a, b = table[name].get(label), table[name].get(f"{label}, {AUG}")
            if not a or not b:
                continue
            lines.append(f"{label:18s} {cell(a, 'ade'):>15s} {cell(b, 'ade'):>15s} "
                         f"{mean(b, 'ade') / mean(a, 'ade') - 1:>+9.1%} {mean(a, 'entry'):10.1f} {mean(b, 'entry'):9.1f}")
        summary.section(f"{name}: ADE (м) без расширения (этап 4) и с расширением, среднее ± СКО по {len(seeds)} seed",
                        lines)
        width = max(len(m) for m in table[name]) + 1
        lines = [f"{'метод':{width}s}" + "".join(f" {c:>11s}" for c in ("ADE", "FDE", "прямая", "вход", "маневр",
                                                                         "выход"))]
        for method, reps in table[name].items():
            lines.append(f"{method:{width}s}" + "".join(f" {cell(reps, k):>11s}"
                                                         for k in ("ade", "fde", "straight", "entry", "turn", "exit")))
        summary.section(f"{name}: все модели (м)", lines)
        source_section(summary, name, table[name])
        main_methods = ["IMM"] + [f"{label}, {AUG}" for label, _, _ in experiments]
        compare(rts[name], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
