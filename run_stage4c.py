"""
Этап 4в: размер сети и объём данных обучения - упирается ли точность гибрида в архитектуру.

Новый компонент - две оси эксперимента для корректоров GRU, LSTM, Transformer (Mamba не
берётся: в 5-10 раз дольше, на этапах 3-4 не лучше остальных):
  размер - ширина SCALE_HIDDEN x слоёв SCALE_LAYERS на нынешних данных; остальные
           гиперпараметры (lr, батч, weight decay, dropout) - из подбора этапа 4;
  данные - конфигурация этапа 4 при доле синтетики SCALE_SYNTH: 0 - только реальные полёты
           train, 0,25 и 0,5 - случайная часть нынешней синтетики, 1 - нынешняя (как на этапе
           4), 2 и 4 - нынешняя плюс 1 и 3 новые партии того же состава (поток "synth_extra");
  плюс большая сеть SCALE_BIG на наибольших данных.
IMM - подобранный на этапе 4. Оценка - val, test, holdout на обзорах итога этапа 4, 3 seed.

Новые партии синтетики порождаются генератором этапа 0 с его калибровкой
(cache/stage0/calibration.json); исходные 1150 траекторий не меняются, окна берутся из
кэша этапа 4.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage4c.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage4c.py --smoke

Проверки: новые траектории физически согласованы и распределены как логи (проверки
этапа 0), не совпадают с исходными; на нынешних данных конфигурация этапа 4 повторяет его
результат; проверки обучения из training/experiment.py.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import generator, phases  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import mask_forecast  # noqa: E402
from run_stage0 import check_synth, compare_distributions, hours  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from tuning import search  # noqa: E402
from viz.filter_plots import METHOD_COLORS  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43), "hidden": (32, 64), "layers": (1, 2),
         "synth": (0.0, 0.5, 1.0, 2.0), "big": (64, 2), "extra_step": 25}

STAGE = "stage4c"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.PROJECT_DIR / "artifacts" / STAGE
EVAL_SETS = ("val", "test", "holdout")


def extra_synth(n_batches: int, limits: generator.Limits, step: int) -> list:
    """Партии новой синтетики того же состава, что SYNTH_COUNT_TRAIN (в смоуке - каждая step-я)."""
    counts = {s: max(1, n // step) for s, n in config.SYNTH_COUNT_TRAIN.items()}
    rng = make_rng("synth_extra")
    batches = []
    for b in range(n_batches):
        tracks = generator.generate(counts, limits, rng, f"synthX{b + 1}")
        for tr in tracks:
            phases.annotate(tr)
        batches.append(tracks)
    return batches


def plot_axis(rows: list, key: str, xlabel: str, path: Path, title: str, logx: bool = False) -> None:
    """rows: [(арх., x, [ADE по seed])]."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arch in config.SCALE_ARCHS:
        pts = sorted((x, np.mean(v), np.std(v)) for a, x, v in rows if a == arch)
        if pts:
            xs, ms, ss = map(np.array, zip(*pts))
            ax.errorbar(xs, ms, yerr=ss, fmt="-o", ms=4, capsize=3, color=METHOD_COLORS.get(f"IMM+{arch}"),
                        label=f"IMM+{arch}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"ADE {key}, м")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 4в: размер сети и объём данных")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", default="", help="только эти архитектуры, например GRU,LSTM")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 4в: размер сети и объём данных" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 4в - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    hiddens = SMOKE["hidden"] if args.smoke else config.SCALE_HIDDEN
    layers_list = SMOKE["layers"] if args.smoke else config.SCALE_LAYERS
    fractions = SMOKE["synth"] if args.smoke else config.SCALE_SYNTH
    big = SMOKE["big"] if args.smoke else config.SCALE_BIG
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    archs = [a for a in config.SCALE_ARCHS if not only or a in only]
    t0 = time.time()

    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    missing = [a for a in archs if f"IMM+{a}" not in best_all]
    if not imm_path.exists() or missing:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {missing} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window = config.TUNE_WINDOWS[0]

    # Данные: реальные train, нынешняя синтетика, новые партии
    sets = load_data(args.smoke)
    real, synth = sets["train"], sets["synth"]
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    n_batches = int(np.ceil(max(fractions))) - 1
    extra = extra_synth(n_batches, limits, SMOKE["extra_step"] if args.smoke else 1)
    new = [tr for b in extra for tr in b]
    if new:
        check_synth(new, limits, summary)
        compare_distributions(real, new, summary)
    clash = {tr.flight_id for tr in new} & {tr.flight_id for tr in synth}
    summary.check("Новая синтетика не совпадает с исходной", not clash and len(new) == n_batches * len(
        extra[0] if extra else []), f"новых траекторий {len(new)} в {n_batches} партиях, совпадений имён {len(clash)}")

    with search.config_override(imm):
        ws_base = dataset.load_or_build(f"train{tag}_w{window}", real + synth, cache4, window=window)
        ws_extra = [dataset.load_or_build(f"synth_extra{b + 1}_w{window}", batch, cache, window=window)
                    for b, batch in enumerate(extra)]
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
    n_real = len(real)
    perm = make_rng("scale").permutation(len(synth))

    def train_set(frac: float) -> tuple:
        """Окна train при доле синтетики frac; (набор окон, часы синтетики)."""
        if frac == 1.0:
            return ws_base, hours(synth)       # ровно набор этапа 4, в том же порядке
        if frac < 1.0:
            keep = np.r_[np.arange(n_real), n_real + np.sort(perm[:int(round(frac * len(synth)))])]
            chosen = [synth[i] for i in sorted(perm[:int(round(frac * len(synth)))])]
            return ws_base.subset_tracks(keep), hours(chosen)
        k = int(round(frac)) - 1
        return dataset.concat([ws_base] + ws_extra[:k]), hours(synth) + sum(hours(b) for b in extra[:k])

    ks = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
          for name, b in built.items()}

    def evaluate(res: dict) -> dict:
        preds = {"val": res["pred"], **res["extra"]}
        out = {}
        for name in EVAL_SETS:
            ws, rts, runs = built[name]
            fcs = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                  preds[name][ws.track == i]), ks[name][i])
                   for i, run in enumerate(runs)]
            out[name] = summarize_method(rts, fcs)
        h = res["history"]
        best = int(np.argmin([e["val_loss"] for e in h["epochs"]]))
        out["gap"] = h["epochs"][best]["val_loss"] / max(h["epochs"][best]["train_loss"], 1e-9)
        out["n_params"] = h["n_params"]
        out["seconds"] = h["train_seconds"]
        return out

    results = []   # {"arch", "hidden", "layers", "frac", "hours", "seed", метрики}

    def run(arch, hidden, layers, frac, ws_tr, synth_h, label):
        win, hp, train_params = search.split_params(best_all[f"IMM+{arch}"]["params"])
        hp = {**hp, "hidden": hidden, "layers": layers}
        params = {**train_params, "epochs": epochs, "patience": patience}
        reps = []
        for s_i, seed in enumerate(seeds):
            with search.config_override(imm):
                res = run_experiment(f"{label}, seed {seed}", arch, "hybrid", ws_tr, built["val"][0], params, device,
                                     summary, log, args.smoke, OUT_DIR, ART, window=win, seed=seed, model_kwargs=hp,
                                     overfit=s_i == 0, extra_sets={n: built[n][0] for n in ("test", "holdout")})
            rep = evaluate(res)
            reps.append(rep)
            results.append({"arch": arch, "hidden": hidden, "layers": layers, "frac": frac, "hours": synth_h,
                            "seed": seed, **{n: rep[n] for n in EVAL_SETS}, "gap": rep["gap"],
                            "n_params": rep["n_params"], "seconds": rep["seconds"]})
        return reps

    # Ось «данные» (конфигурация этапа 4) и большая сеть на наибольших данных
    for frac in fractions:
        ws_tr, synth_h = train_set(frac)
        log.info("Синтетика x%.2f: %d окон, синтетика %.1f ч", frac, len(ws_tr), synth_h)
        for arch in archs:
            hp4 = search.split_params(best_all[f"IMM+{arch}"]["params"])[1]
            reps = run(arch, hp4["hidden"], hp4["layers"], frac, ws_tr, synth_h,
                       f"IMM+{arch}, синтетика x{frac:g}")
            if frac == 1.0:
                ga, got = best_all[f"IMM+{arch}"]["ade_val"], reps[0]["val"]["ade"]
                summary.check(f"IMM+{arch}: на данных этапа 4 повторяет его результат", abs(got / ga - 1) <= 0.05,
                              f"ADE val на этапе 4 {ga:.2f} м, здесь (seed {seeds[0]}) {got:.2f} м")
            if frac == max(fractions):
                run(arch, big[0], big[1], frac, ws_tr, synth_h, f"IMM+{arch} {big[0]}x{big[1]}, синтетика x{frac:g}")
        del ws_tr

    # Ось «размер сети» на нынешних данных
    ws_tr, synth_h = train_set(1.0)
    for arch in archs:
        hp4 = search.split_params(best_all[f"IMM+{arch}"]["params"])[1]
        for hidden in hiddens:
            for layers in layers_list:
                if (hidden, layers) == (hp4["hidden"], hp4["layers"]):
                    continue        # эта точка уже есть на оси данных (x1)
                run(arch, hidden, layers, 1.0, ws_tr, synth_h, f"IMM+{arch} {hidden}x{layers}")

    # Сводка
    def agg(rows, key, name="test"):
        v = np.array([r[name][key] for r in rows])
        return f"{v.mean():.1f}±{v.std():.1f}"

    def group(pred):
        out = {}
        for r in results:
            if pred(r):
                out.setdefault((r["arch"], r["hidden"], r["layers"], r["frac"]), []).append(r)
        return out

    hp4s = {a: search.split_params(best_all[f"IMM+{a}"]["params"])[1] for a in archs}
    data_rows = group(lambda r: (r["hidden"], r["layers"]) == (hp4s[r["arch"]]["hidden"], hp4s[r["arch"]]["layers"]))
    size_rows = group(lambda r: r["frac"] == 1.0)
    for title, rows, xs in (("данные (конфигурация этапа 4)", data_rows, "frac"),
                            ("размер сети (синтетика x1)", size_rows, "size")):
        lines = [f"{'арх.':12s} {'ширина':>6s} {'слоёв':>5s} {'синт.':>6s} {'парам.':>8s} {'val':>10s} {'test':>10s} "
                 f"{'holdout':>10s} {'вход test':>10s} {'val/train':>9s} {'обуч., с':>8s}"]
        for (arch, hidden, layers, frac), rs in sorted(rows.items()):
            lines.append(f"{arch:12s} {hidden:6d} {layers:5d} {frac:6g} {rs[0]['n_params']:8d} {agg(rs, 'ade', 'val'):>10s} "
                         f"{agg(rs, 'ade'):>10s} {agg(rs, 'ade', 'holdout'):>10s} {agg(rs, 'entry'):>10s} "
                         f"{np.mean([r['gap'] for r in rs]):9.2f} {np.mean([r['seconds'] for r in rs]):8.0f}")
        summary.section(f"Ось «{title}»: ADE (м), среднее ± СКО по {len(seeds)} seed", lines + [
            "val/train - отношение val loss к train loss в лучшую эпоху (больше 1 - переобучение)"])
    big_rows = group(lambda r: (r["hidden"], r["layers"]) == tuple(big) and r["frac"] == max(fractions))
    summary.section(f"Большая сеть {big[0]}x{big[1]} на наибольших данных (синтетика x{max(fractions):g})", [
        f"{arch:12s} test {agg(rs, 'ade')}, holdout {agg(rs, 'ade', 'holdout')}, вход {agg(rs, 'entry')}"
        for (arch, _, _, _), rs in sorted(big_rows.items())])

    plot_axis([(a, r[0]["hours"] + hours(real), [x["test"]["ade"] for x in r]) for (a, _, _, _), r in data_rows.items()],
              "test", "часов полётов в обучении (реальные + синтетика)", OUT_DIR / f"ade_vs_data{tag}.png",
              "Ошибка от объёма данных (конфигурация этапа 4)")
    plot_axis([(a, r[0]["n_params"], [x["test"]["ade"] for x in r]) for (a, _, _, _), r in size_rows.items()],
              "test", "параметров сети", OUT_DIR / f"ade_vs_size{tag}.png", "Ошибка от размера сети (синтетика x1)",
              logx=True)
    plot_axis([(a, r[0]["n_params"], [x["test"]["entry"] for x in r]) for (a, _, _, _), r in size_rows.items()],
              "test (вход в маневр)", "параметров сети", OUT_DIR / f"entry_vs_size{tag}.png",
              "Ошибка на входе в маневр от размера сети", logx=True)

    summary.section("Данные", [f"реальные train: {n_real} кусков, {hours(real):.1f} ч; синтетика x1: {len(synth)} траекторий, "
                               f"{hours(synth):.1f} ч; новых партий {n_batches} по {len(extra[0]) if extra else 0}"])
    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "synth_extra.pkl", "wb") as fh:
        pickle.dump(new, fh)
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
