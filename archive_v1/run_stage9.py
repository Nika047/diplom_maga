"""
Этап 9: расширение реальных полётов в обучении (эксперимент 2 из PLAN.md, «Этапы 8+»).

Новый компонент - расширение реальных трасс train (data/augment.py): переразметка РЛС
(AUG_RADAR_COPIES копий с другими положением РЛС и шумом), зеркальное отражение, растяжение
по скорости (AUG_SPEED_FACTORS, только в пределах калибровки генератора). Корректор - IMM+GRU
с гиперпараметрами этапа 4, IMM - подобранный там же; меняется только набор обучения:
  этап 4 (реальные + синтетика x1), плюс каждое расширение по отдельности, плюс все вместе;
  только реальные и только реальные плюс все расширения - может ли расширение заменить
  синтетику.
Оценка - на обзорах итога этапа 4 (config.EVAL_SETS), 3 seed. val и test не расширяются.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage9.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage9.py --smoke

Проверки: известный ответ расширений на подставной трассе (отражение сохраняет скорость и
|w| и меняет знак w, растяжение умножает скорость на f при той же w, копия РЛС - та же истина
и другие отметки); копии только из train и не совпадают по именам с val и test; на данных
этапа 4 гибрид повторяет его результат; проверки обучения из training/experiment.py.
"""

import argparse
import json
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
from data import augment, generator, phases  # noqa: E402
from data.track import Track  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import mask_forecast, source_section  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage0 import hours  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import pick_device  # noqa: E402
from tuning import search  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage9"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
KINDS = {"radar": "переразметка РЛС", "mirror": "отражение", "speed": "растяжение по скорости"}
BASE = "этап 4: реальные + синтетика x1"
REAL = "только реальные"


def test_track() -> Track:
    """Подставная трасса: 60 с прямо на север, разворот 10 град/с вправо 90 с, 60 с прямо; 20 м/с."""
    t = np.arange(0.0, 212.0, 2.0)
    w = np.where((t >= 60.0) & (t < 150.0), np.radians(10.0), 0.0)
    heading = np.concatenate([[0.0], np.cumsum(w[:-1] * 2.0)])
    v = 20.0 * np.stack([np.sin(heading), np.cos(heading)], axis=1)
    xy = np.vstack([[0.0, 0.0], np.cumsum(v[:-1] * 2.0, axis=0)])
    pos = np.column_stack([xy, np.full(len(t), 100.0)])
    return phases.annotate(Track("aug_test#0", "real", "test", t, pos, log_file="aug_test.csv"))


def augmented(real: list, window: int, tag: str) -> tuple:
    """
    Расширение для обучения в других этапах (этап 19: 6в, 6г с --aug): копии реальных треков train
    всех видов в порядке KINDS и их окна из кэша этапа 9 (порядок совпадает с окнами). Вызывать
    внутри config_override с параметрами IMM этапа 4. Возвращает (копии, [окна по видам]).
    """
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    copies = {kind: augment.augment(real, kind, limits)[0] for kind in KINDS}
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    ws = [dataset.load_or_build(f"aug_{kind}{tag}_w{window}", cs, cache, window=window) for kind, cs in copies.items()]
    return [c for cs in copies.values() for c in cs], ws


def check_augment(summary: StageSummary) -> None:
    tr = test_track()
    m, s, r = augment.mirror(tr), augment.speed_scale(tr, 1.15), augment.radar_copy(tr, 1)
    ok_w = np.isfinite(tr.omega)
    d_mv, d_mw = np.abs(m.speed - tr.speed).max(), np.abs(m.omega[ok_w] + tr.omega[ok_w]).max()
    summary.check("Отражение: скорость та же, w меняет знак, участки те же",
                  d_mv < 1e-9 and d_mw < 1e-9 and np.array_equal(m.phase, tr.phase) and tr.omega[ok_w].max() > 9.0,
                  f"|dV| {d_mv:.1e} м/с, |w + w'| {d_mw:.1e} град/с, |w| на развороте {tr.omega[ok_w].max():.1f} град/с")
    d_sv, d_sw = np.abs(s.speed / tr.speed - 1.15).max(), np.abs(s.omega[ok_w] - tr.omega[ok_w]).max()
    summary.check("Растяжение x1,15: скорость в 1,15 раза, w та же", d_sv < 1e-9 and d_sw < 1e-9,
                  f"|V'/V - 1,15| {d_sv:.1e}, |w' - w| {d_sw:.1e} град/с")
    a, b = observe(tr), observe(r)
    same = np.abs(a.truth - b.truth).max()
    summary.check("Копия РЛС: та же истина, другие РЛС и отметки",
                  same < 1e-9 and np.linalg.norm(a.radar_pos - b.radar_pos) > 1.0 and np.abs(a.meas - b.meas).max() > 1.0,
                  f"|истина - истина'| {same:.1e} м, РЛС разнесены на {np.linalg.norm(a.radar_pos - b.radar_pos):.0f} м")


def plot_example(real: list, copies: dict, path: Path) -> None:
    """Самая маневренная реальная трасса train и её копии в плане, положения РЛС копий."""
    tr = max(real, key=lambda x: len(x.turn_onsets))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(tr.pos[:, 0], tr.pos[:, 1], "-", color="black", lw=1.6, label="исходная")
    styles = {"mirror": ("tab:blue", "отражение"), "speed": ("tab:orange", "растяжение")}
    for kind, (color, name) in styles.items():
        for i, c in enumerate(x for x in copies[kind] if x.flight_id.startswith(tr.flight_id + "~")):
            ax.plot(c.pos[:, 0], c.pos[:, 1], "-", color=color, lw=1.0, alpha=0.8,
                    label=f"{name}: {c.flight_id.split('~')[1]}")
    radars = [tr] + [x for x in copies["radar"] if x.flight_id.startswith(tr.flight_id + "~")]
    for i, x in enumerate(radars):
        p = observe(x).radar_pos
        ax.plot(p[0], p[1], "^", color="tab:red", ms=8, label="РЛС исходной и копий" if i == 0 else None)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("восток, м")
    ax.set_ylabel("север, м")
    ax.legend(fontsize=7)
    ax.set_title(f"Расширение реальной трассы {tr.flight_id}", fontsize=10)
    _save(fig, path)


def plot_variants(rows: dict, path: Path, merged: bool) -> None:
    """ADE test по вариантам обучения; в merged - отдельно реальная и сгенерированная части."""
    keys = [("ade_real", "реальные"), ("ade_synth", "сгенерированные")] if merged else [("ade", "test")]
    labels = list(rows)
    x = np.arange(len(labels))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for j, (key, name) in enumerate(keys):
        vals = np.array([[r["test"][key] for r in rows[lb]] for lb in labels])
        ax.bar(x + (j - (len(keys) - 1) / 2) * width, vals.mean(axis=1), width, yerr=vals.std(axis=1), capsize=3,
               label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("ADE test, м")
    lo = min(r["test"][k] for rs in rows.values() for r in rs for k, _ in keys)
    ax.set_ylim(lo * 0.9, None)
    ax.legend(fontsize=8)
    ax.set_title("IMM+GRU: ошибка на test в зависимости от набора обучения", fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 9: расширение реальных полётов")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 9: расширение реальных полётов" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 9 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    t0 = time.time()

    check_augment(summary)
    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window, hp, train_params = search.split_params(best_all[HYB]["params"])
    params = {**train_params, "epochs": epochs, "patience": patience}

    # Расширения реальных трасс train
    sets = load_data(args.smoke)
    real, synth = sets["train"], sets["synth"]
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    copies, data_lines = {}, [f"реальные train: {len(real)} трасс, {hours(real):.1f} ч; синтетика x1: {len(synth)} "
                              f"траекторий, {hours(synth):.1f} ч"]
    for kind, name in KINDS.items():
        copies[kind], skipped = augment.augment(real, kind, limits)
        data_lines.append(f"{name}: {len(copies[kind])} копий, {hours(copies[kind]):.1f} ч"
                          + (f"; вне пределов калибровки и не взято {skipped} из {len(real) * len(config.AUG_SPEED_FACTORS)}"
                             if kind == "speed" else ""))
    eval_ids = {tr.flight_id for name in EVAL_SETS for tr in sets[name]}
    train_logs = {tr.log_file for tr in real}
    all_copies = [c for cs in copies.values() for c in cs]
    clash = {c.flight_id for c in all_copies} & eval_ids
    foreign = {c.log_file for c in all_copies} - train_logs
    summary.check("Расширения - только из train, имена не совпадают с val и test",
                  bool(all_copies) and not clash and not foreign,
                  f"копий {len(all_copies)}; совпадений имён {len(clash)}, логов не из train {len(foreign)}")
    plot_example(real, copies, OUT_DIR / f"augment_example{tag}.png")

    with search.config_override(imm):
        ws_base = dataset.load_or_build(f"train{tag}_w{window}", real + synth, cache4, window=window)
        ws_aug = {kind: dataset.load_or_build(f"aug_{kind}{tag}_w{window}", cs, cache, window=window)
                  for kind, cs in copies.items()}
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
    ks = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
          for name, b in built.items()}
    ws_real = ws_base.subset_tracks(np.arange(len(real)))
    variants = {BASE: lambda: ws_base}
    variants.update({f"+ {name}": (lambda k=kind: dataset.concat([ws_base, ws_aug[k]])) for kind, name in KINDS.items()})
    variants["+ все расширения"] = lambda: dataset.concat([ws_base, *ws_aug.values()])
    variants[REAL] = lambda: ws_real
    variants[f"{REAL} + все расширения"] = lambda: dataset.concat([ws_real, *ws_aug.values()])

    def evaluate(res: dict) -> dict:
        preds = {"val": res["pred"], **res["extra"]}
        out = {}
        for name in EVAL_SETS:
            ws, rts, runs = built[name]
            fcs = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                  preds[name][ws.track == i]), ks[name][i])
                   for i, run in enumerate(runs)]
            out[name] = summarize_method(rts, fcs)
        return out

    rows, sizes = {}, {}
    for label, make in variants.items():
        ws_tr = make()
        sizes[label] = len(ws_tr)
        log.info("%s: %d окон", label, len(ws_tr))
        for s_i, seed in enumerate(seeds):
            with search.config_override(imm):
                res = run_experiment(f"{label}, seed {seed}", "GRU", "hybrid", ws_tr, built["val"][0], params, device,
                                     summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                     model_kwargs=hp, overfit=s_i == 0,
                                     extra_sets={n: built[n][0] for n in EVAL_SETS[1:]})
            rep = evaluate(res)
            rep["seconds"] = res["history"]["train_seconds"]
            rows.setdefault(label, []).append(rep)
            if label == BASE and s_i == 0:
                ga, got = best_all[HYB]["ade_val"], rep["val"]["ade"]
                summary.check("На данных этапа 4 гибрид повторяет его результат", abs(got / ga - 1) <= 0.05,
                              f"ADE val на этапе 4 {ga:.2f} м, здесь (seed {seed}) {got:.2f} м")
        del ws_tr

    def agg(rs, name, key):
        v = np.array([r[name][key] for r in rs])
        return f"{v.mean():.1f}±{v.std():.1f}"

    cols = [(n, "ade") for n in EVAL_SETS] + [("test", "entry"), ("test", "turn")]
    width = max(len(lb) for lb in rows) + 1
    lines = [f"{'обучение':{width}s} {'окон':>7s}" + "".join(f" {f'{n} {k}' if k != 'ade' else n:>12s}" for n, k in cols)
             + f" {'обуч., с':>8s}"]
    base_ade = np.mean([r["test"]["ade"] for r in rows[BASE]])
    for lb, rs in rows.items():
        lines.append(f"{lb:{width}s} {sizes[lb]:7d}" + "".join(f" {agg(rs, n, k):>12s}" for n, k in cols)
                     + f" {np.mean([r['seconds'] for r in rs]):8.0f}")
    lines.append(f"entry, turn - ADE test на входе в маневр и на маневре; test этапа 4 здесь {base_ade:.1f} м")
    summary.section(f"IMM+GRU: ADE (м) по набору обучения, среднее ± СКО по {len(seeds)} seed", lines)
    for name in EVAL_SETS:
        source_section(summary, name, {lb: [r[name] for r in rs] for lb, rs in rows.items()})
    summary.section("Данные", data_lines)
    plot_variants(rows, OUT_DIR / f"ade_by_train_set{tag}.png", "ade_real" in rows[BASE][0]["test"])

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
