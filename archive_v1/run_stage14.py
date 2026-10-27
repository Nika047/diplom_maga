"""
Этап 14: несколько вариантов прогноза вместо одного (эксперимент 7 из PLAN.md, «Этапы 8+»).

Новый компонент - корректор с K гипотезами поправки и их вероятностями (models/multi.py,
config.MULTI_HYPOTHESES = 3). Два способа обучения:
  семантические - гипотезы «прямо / влево / вправо», учится гипотеза класса будущего движения
                  истины (data/windows.turn_class, порог config.MULTI_TURN_DEG);
  свободные     - winner takes all: учится ближайшая к истине гипотеза.
Из гипотез строятся три прогноза:
  наиболее вероятная гипотеза - один прогноз, честный;
  смесь - среднее гипотез с их вероятностями, один прогноз, честный;
  лучшая из K задним числом - не прогноз, а оценка: насколько покрывают возможное будущее
  все K вариантов вместе (minADE, как в работах по прогнозу траекторий).
IMM+GRU с гиперпараметрами этапа 4 на двух наборах обучения (решение пользователя 24.10.2026):
набор этапа 4 и «+ все расширения» этапа 9; 3 seed. Один прогноз - веса этапов 4 и 9.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage14.py
Смоук (рабочий ПК, нужны смоуки этапов 4 и 9): .venv\\Scripts\\python.exe run_stage14.py --smoke

Проверки: класс движения на подставных трассах (прямая, разворот вправо и влево); семантические
гипотезы до обучения = IMM; семантический loss учит только гипотезу класса; лучшая из K не
хуже каждой гипотезы; веса этапов 4 и 9 повторяют свои результаты; overfit one batch, модель
обучилась, train loss падает.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data import augment, generator  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction, turn_class  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from models.multi import MultiHypGRU, semantic_loss, wta_loss  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage9 import KINDS  # noqa: E402
from run_stage11 import reference_ade  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import slug  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from tuning import search  # noqa: E402
from viz import train_plots  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage14"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
TRAIN_SETS = {"stage4": "этап 4", "aug": "+ все расширения"}
VARIANTS = {"semantic": "гипотезы прямо / влево / вправо", "wta": "свободные гипотезы"}
OUTPUTS = {"top": "наиболее вероятная", "mix": "смесь", "best": "лучшая из 3 задним числом"}
CLASSES = ("прямо", "влево", "вправо")
BASE_WEIGHTS = {"stage4": ("stage4", HYB, "stage4"), "aug": ("stage9", "+ все расширения", "stage9")}


def name_of(set_key: str, variant: str = None, output: str = None) -> str:
    if variant is None:
        return f"{HYB}, {TRAIN_SETS[set_key]}, один прогноз"
    return f"{HYB}, {TRAIN_SETS[set_key]}, {VARIANTS[variant]}: {OUTPUTS[output]}"


def classes(ws, rts: dict) -> np.ndarray:
    out = np.zeros(len(ws), dtype=np.int64)
    for i, fid in enumerate(ws.flight_ids):
        sel = ws.track == i
        if sel.any():
            out[sel] = turn_class(rts[fid], ws.k[sel])
    return out


def check_known(summary: StageSummary) -> None:
    t = np.arange(60) * config.SCAN_PERIOD
    w = np.radians(8.0)

    def circle(sign):
        psi = sign * w * t
        return np.column_stack([np.cumsum(20 * np.sin(psi)) * 2, np.cumsum(20 * np.cos(psi)) * 2, 0 * t])

    tracks = {"прямо": np.column_stack([0 * t, 20 * t, 0 * t]), "вправо": circle(1.0), "влево": circle(-1.0)}
    got = {name: set(turn_class(SimpleNamespace(truth=tr), np.arange(5, 50)).tolist()) for name, tr in tracks.items()}
    summary.check("Класс движения: известный ответ (прямая, разворот вправо, влево)",
                  got == {"прямо": {0}, "вправо": {2}, "влево": {1}}, str(got))

    m = MultiHypGRU(17, config.HORIZON_SCANS, hidden=16, layers=2)
    x = torch.randn(8, 20, 17)
    out = m(x)
    y = torch.cat([torch.randn(8, 2 * config.HORIZON_SCANS), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1.0])[:, None]], 1)
    shifted = out.detach().clone()
    shifted[0, 2 * config.HORIZON_SCANS:4 * config.HORIZON_SCANS] += 5.0     # пример класса 0, сдвиг гипотезы 1
    loss = semantic_loss()
    same = float(loss(out.detach(), y)) == float(loss(shifted, y))
    zero = float(out[:, :-config.MULTI_HYPOTHESES].abs().max())
    summary.check("Семантические гипотезы: до обучения нулевые (= IMM), loss учит только гипотезу класса",
                  zero == 0.0 and same and out.shape[1] == config.MULTI_HYPOTHESES * (2 * config.HORIZON_SCANS + 1),
                  f"выход {tuple(out.shape)}, max |гипотеза| {zero}, чужая гипотеза на loss не влияет: {same}")


def decode(raw: np.ndarray, norm, n_hyp: int) -> tuple:
    """Выход сети -> гипотезы в метрах (N, K, H, 2) и вероятности (N, K)."""
    hyps = raw[:, :-n_hyp].reshape(len(raw), n_hyp, config.HORIZON_SCANS, 2) * norm.y_std
    logits = raw[:, -n_hyp:]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return hyps, p / p.sum(axis=1, keepdims=True)


def plot_fan(rt, run, ws, hyps, probs, path: Path) -> None:
    """Веер гипотез перед началом маневра: истина, прогноз IMM и K гипотез с вероятностями."""
    onsets = [o for o in rt.onsets if np.isin(o - 1, ws.k)]
    if not onsets:
        return
    k = onsets[len(onsets) // 2] - 1
    i = int(np.flatnonzero(ws.k == k)[0])
    lo, hi = max(k - 15, 0), min(k + config.HORIZON_SCANS + 1, len(rt.t))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rt.truth[lo:hi, 0], rt.truth[lo:hi, 1], "-", color="black", lw=1.5, label="истина")
    ax.plot(rt.meas[lo:k + 1, 0], rt.meas[lo:k + 1, 1], ".", color="grey", ms=5, label="отметки до момента прогноза")
    ax.plot(run.forecast[k, :, 0], run.forecast[k, :, 1], "s--", color="#d32f2f", ms=4, label="IMM")
    colors = ("#2e7d32", "#1976d2", "#ef6c00")
    for j in range(hyps.shape[1]):
        fc = apply_correction(run.forecast, np.array([k]), ws.psi[[i]], hyps[[i], j])[k]
        ax.plot(fc[:, 0], fc[:, 1], "o-", color=colors[j % 3], lw=1 + 4 * probs[i, j], ms=4,
                label=f"{CLASSES[j] if j < 3 else j}: p = {probs[i, j]:.2f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("восток, м")
    ax.set_ylabel("север, м")
    ax.legend(fontsize=7)
    ax.set_title(f"Гипотезы за обзор до начала маневра ({rt.flight_id})", fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 14: несколько вариантов прогноза")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 14: несколько вариантов прогноза" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 14 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache9 = config.CACHE_DIR / f"stage9{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    n_hyp = config.MULTI_HYPOTHESES
    t0 = time.time()

    check_known(summary)
    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all:
        summary.check("Результаты этапа 4 найдены", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window, hp, train_params = search.split_params(best_all[HYB]["params"])
    params = {**train_params, "epochs": epochs, "patience": patience}

    sets = load_data(args.smoke)
    real, synth = sets["train"], sets["synth"]
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    copies = {kind: augment.augment(real, kind, limits)[0] for kind in KINDS}
    with search.config_override(imm):
        ws_base = dataset.load_or_build(f"train{tag}_w{window}", real + synth, cache4, window=window)
        ws_aug = dataset.concat([ws_base] + [dataset.load_or_build(f"aug_{kind}{tag}_w{window}", copies[kind], cache9,
                                                                   window=window) for kind in KINDS])
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
    train_ws = {"stage4": ws_base, "aug": ws_aug}
    rts = {tr.flight_id: observe(tr) for tr in real + synth + [c for cs in copies.values() for c in cs]}
    rts.update({rt.flight_id: rt for name in EVAL_SETS for rt in built[name][1]})
    cls_tr = {key: classes(ws, rts) for key, ws in train_ws.items()}
    cls_ev = {name: classes(built[name][0], rts) for name in EVAL_SETS}
    summary.section("Классы будущего движения (доля окон)", [
        f"{label:24s} " + ", ".join(f"{c} {np.mean(cl == j):.1%}" for j, c in enumerate(CLASSES))
        for label, cl in [(f"train, {TRAIN_SETS[k]}", v) for k, v in cls_tr.items()] + list(cls_ev.items())])
    ks = {name: [np.array(sorted(built[name][0].k[built[name][0].track == i]), dtype=int)
                 for i in range(len(sets[name]))] for name in EVAL_SETS}

    table = {name: {} for name in EVAL_SETS}
    forecasts = {name: {} for name in EVAL_SETS}
    accuracy = {}
    for name in EVAL_SETS:
        fc = [mask_forecast(r.forecast, k) for r, k in zip(built[name][2], ks[name])]
        table[name]["IMM"] = [summarize_method(built[name][1], fc)]
        forecasts[name]["IMM"] = fc

    def record(method, s_i, corr_by_set):
        for name in EVAL_SETS:
            ws, rts_ev, runs = built[name]
            fc = [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                 corr_by_set[name][ws.track == i]), ks[name][i])
                  for i, run in enumerate(runs)]
            table[name].setdefault(method, []).append(summarize_method(rts_ev, fc))
            if s_i == 0:
                forecasts[name][method] = fc

    # Один прогноз - сохранённые веса этапов 4 и 9
    for set_key, (art, name_exp, ref_stage) in BASE_WEIGHTS.items():
        method = name_of(set_key)
        ps = [config.ARTIFACTS_DIR / art / f"{slug(f'{name_exp}, seed {s}')}{tag}.pt" for s in seeds]
        if not all(p.exists() for p in ps):
            summary.check(f"{method}: веса этапа {ref_stage[5:]} найдены", False,
                          f"нет {[str(p) for p in ps if not p.exists()]}")
            continue
        for s_i, path in enumerate(ps):
            model, norm, _ = load_model(path, device)
            record(method, s_i, {name: norm.y_inv(predict(model, norm.x(built[name][0].x), device))
                                 for name in EVAL_SETS})
        ref, got = reference_ade(ref_stage, name_exp, tag, best_all), table["val"][method][0]["ade"]
        summary.check(f"{method}: веса повторяют свой результат на val (этап {ref_stage[5:]})",
                      ref is not None and abs(got / ref - 1) <= 0.005,
                      f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь (seed {seeds[0]}) {got:.2f} м")

    # Несколько гипотез
    best_ok, fan_done = True, False
    for variant in VARIANTS:
        loss_fn = semantic_loss() if variant == "semantic" else wta_loss()
        for set_key, ws_tr in train_ws.items():
            norm = dataset.Normalizer.fit(ws_tr)
            ws_val = built["val"][0]

            def pack(ws, cl):
                return np.concatenate([norm.y(ws.y).reshape(len(ws), -1), cl[:, None]], axis=1).astype(np.float32)

            x_tr, y_tr = norm.x(ws_tr.x), pack(ws_tr, cls_tr[set_key])
            x_val, y_val = norm.x(ws_val.x), pack(ws_val, cls_ev["val"])
            label = f"{HYB}, {TRAIN_SETS[set_key]}, {VARIANTS[variant]}"
            for s_i, seed in enumerate(seeds):
                key = f"{label}, seed {seed}"
                if s_i == 0:
                    rng = make_rng("training", seed)
                    idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
                    torch.manual_seed(seed)
                    m = MultiHypGRU(x_tr.shape[-1], config.HORIZON_SCANS, zero_init=variant == "semantic", **hp)
                    first, last = overfit_one_batch(m, x_tr[idx], y_tr[idx], config.OVERFIT_STEPS, 3e-3, device,
                                                    loss_fn=loss_fn)
                    summary.check(f"{label}: overfit one batch (loss с энтропией, до нуля не обязан)",
                                  last < 0.5 * first, f"loss {first:.4f} -> {last:.4f}")
                rng = make_rng("training", seed)
                torch.manual_seed(seed)
                model = MultiHypGRU(x_tr.shape[-1], config.HORIZON_SCANS, zero_init=variant == "semantic", **hp)
                history, _ = train(model, x_tr, y_tr, x_val, y_val, params, device, rng, loss_fn=loss_fn)
                ep = history["epochs"]
                summary.check(f"{key}: модель обучилась", history["best_val_loss"] < history["val_loss_initial"],
                              f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
                summary.check(f"{key}: train loss падает", len(ep) >= 2 and ep[-1]["train_loss"] < ep[0]["train_loss"],
                              f"{ep[0]['train_loss']:.4f} -> {ep[-1]['train_loss']:.4f} за {len(ep)} эпох")
                file_key = f"{variant}_{set_key}_seed_{seed}{tag}"      # в названии варианта есть «/»
                train_plots.plot_history(history, OUT_DIR / f"loss_{file_key}.png", f"Обучение {key}")
                torch.save({"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "variant": variant,
                            "window": window, "seed": seed, "model_kwargs": hp, "smoke": args.smoke},
                           ART / f"{file_key}.pt")
                outs = {o: {} for o in OUTPUTS}
                for name in EVAL_SETS:
                    ws = built[name][0]
                    hyps, probs = decode(predict(model, norm.x(ws.x), device), norm, n_hyp)
                    err = np.linalg.norm(hyps - ws.y[:, None], axis=-1).mean(axis=-1)       # (N, K), м
                    best = err.argmin(axis=1)
                    outs["top"][name] = hyps[np.arange(len(ws)), probs.argmax(axis=1)]
                    outs["mix"][name] = (probs[:, :, None, None] * hyps).sum(axis=1)
                    outs["best"][name] = hyps[np.arange(len(ws)), best]
                    best_ok &= bool((err[np.arange(len(ws)), best][:, None] <= err + 1e-9).all())
                    if variant == "semantic":
                        rts_ev = built[name][1]
                        hit = probs.argmax(axis=1) == cls_ev[name]
                        is_entry = np.array([rts_ev[t].phase[k] == PHASE_CODE["entry"] for t, k in zip(ws.track, ws.k)],
                                            dtype=bool)
                        accuracy.setdefault((name, set_key), []).append(
                            (float(hit.mean()), float(hit[is_entry].mean()) if is_entry.any() else float("nan")))
                    if name == "test" and variant == "semantic" and not fan_done:
                        rt_i = max(range(len(built[name][1])), key=lambda j: len(built[name][1][j].onsets))
                        sel = ws.track == rt_i
                        sub = SimpleNamespace(k=ws.k[sel], psi=ws.psi[sel])
                        plot_fan(built[name][1][rt_i], built[name][2][rt_i], sub, hyps[sel], probs[sel],
                                 OUT_DIR / f"fan_example{tag}.png")
                        fan_done = True
                for o in OUTPUTS:
                    record(name_of(set_key, variant, o), s_i, outs[o])
    summary.check("Лучшая из гипотез не хуже каждой гипотезы (по ошибке поправки)", best_ok, "все окна, все наборы")

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    for name in EVAL_SETS:
        width = max(len(m) for m in table[name]) + 1
        lines = [f"{'метод':{width}s}" + "".join(f" {c:>11s}" for c in ("ADE", "FDE", "прямая", "вход", "маневр",
                                                                         "выход"))]
        for method, reps in table[name].items():
            lines.append(f"{method:{width}s}" + "".join(f" {cell(reps, k):>11s}"
                                                         for k in ("ade", "fde", "straight", "entry", "turn", "exit")))
        summary.section(f"{name}: ADE (м), сети - среднее ± СКО по {len(seeds)} seed; «лучшая из 3» - не прогноз, "
                        f"а покрытие будущего тремя вариантами", lines)
        source_section(summary, name, table[name])
        main_methods = ["IMM", name_of("stage4"), name_of("stage4", "semantic", "top"),
                        name_of("stage4", "semantic", "mix"), name_of("stage4", "semantic", "best")]
        compare(built[name][1], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")
    summary.section("Семантические гипотезы: доля верно угаданного класса (наиболее вероятная гипотеза), "
                    "среднее по seed", [
                        f"{name:8s} {TRAIN_SETS[k]:18s} все окна {np.mean([a for a, _ in v]):.1%}, "
                        f"вход в маневр {np.mean([b for _, b in v]):.1%}" for (name, k), v in accuracy.items()])

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
