"""
Этап 6в: сравнение схем гибридизации фильтра и нейросети (вариант Б).

Новый компонент - схемы, в которых сеть встроена в фильтр по-разному (models/hybrids.py):
  A  - поправка к прогнозу IMM после фильтра (IMM+GRU этапа 4, обучается заново как эталон);
  Б1 - сеть выбирает модель: GRU пересчитывает веса моделей CV и CT в прогнозе IMM вместо
       mu_k P^h (ModelWeightGRU); до обучения = IMM;
  Б2 - сеть внутри фильтра по типу KalmanNet: фильтр Калмана с моделью CV, GRU добавляет
       поправку к коэффициенту усиления (NeuralKF, mode="gain"); до обучения = фильтр Калмана;
  Б3 - сеть подстраивает шум: тот же фильтр, GRU задаёт СКО шума ускорения на каждом
       обзоре (NeuralKF, mode="noise"); до обучения = фильтр Калмана.
A и Б1 - на окнах этапа 4 с гиперпараметрами IMM+GRU; Б2 и Б3 - на кусках трасс по
HYB_CHUNK_SCANS обзоров, шаг обучения выбирается по val из KF_LRS (подбор скромнее, чем у
A и Б1 - это оговорка к сравнению). Оценка на одних и тех же обзорах val, test, holdout (как в итоге
этапа 4), 3 seed. Только P_d = 1.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage6c.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage6c.py --smoke

Проверки: Б1 до обучения = IMM; фильтр на PyTorch без сети близок к фильтру Калмана
этапа 1; loss на одном батче падает (Б1-Б3 ограничены схемой и до нуля не обязаны);
модели обучились; A повторяет этап 4.
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
from common.rng import make_rng  # noqa: E402
from data.windows import apply_correction, model_offsets, sequence_arrays, sequence_chunks  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from filters.imm import IMM  # noqa: E402
from models.hybrids import ModelWeightGRU, NeuralKF, model_weight_loss, sequence_loss  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from tuning import search  # noqa: E402
from viz import train_plots  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
KF_TRAIN = {"batch": 128, "lr": 1e-3, "weight_decay": 1e-5, "epochs": 40, "patience": 6}
KF_LRS = (1e-3, 3e-3, 1e-2)   # Б2, Б3: шаг выбирается по val при первом seed (у A и Б1 - подбор этапа 4)
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43), "chunk": 30, "lrs": (1e-3, 1e-2)}

STAGE = "stage6c"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.PROJECT_DIR / "artifacts" / STAGE
EVAL_SETS = ("val", "test", "holdout")
HYB = "IMM+GRU"
NAMES = {"A": "A. IMM+GRU, поправка после", "B1": "Б1. IMM, сеть выбирает модель",   # без «:» - идут в имена файлов
         "B2": "Б2. KF, сеть в усилении", "B3": "Б3. KF, сеть задаёт шум"}
KF2D = "KF 2D на PyTorch, без сети"


def offsets_for(ws, runs: list, p_pow: np.ndarray) -> tuple:
    n, h, m = len(ws), config.HORIZON_SCANS, p_pow.shape[1]
    d, logw = np.zeros((n, h, m, 2), np.float32), np.zeros((n, h, m), np.float32)
    for i, run in enumerate(runs):
        sel = ws.track == i
        d[sel], logw[sel] = model_offsets(run, ws.k[sel], ws.psi[sel], p_pow)
    return d, logw


def kf_forecast(model, rt, imm_forecast: np.ndarray, device) -> np.ndarray:
    """Прогноз фильтра с сетью по всей трассе (K, H, 3); высота - от IMM (метрики - по горизонтали)."""
    s = sequence_arrays(rt)
    origin = s["z"][0]
    seq = np.concatenate([s["z"] - origin, s["r"].reshape(-1, 4), s["det"][:, None]], axis=1).astype(np.float32)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(seq)[None].to(device))[0].cpu().numpy().astype(float) + origin
    return np.concatenate([out, imm_forecast[..., 2:3]], axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 6в: схемы гибридизации")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 6в: схемы гибридизации" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 6в - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    chunk = SMOKE["chunk"] if args.smoke else config.HYB_CHUNK_SCANS
    kf_params = {**KF_TRAIN, **({"epochs": epochs, "patience": patience} if args.smoke else {})}
    kf_lrs = SMOKE["lrs"] if args.smoke else KF_LRS
    t0 = time.time()

    imm_path, best_path = cache4 / "imm_best.json", cache4 / "best_params.json"
    best_all = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    if not imm_path.exists() or HYB not in best_all or config.RADAR_PD != 1.0:
        summary.check("Результаты этапа 4 найдены, P_d = 1", False, f"нужны {imm_path} и {HYB} в {best_path}")
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1
    imm = json.loads(imm_path.read_text(encoding="utf-8"))["params"]
    window, model_kwargs, train_params = search.split_params(best_all[HYB]["params"])
    params = {**train_params, "epochs": epochs, "patience": patience}

    sets = load_data(args.smoke)
    tracks_tr = sets["train"] + sets["synth"]
    with search.config_override(imm):
        ws_tr = dataset.load_or_build(f"train{tag}_w{window}", tracks_tr, cache4, window=window)
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
        imm_m = IMM()
        log.info("Прогон IMM с прогнозами моделей: train %d трасс", len(tracks_tr))
        rts_tr = [observe(tr) for tr in tracks_tr]
        runs_tr = [imm_m.run(rt, with_models=True) for rt in rts_tr]
        runs_ev = {name: [imm_m.run(rt, with_models=True) for rt in built[name][1]] for name in EVAL_SETS}
    p_pow = np.stack(imm_m.p_pow)
    n_models = p_pow.shape[1]
    ks = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
          for name, b in built.items()}
    forecasts = {name: {"IMM": [mask_forecast(r.forecast, k) for r, k in zip(built[name][2], ks[name])]}
                 for name in EVAL_SETS}
    kf = IMM(("CV",), mu0=(1.0,))
    for name in EVAL_SETS:
        forecasts[name]["KF"] = [mask_forecast(kf.run(rt).forecast, k) for rt, k in zip(built[name][1], ks[name])]
    reports = {name: {m: [summarize_method(built[name][1], forecasts[name][m])] for m in ("IMM", "KF")}
               for name in EVAL_SETS}
    costs = {}

    def evaluate(method, fcs_by_set, s_i):
        for name in EVAL_SETS:
            reports[name].setdefault(method, []).append(summarize_method(built[name][1], fcs_by_set[name]))
            if s_i == 0:
                forecasts[name][method] = fcs_by_set[name]

    def hybrid_fcs(name, corr):
        ws, _, runs = built[name]
        return [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                               corr[ws.track == i]), ks[name][i]) for i, run in enumerate(runs)]

    # Б1: окна этапа 4 + смещения прогнозов моделей
    norm = dataset.Normalizer.fit(ws_tr)
    d_tr, lw_tr = offsets_for(ws_tr, runs_tr, p_pow)
    ev = {name: offsets_for(built[name][0], runs_ev[name], p_pow) for name in EVAL_SETS}

    def b1_target(ws, d, lw):
        return np.concatenate([norm.y(ws.y).reshape(len(ws), -1), norm.y(d).reshape(len(ws), -1),
                               lw.reshape(len(ws), -1)], axis=1).astype(np.float32)

    def b1_correction(logits, d, lw):
        z = lw + logits.reshape(lw.shape)
        w = np.exp(z - z.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        return (w[..., None] * d).sum(axis=2)          # в метрах: d не нормирован

    x_tr, y1_tr = norm.x(ws_tr.x), b1_target(ws_tr, d_tr, lw_tr)
    x_ev = {name: norm.x(built[name][0].x) for name in EVAL_SETS}
    y1_val = b1_target(built["val"][0], *ev["val"])
    loss_b1 = model_weight_loss(config.HORIZON_SCANS, n_models)

    # Б2, Б3: куски трасс
    seqs_tr = [sequence_arrays(rt) for rt in rts_tr]
    xs_tr, ys_tr = sequence_chunks(seqs_tr, chunk)
    xs_val, ys_val = sequence_chunks([sequence_arrays(rt) for rt in built["val"][1]], chunk)
    loss_kf = sequence_loss(config.WARMUP_SCANS, config.HYB_SCALE_M)
    log.info("Куски трасс по %d обзоров: train %d, val %d", chunk, len(xs_tr), len(xs_val))

    # Фильтр без сети против фильтра Калмана этапа 1
    plain = NeuralKF("noise", config.HYB_KF_SIGMA_A, config.HYB_HIDDEN).to(device)
    fcs = {name: [mask_forecast(kf_forecast(plain, rt, f, device), k)
                  for rt, f, k in zip(built[name][1], forecasts[name]["IMM"], ks[name])] for name in EVAL_SETS}
    evaluate(KF2D, fcs, 0)
    a2, a3 = reports["val"][KF2D][0]["ade"], reports["val"]["KF"][0]["ade"]
    summary.check("Фильтр на PyTorch без сети близок к фильтру Калмана этапа 1", abs(a2 / a3 - 1) <= 0.05,
                  f"ADE val {a2:.2f} м против {a3:.2f} м (2D против 3D: у ковариации отметки есть связь с высотой)")

    # Б2, Б3: выбор шага обучения по val (seed - первый)
    kf_lr, lr_lines = {}, []
    for code, mode in (("B2", "gain"), ("B3", "noise")):
        val_loss = {}
        for lr in kf_lrs:
            rng = make_rng("training", seeds[0])
            torch.manual_seed(seeds[0])
            m = NeuralKF(mode, config.HYB_KF_SIGMA_A, config.HYB_HIDDEN)
            log.info("Выбор шага %s: lr %g", NAMES[code], lr)
            hist, _ = train(m, xs_tr, ys_tr, xs_val, ys_val, {**kf_params, "lr": lr}, device, rng, quiet=True,
                            loss_fn=loss_kf)
            val_loss[lr] = hist["best_val_loss"]
        kf_lr[code] = min(val_loss, key=val_loss.get)
        lr_lines.append(f"{NAMES[code]}: " + ", ".join(f"lr {lr:g} -> val loss {v:.4f}" for lr, v in val_loss.items())
                        + f"; выбран {kf_lr[code]:g}")

    with search.config_override(imm):
        for s_i, seed in enumerate(seeds):
            # A
            res = run_experiment(f"{NAMES['A']}, seed {seed}", "GRU", "hybrid", ws_tr, built["val"][0], params,
                                 device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                 model_kwargs=model_kwargs, overfit=False,
                                 extra_sets={n: built[n][0] for n in ("test", "holdout")})
            preds = {"val": res["pred"], **res["extra"]}
            evaluate(NAMES["A"], {name: hybrid_fcs(name, preds[name]) for name in EVAL_SETS}, s_i)
            costs.setdefault(NAMES["A"], []).append((res["history"]["n_params"], res["history"]["train_seconds"]))
            if s_i == 0:
                ga, got = best_all[HYB]["ade_val"], reports["val"][NAMES["A"]][0]["ade"]
                summary.check("A повторяет этап 4", abs(got / ga - 1) <= 0.05,
                              f"ADE val на этапе 4 {ga:.2f} м, здесь (seed {seed}) {got:.2f} м")

            # Б1
            key = f"{NAMES['B1']}, seed {seed}"
            torch.manual_seed(seed)
            model = ModelWeightGRU(x_tr.shape[-1], config.HORIZON_SCANS, n_models, **model_kwargs).to(device)
            if s_i == 0:
                zero = np.abs(b1_correction(predict(model, x_ev["val"], device), *ev["val"])).max()
                summary.check("Б1 до обучения = IMM", zero < 1e-3, f"max |поправка| {zero:.2e} м")
                rng = make_rng("training", seed)
                idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
                torch.manual_seed(seed)
                m = ModelWeightGRU(x_tr.shape[-1], config.HORIZON_SCANS, n_models, **model_kwargs)
                first, last = overfit_one_batch(m, x_tr[idx], y1_tr[idx], config.OVERFIT_STEPS, 3e-3, device,
                                                loss_fn=loss_b1)
                summary.check("Б1: loss на одном батче падает", last < first, f"loss {first:.4f} -> {last:.4f}")
                torch.manual_seed(seed)
                model = ModelWeightGRU(x_tr.shape[-1], config.HORIZON_SCANS, n_models, **model_kwargs).to(device)
            rng = make_rng("training", seed)
            t1 = time.time()
            history, _ = train(model, x_tr, y1_tr, x_ev["val"], y1_val, params, device, rng, loss_fn=loss_b1)
            costs.setdefault(NAMES["B1"], []).append((sum(p.numel() for p in model.parameters()), time.time() - t1))
            summary.check(f"{key}: модель обучилась", history["best_val_loss"] < history["val_loss_initial"],
                          f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
            train_plots.plot_history(history, OUT_DIR / f"loss_B1_{seed}{tag}.png", f"Обучение: {key}")
            fcs = {name: hybrid_fcs(name, b1_correction(predict(model, x_ev[name], device), *ev[name]))
                   for name in EVAL_SETS}
            evaluate(NAMES["B1"], fcs, s_i)

            # Б2, Б3
            for code, mode in (("B2", "gain"), ("B3", "noise")):
                key = f"{NAMES[code]}, seed {seed}"
                if s_i == 0:
                    rng = make_rng("training", seed)
                    idx = rng.choice(len(xs_tr), size=min(32, len(xs_tr)), replace=False)
                    torch.manual_seed(seed)
                    m = NeuralKF(mode, config.HYB_KF_SIGMA_A, config.HYB_HIDDEN)
                    first, last = overfit_one_batch(m, xs_tr[idx], ys_tr[idx], 100 if not args.smoke else 20,
                                                    kf_lr[code], device, loss_fn=loss_kf,
                                                    clip=kf_params.get("clip", 1.0))
                    summary.check(f"{NAMES[code]}: loss на одном батче падает", last < first,
                                  f"loss {first:.4f} -> {last:.4f}")
                rng = make_rng("training", seed)
                torch.manual_seed(seed)
                model = NeuralKF(mode, config.HYB_KF_SIGMA_A, config.HYB_HIDDEN)
                log.info("Обучение %s", key)
                t1 = time.time()
                history, _ = train(model, xs_tr, ys_tr, xs_val, ys_val, {**kf_params, "lr": kf_lr[code]}, device, rng,
                                   loss_fn=loss_kf)
                costs.setdefault(NAMES[code], []).append((sum(p.numel() for p in model.parameters()),
                                                          time.time() - t1))
                summary.check(f"{key}: модель обучилась", history["best_val_loss"] < history["val_loss_initial"],
                              f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
                train_plots.plot_history(history, OUT_DIR / f"loss_{code}_{seed}{tag}.png", f"Обучение: {key}")
                ART.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "mode": mode, "seed": seed},
                           ART / f"{code}_{seed}{tag}.pt")
                fcs = {name: [mask_forecast(kf_forecast(model, rt, f, device), k)
                              for rt, f, k in zip(built[name][1], forecasts[name]["IMM"], ks[name])]
                       for name in EVAL_SETS}
                evaluate(NAMES[code], fcs, s_i)

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    order = ["KF", KF2D, "IMM", NAMES["A"], NAMES["B1"], NAMES["B2"], NAMES["B3"]]
    for name in EVAL_SETS:
        lines = [f"{'метод':32s}" + "".join(f" {c:>11s}" for c in ("ADE", "FDE", "прямая", "вход", "маневр", "выход"))]
        for method in order:
            reps = reports[name][method]
            lines.append(f"{method:32s}" + "".join(f" {cell(reps, k):>11s}"
                                                   for k in ("ade", "fde", "straight", "entry", "turn", "exit")))
        summary.section(f"{name}: ADE (м), обучаемые схемы - среднее ± СКО по {len(seeds)} seed", lines)
        compare(built[name][1], {m: forecasts[name][m] for m in order}, summary, f"{name} (seed {seeds[0]})",
                OUT_DIR, f"{name}{tag}")
    summary.section(f"Б2, Б3: выбор шага обучения по val (seed {seeds[0]})", lr_lines)
    summary.section("Затраты: параметров, обучение (с, среднее по seed)", [
        f"{m:32s} {int(np.mean([c[0] for c in v])):8d} {np.mean([c[1] for c in v]):10.1f}" for m, v in costs.items()])
    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
