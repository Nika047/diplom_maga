"""
Этап 6г: ещё три схемы гибридизации (продолжение варианта Б, этап 6в).

  Б4     - сеть управляет переключением IMM: GRU по окну признаков «тени» (обычного IMM по
           тем же отметкам) оценивает вероятность маневра p; в рабочем IMM к логарифмам
           вероятностей моделей CV и CT после каждого обновления добавляется
           alpha * [log(1 - p), log p] (filters/imm.py, mode_logit). Меняется не только
           прогноз, но и сама фильтрация. alpha - по ADE на val из HYB_B4_ALPHAS;
  Б4 + A - корректор A поверх IMM схемы Б4 (окна строятся по рабочему IMM);
  A + Б1 - одна GRU с двумя головами: веса моделей IMM в прогнозе (Б1) и поправка к нему (A);
  Б5     - прогноз корректора A - третий режим IMM рядом с CV и CT (filters/neural_mode.py,
           по идее PR-IMM): вероятность режима IMM считает по невязке, как у моделей движения.
Эталоны на тех же обзорах: IMM и A (IMM+GRU этапа 4, обучается заново). Гиперпараметры всех
сетей - IMM+GRU из подбора этапа 4, 3 seed, оценка на val, test, holdout.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage6d.py
Смоук (рабочий ПК, нужен смоук этапа 4): .venv\\Scripts\\python.exe run_stage6d.py --smoke

Проверки: при alpha = 0 Б4 = IMM; режим сети без прогноза совпадает с CV; A + Б1 до обучения =
IMM; A повторяет этап 4; проверки обучения.
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
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast  # noqa: E402
from filters.imm import IMM  # noqa: E402
from filters.neural_mode import NeuralMode  # noqa: E402
from models.hybrids import ModeGRU, WeightResidualGRU, bce_loss, weight_residual_loss  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage6c import offsets_for  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from tuning import search  # noqa: E402
from viz import train_plots  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43), "alphas": (0.5, 2.0)}

STAGE = "stage6d"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.PROJECT_DIR / "artifacts" / STAGE
EVAL_SETS = ("val", "test", "holdout")
HYB = "IMM+GRU"
NAMES = {"A": "A. IMM+GRU, поправка после", "B4": "Б4. сеть управляет переключением IMM",
         "B4A": "Б4 + A", "AB1": "A + Б1. веса моделей и поправка", "B5": "Б5. прогноз сети - режим IMM"}
TURN = (PHASE_CODE["entry"], PHASE_CODE["turn"])


def maneuver_labels(ws, rts: list) -> np.ndarray:
    """1, если в обзоре окна идёт маневр (участок entry или turn по истине) - цель сети Б4."""
    return np.isin(np.array([rts[t].phase[k] for t, k in zip(ws.track, ws.k)]), TURN).astype(np.float32)


def mode_logits(ws, n_scans: list, logit: np.ndarray, alpha: float) -> list:
    """Добавки mode_logit (K, M) по трекам: alpha * [log(1 - p), log p] в обзорах окон, nan - вне их."""
    i_cv, i_ct = config.IMM_MODELS.index("CV"), config.IMM_MODELS.index("CT")
    logp = -np.logaddexp(0.0, -logit.astype(float))         # log p
    log1p = -np.logaddexp(0.0, logit.astype(float))         # log(1 - p)
    out = []
    for i, k_total in enumerate(n_scans):
        ml = np.full((k_total, len(config.IMM_MODELS)), np.nan)
        sel = ws.track == i
        ml[ws.k[sel], i_cv] = alpha * log1p[sel]
        ml[ws.k[sel], i_ct] = alpha * logp[sel]
        out.append(ml)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 6г: схемы Б4, A + Б1, Б5")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 6г: схемы Б4, A + Б1, Б5" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 6г - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    alphas = SMOKE["alphas"] if args.smoke else config.HYB_B4_ALPHAS
    t0 = time.time()

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
    tracks_tr = sets["train"] + sets["synth"]
    with search.config_override(imm):
        ws_tr = dataset.load_or_build(f"train{tag}_w{window}", tracks_tr, cache4, window=window)
        built = {name: dataset.build(sets[name], window=window) for name in EVAL_SETS}
        imm0 = IMM()
        log.info("Тень IMM с прогнозами моделей по train: %d трасс", len(tracks_tr))
        rts_tr = [observe(tr) for tr in tracks_tr]
        runs_tr = [imm0.run(rt, with_models=True) for rt in rts_tr]
        runs_ev = {name: [imm0.run(rt, with_models=True) for rt in built[name][1]] for name in EVAL_SETS}
    p_pow = np.stack(imm0.p_pow)
    n_models = p_pow.shape[1]
    ks = {name: [np.array(sorted(b[0].k[b[0].track == i]), dtype=int) for i in range(len(b[1]))]
          for name, b in built.items()}
    n_scans = {name: [len(rt.t) for rt in built[name][1]] for name in EVAL_SETS}

    def run_imm(name, **kw_list):
        """Прогнозы рабочего IMM по набору name; kw_list - аргументы для каждой трассы."""
        out = []
        with search.config_override(imm):
            for i, rt in enumerate(built[name][1]):
                kw = {key: val[i] for key, val in kw_list.items()}
                extra = kw.pop("extra_models", None)
                out.append(IMM(extra_models=extra).run(rt, **kw).forecast)
        return out

    forecasts = {name: {"IMM": [mask_forecast(r.forecast, k) for r, k in zip(built[name][2], ks[name])]}
                 for name in EVAL_SETS}
    reports = {name: {"IMM": [summarize_method(built[name][1], forecasts[name]["IMM"])]} for name in EVAL_SETS}

    def evaluate(method, fcs_by_set, s_i):
        for name in EVAL_SETS:
            fcs = [mask_forecast(fc, k) for fc, k in zip(fcs_by_set[name], ks[name])]
            reports[name].setdefault(method, []).append(summarize_method(built[name][1], fcs))
            if s_i == 0:
                forecasts[name][method] = fcs

    def corrected(name, corr, runs=None):
        ws = built[name][0]
        runs = runs or built[name][2]
        return [apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i], corr[ws.track == i])
                for i, run in enumerate(runs)]

    norm = dataset.Normalizer.fit(ws_tr)
    x_tr = norm.x(ws_tr.x)
    x_ev = {name: norm.x(built[name][0].x) for name in EVAL_SETS}
    lab_tr = maneuver_labels(ws_tr, rts_tr)[:, None]
    lab_ev = {name: maneuver_labels(built[name][0], built[name][1])[:, None] for name in EVAL_SETS}

    # Проверки встраивания до обучения
    rt0 = built["val"][1][0]
    with search.config_override(imm):
        a = IMM().run(rt0).forecast
        b = IMM().run(rt0, mode_logit=np.zeros((len(rt0.t), n_models))).forecast
        c = IMM(extra_models=[NeuralMode(np.full((len(rt0.t), config.HORIZON_SCANS, 3), np.nan), np.eye(2))]).run(
            rt0).forecast
        d = IMM(model_names=tuple(config.IMM_MODELS) + ("CV",)).run(rt0).forecast
    summary.check("Б4: при нулевой добавке рабочий IMM = IMM", np.array_equal(np.nan_to_num(a), np.nan_to_num(b)),
                  "прогнозы совпали до бита" if np.array_equal(np.nan_to_num(a), np.nan_to_num(b)) else "расхождение")
    diff = float(np.nanmax(np.abs(c - d)))
    summary.check("Б5: режим сети без прогноза совпадает с моделью CV", diff < 1e-6,
                  f"IMM (CV, CT, сеть без прогноза) против IMM (CV, CT, CV): {diff:.2e} м")

    pos_cov_info, b4_info, ap_lines = [], [], []
    with search.config_override(imm):
        for s_i, seed in enumerate(seeds):
            # A - эталон и источник прогноза для Б5
            res = run_experiment(f"{NAMES['A']}, seed {seed}", "GRU", "hybrid", ws_tr, built["val"][0], params,
                                 device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                 model_kwargs=hp, overfit=False, extra_sets={n: built[n][0] for n in ("test", "holdout")})
            preds_a = {"val": res["pred"], **res["extra"]}
            fc_a = {name: [mask_forecast(f, k) for f, k in zip(corrected(name, preds_a[name]), ks[name])]
                    for name in EVAL_SETS}      # nan вне окон: там режим сети Б5 ведёт себя как CV
            evaluate(NAMES["A"], fc_a, s_i)
            if s_i == 0:
                ga, got = best_all[HYB]["ade_val"], reports["val"][NAMES["A"]][0]["ade"]
                summary.check("A повторяет этап 4", abs(got / ga - 1) <= 0.05,
                              f"ADE val на этапе 4 {ga:.2f} м, здесь (seed {seed}) {got:.2f} м")

            # Б4: классификатор маневра и выбор alpha по val
            key = f"{NAMES['B4']}, seed {seed}"
            rng = make_rng("training", seed)
            if s_i == 0:
                idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
                torch.manual_seed(seed)
                first, last = overfit_one_batch(ModeGRU(x_tr.shape[-1], **hp), x_tr[idx], lab_tr[idx],
                                                config.OVERFIT_STEPS, 3e-3, device, loss_fn=bce_loss)
                summary.check("Б4: overfit one batch", last < config.OVERFIT_TARGET * first, f"loss {first:.4f} -> {last:.4f}")
                rng = make_rng("training", seed)
            torch.manual_seed(seed)
            clf = ModeGRU(x_tr.shape[-1], **hp)
            hist, _ = train(clf, x_tr, lab_tr, x_ev["val"], lab_ev["val"], params, device, rng, loss_fn=bce_loss)
            summary.check(f"{key}: классификатор обучился", hist["best_val_loss"] < hist["val_loss_initial"],
                          f"val BCE {hist['val_loss_initial']:.4f} -> {hist['best_val_loss']:.4f}")
            train_plots.plot_history(hist, OUT_DIR / f"loss_B4_{seed}{tag}.png", f"Обучение: {key}")
            logit = {name: predict(clf, x_ev[name], device)[:, 0] for name in EVAL_SETS}
            for name in EVAL_SETS:
                ws = built[name][0]
                mu_ct = np.array([built[name][2][t].mu[k, config.IMM_MODELS.index("CT")] for t, k in zip(ws.track, ws.k)])
                ap_lines.append(f"{name}, seed {seed}: AP сети {metrics.average_precision(logit[name], lab_ev[name][:, 0]):.3f}, "
                                f"AP вероятности CT в IMM {metrics.average_precision(mu_ct, lab_ev[name][:, 0]):.3f}, "
                                f"доля маневра {lab_ev[name].mean():.3f}")
            ade_alpha = {}
            for alpha in alphas:
                fcs = run_imm("val", mode_logit=mode_logits(built["val"][0], n_scans["val"], logit["val"], alpha))
                ade_alpha[alpha] = summarize_method(built["val"][1], [mask_forecast(f, k) for f, k in
                                                                        zip(fcs, ks["val"])])["ade"]
            alpha = min(ade_alpha, key=ade_alpha.get)
            b4_info.append(f"seed {seed}: " + ", ".join(f"alpha {a_:g} -> {v:.2f} м" for a_, v in ade_alpha.items())
                           + f"; выбрано {alpha:g} (IMM на val {reports['val']['IMM'][0]['ade']:.2f} м)")
            ml = {name: mode_logits(built[name][0], n_scans[name], logit[name], alpha) for name in EVAL_SETS}
            evaluate(NAMES["B4"], {name: run_imm(name, mode_logit=ml[name]) for name in EVAL_SETS}, s_i)

            # Б4 + A: окна по рабочему IMM схемы Б4, корректор поверх
            logit_tr = predict(clf, x_tr, device)[:, 0]
            ml_tr = mode_logits(ws_tr, [len(rt.t) for rt in rts_tr], logit_tr, alpha)
            log.info("Окна по IMM схемы Б4: train %d трасс", len(tracks_tr))
            ws_tr4, _, _ = dataset.build(tracks_tr, window=window, run_kwargs=[{"mode_logit": m} for m in ml_tr])
            b4 = {name: dataset.build(sets[name], window=window, run_kwargs=[{"mode_logit": m} for m in ml[name]])
                  for name in EVAL_SETS}
            res4 = run_experiment(f"{NAMES['B4A']}, seed {seed}", "GRU", "hybrid", ws_tr4, b4["val"][0], params, device,
                                  summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed, model_kwargs=hp,
                                  overfit=False, extra_sets={n: b4[n][0] for n in ("test", "holdout")})
            preds4 = {"val": res4["pred"], **res4["extra"]}
            fc4 = {}
            for name in EVAL_SETS:
                ws, _, runs = b4[name]
                fc4[name] = [apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                              preds4[name][ws.track == i]) for i, run in enumerate(runs)]
            evaluate(NAMES["B4A"], fc4, s_i)
            del ws_tr4, b4

            # A + Б1
            key = f"{NAMES['AB1']}, seed {seed}"
            d_tr, lw_tr = offsets_for(ws_tr, runs_tr, p_pow)
            y_tr = np.concatenate([norm.y(ws_tr.y).reshape(len(ws_tr), -1), norm.y(d_tr).reshape(len(ws_tr), -1),
                                   lw_tr.reshape(len(ws_tr), -1)], axis=1).astype(np.float32)
            ev = {name: offsets_for(built[name][0], runs_ev[name], p_pow) for name in EVAL_SETS}
            ws_v = built["val"][0]
            y_val = np.concatenate([norm.y(ws_v.y).reshape(len(ws_v), -1), norm.y(ev["val"][0]).reshape(len(ws_v), -1),
                                    ev["val"][1].reshape(len(ws_v), -1)], axis=1).astype(np.float32)
            loss_ab1 = weight_residual_loss(config.HORIZON_SCANS, n_models)
            torch.manual_seed(seed)
            model = WeightResidualGRU(x_tr.shape[-1], config.HORIZON_SCANS, n_models, **hp)
            hm = config.HORIZON_SCANS * n_models

            def ab1_correction(out, d, lw):
                z = lw + out[:, :hm].reshape(lw.shape)
                w = np.exp(z - z.max(axis=-1, keepdims=True))
                w /= w.sum(axis=-1, keepdims=True)
                return (w[..., None] * d).sum(axis=2) + norm.y_inv(out[:, hm:].reshape(-1, config.HORIZON_SCANS, 2))

            if s_i == 0:
                zero = np.abs(ab1_correction(predict(model, x_ev["val"], device), *ev["val"])).max()
                summary.check("A + Б1 до обучения = IMM", zero < 1e-3, f"max |поправка| {zero:.2e} м")
            rng = make_rng("training", seed)
            hist, _ = train(model, x_tr, y_tr, x_ev["val"], y_val, params, device, rng, loss_fn=loss_ab1)
            summary.check(f"{key}: модель обучилась", hist["best_val_loss"] < hist["val_loss_initial"],
                          f"val loss {hist['val_loss_initial']:.4f} -> {hist['best_val_loss']:.4f}")
            train_plots.plot_history(hist, OUT_DIR / f"loss_AB1_{seed}{tag}.png", f"Обучение: {key}")
            evaluate(NAMES["AB1"], {name: corrected(name, ab1_correction(predict(model, x_ev[name], device), *ev[name]))
                                    for name in EVAL_SETS}, s_i)

            # Б5: прогноз A - третий режим IMM. Ковариация - ошибка A на 1 обзор на val
            err = []
            for rt, fc in zip(built["val"][1], fc_a["val"]):
                k = np.flatnonzero(~np.isnan(fc[:-1, 0, 0]))
                err.append(rt.truth[k + 1, :2] - fc[k, 0, :2])
            err = np.concatenate(err)
            pos_cov = np.cov(err.T) + np.outer(err.mean(axis=0), err.mean(axis=0))
            pos_cov_info.append(f"seed {seed}: СКО ошибки A на 2 с по осям {np.sqrt(np.diag(pos_cov)).round(1)} м")
            fc5, share = {}, []
            for name in EVAL_SETS:
                fc5[name] = []
                for i, rt in enumerate(built[name][1]):
                    run5 = IMM(extra_models=[NeuralMode(fc_a[name][i], pos_cov)]).run(rt)
                    fc5[name].append(run5.forecast)
                    if name == "test":
                        share.append((run5.mu[:, -1], rt.phase))
            evaluate(NAMES["B5"], fc5, s_i)
            if s_i == 0:
                mu_nn = np.concatenate([m for m, _ in share])
                ph = np.concatenate([p for _, p in share])
                ok = ~np.isnan(mu_nn)
                pos_cov_info.append("средний вес режима сети на test: " + ", ".join(
                    f"{p} {np.mean(mu_nn[ok & (ph == PHASE_CODE[p])]):.2f}" for p in ("straight", "entry", "turn", "exit")))

    def cell(reps, key):
        vals = np.array([r[key] for r in reps], dtype=float)
        return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f}±{vals.std():.1f}"

    order = ["IMM", NAMES["A"], NAMES["B4"], NAMES["B4A"], NAMES["AB1"], NAMES["B5"]]
    for name in EVAL_SETS:
        lines = [f"{'метод':40s}" + "".join(f" {c:>11s}" for c in ("ADE", "FDE", "прямая", "вход", "маневр", "выход"))]
        for method in order:
            lines.append(f"{method:40s}" + "".join(f" {cell(reports[name][method], k):>11s}"
                                                   for k in ("ade", "fde", "straight", "entry", "turn", "exit")))
        summary.section(f"{name}: ADE (м), сети - среднее ± СКО по {len(seeds)} seed", lines)
        compare(built[name][1], {m: forecasts[name][m] for m in order}, summary, f"{name} (seed {seeds[0]})",
                OUT_DIR, f"{name}{tag}")
    summary.section("Б4: распознавание маневра (AP по обзорам окон)", ap_lines)
    summary.section("Б4: выбор alpha по ADE на val", b4_info)
    summary.section("Б5: режим сети", pos_cov_info)
    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
