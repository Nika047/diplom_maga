"""
Этап 5 (версия 2): схемы встраивания нейросети в фильтр. Схема A (поправка после IMM) - IMM + GRU этапа 2
(лучшая конфигурация), остальные обучаются здесь с гиперпараметрами сети A; 3 seed, оценка на одних обзорах.
  Б1  - сеть выбирает модель: GRU пересчитывает веса CV и CT в прогнозе IMM (до обучения = IMM);
  Б2  - сеть внутри фильтра Калмана (по типу KalmanNet): поправка к коэффициенту усиления;
  Б3  - тот же фильтр, сеть задаёт СКО шума ускорения; в случаях с доплером фильтр дообновляется по
        радиальной скорости (models/hybrids.NeuralKF, проекция скорости на луч визирования);
  Б4  - сеть оценивает «идёт маневр», её выход смещает вероятности моделей рабочего IMM (alpha по val);
  Б4 + A - корректор поверх IMM схемы Б4; A + Б1 - одна сеть: веса моделей и поправка;
  Б5  - прогноз сети A - ещё один режим IMM.
Итог этапа - основа случая (pipeline/selection.choose_base): лучшая по ADE на val среди гибридов этапов 2,
4, 5; если есть не значимо худшие (95% интервал парной разности по полётам накрывает 0), берётся самая
быстрая по времени отклика. Состояние base_<случай> читают этапы 6-8.

  .venv\\Scripts\\python.exe run_stage5.py [--case ...] [--smoke]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from common import cases  # noqa: E402
from common.cases import config_override  # noqa: E402
from common.rng import make_rng  # noqa: E402
from data.phases import PHASE_CODE  # noqa: E402
from data.windows import apply_correction, model_offsets, sequence_arrays, sequence_input  # noqa: E402
from data.windows import sequence_chunks  # noqa: E402
from eval import metrics  # noqa: E402
from eval.compare import mask_forecast  # noqa: E402
from eval.latency import net_ms  # noqa: E402
from filters.imm import IMM  # noqa: E402
from filters.neural_mode import NeuralMode  # noqa: E402
from models.hybrids import (ModeGRU, ModelWeightGRU, NeuralKF, WeightResidualGRU, bce_loss, model_weight_loss,  # noqa: E402
                            sequence_loss, weight_residual_loss)
from pipeline import data as pdata  # noqa: E402
from pipeline import evaluation, latency, selection  # noqa: E402
from pipeline.custom import fit_model, ms_per_scan_sequence  # noqa: E402
from pipeline.runs import fit_and_eval  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from pipeline.tables import method_table  # noqa: E402
from radar.model import observe  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment  # noqa: E402
from training.trainer import predict, train  # noqa: E402
from tuning import corrector_search  # noqa: E402
from viz.tuning_plots import plot_bars_ci  # noqa: E402

STAGE = "stage5"
NAMES = {"B1": "Б1. IMM, сеть выбирает модель", "B2": "Б2. KF, сеть в усилении", "B3": "Б3. KF, сеть задаёт шум",
         "B4": "Б4. сеть управляет переключением IMM", "B4A": "Б4 + A", "AB1": "A + Б1. веса моделей и поправка",
         "B5": "Б5. прогноз сети - режим IMM"}
A_METHOD = "stage2/IMM+GRU"
TURN = (PHASE_CODE["entry"], PHASE_CODE["turn"])
KF_TRAIN = {"batch": 128, "weight_decay": 1e-5, "skip_nonfinite": True}


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
    seq = sequence_input(s, slice(None), s["z"][0])
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(seq)[None].to(device))[0].cpu().numpy().astype(float) + s["z"][0]
    return np.concatenate([out, imm_forecast[..., 2:3]], axis=-1)


def maneuver_labels(ws, rts: list) -> np.ndarray:
    return np.isin(np.array([rts[t].phase[k] for t, k in zip(ws.track, ws.k)]), TURN).astype(np.float32)


def mode_logits(ws, n_scans: list, logit: np.ndarray, alpha: float) -> list:
    """Добавки mode_logit (K, M) по трекам: alpha * [log(1 - p), log p] в обзорах окон, nan - вне их."""
    i_cv, i_ct = config.IMM_MODELS.index("CV"), config.IMM_MODELS.index("CT")
    logp = -np.logaddexp(0.0, -logit.astype(float))
    log1p = -np.logaddexp(0.0, logit.astype(float))
    out = []
    for i, k_total in enumerate(n_scans):
        ml = np.full((k_total, len(config.IMM_MODELS)), np.nan)
        sel = ws.track == i
        ml[ws.k[sel], i_cv] = alpha * log1p[sel]
        ml[ws.k[sel], i_ct] = alpha * logp[sel]
        out.append(ml)
    return out


def run_case(ctx: Ctx) -> bool:
    st, summary, log, case = ctx.st, ctx.summary, ctx.log, ctx.case
    state = ctx.state("corr")
    if state is None or "IMM+GRU" not in state:
        summary.check("Результаты этапа 2 найдены", False, "нужен IMM+GRU в состоянии corr")
        return False
    window, net, hp, train_params = corrector_search.split(state["IMM+GRU"]["best"])
    params = {**train_params, "epochs": st.epochs, "patience": st.patience}
    cd = ctx.casedata(window=window, net_input=net)
    sets, seeds = ctx.sets, st.seeds
    tracks_tr = pdata.train_tracks(sets)
    ws_tr = cd.windows("main", tracks_tr)
    names = config.EVAL_SETS
    reg = ctx.registry
    imm_ms = latency.imm_ms_per_scan(cd, "val")
    n_feat = ws_tr.x.shape[-1]

    # тень IMM с прогнозами моделей: train и наборы оценки
    with config_override(cd.override()):
        imm_m = IMM()
        log.info("Прогон IMM с прогнозами моделей: train %d трасс", len(tracks_tr))
        rts_tr = [observe(tr) for tr in tracks_tr]
        runs_tr = [imm_m.run(rt, with_models=True) for rt in rts_tr]
        runs_ev = {n: [imm_m.run(rt, with_models=True) for rt in cd.eval(n)[1]] for n in names}
    p_pow = np.stack(imm_m.p_pow)
    n_models = p_pow.shape[1]
    n_scans = {n: [len(rt.t) for rt in cd.eval(n)[1]] for n in names}
    norm = dataset.Normalizer.fit(ws_tr)
    x_tr = norm.x(ws_tr.x)
    x_ev = {n: norm.x(cd.eval(n)[0].x) for n in names}
    ws_val = cd.eval("val")[0]

    def record(label: str, seed: int, fcs: dict, costs: dict, extra: dict = None) -> None:
        reg.record(f"{STAGE}/{label}", seed, evaluation.reports(cd, fcs), {**costs, **(extra or {})}, fcs=fcs)

    def net_cost(model) -> dict:
        return {"n_params": latency.n_params(model), **latency.response_times(imm_ms, model, n_feat, window)}

    def run_imm(name: str, **kw_list) -> list:
        out = []
        with config_override(cd.override()):
            for i, rt in enumerate(cd.eval(name)[1]):
                kw = {key: val[i] for key, val in kw_list.items()}
                extra = kw.pop("extra_models", None)
                out.append(IMM(extra_models=extra).run(rt, **kw).forecast)
        return cd.mask(name, out)

    def corrected(name: str, corr: np.ndarray, runs=None, ws=None) -> list:
        ws = ws or cd.eval(name)[0]
        runs = runs or cd.eval(name)[2]
        return [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                               corr[ws.track == i]), cd.ks(name)[i]) for i, run in enumerate(runs)]

    # --- A: сети этапа 2 из хранилища (без обучения), прогнозы нужны для Б5 ---
    outs_a = fit_and_eval(ctx, cd, "stage2", "IMM+GRU", "GRU", "hybrid", ws_tr, train_params, hp, window,
                          overfit=False, record=False)
    if not reg.has(A_METHOD):
        summary.check("A (IMM+GRU этапа 2) есть в реестре", False, "запустите run_stage2.py")
        return False

    # --- Б1 ---
    d_tr, lw_tr = offsets_for(ws_tr, runs_tr, p_pow)
    ev = {n: offsets_for(cd.eval(n)[0], runs_ev[n], p_pow) for n in names}

    def b1_target(ws, d, lw):
        return np.concatenate([norm.y(ws.y).reshape(len(ws), -1), norm.y(d).reshape(len(ws), -1),
                               lw.reshape(len(ws), -1)], axis=1).astype(np.float32)

    def b1_correction(logits, d, lw):
        z = lw + logits.reshape(lw.shape)
        w = np.exp(z - z.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        return (w[..., None] * d).sum(axis=2)

    y1_tr, y1_val = b1_target(ws_tr, d_tr, lw_tr), b1_target(ws_val, *ev["val"])
    loss_b1 = model_weight_loss(config.HORIZON_SCANS, n_models)
    for s_i, seed in enumerate(seeds):
        model, _, _ = fit_model(ctx, STAGE, NAMES["B1"], seed, lambda: ModelWeightGRU(
            n_feat, config.HORIZON_SCANS, n_models, **hp), x_tr, y1_tr, x_ev["val"], y1_val, params, loss_fn=loss_b1,
            overfit={"ok": lambda f, l: l < f} if s_i == 0 else None)
        if s_i == 0:
            zero = np.abs(b1_correction(predict(ModelWeightGRU(n_feat, config.HORIZON_SCANS, n_models, **hp),
                                                x_ev["val"], ctx.device), *ev["val"])).max()
            summary.check("Б1 до обучения = IMM", zero < 1e-3, f"max |поправка| {zero:.2e} м")
        fcs = {n: corrected(n, b1_correction(predict(model, x_ev[n], ctx.device), *ev[n])) for n in names}
        record(NAMES["B1"], seed, fcs, net_cost(model))

    # --- Б2, Б3: фильтр с сетью внутри ---
    seqs_tr = [sequence_arrays(rt) for rt in rts_tr]
    xs_tr, ys_tr = sequence_chunks(seqs_tr, st.chunk)
    xs_val, ys_val = sequence_chunks([sequence_arrays(rt) for rt in cd.eval("val")[1]], st.chunk)
    loss_kf = sequence_loss(config.WARMUP_SCANS, config.HYB_SCALE_M)
    dop = case != "nodop"
    sigma_vr = ctx.imm_state()["sigma_vr"] or config.RADAR_SIGMA_VR
    kf_params = {**KF_TRAIN, "epochs": st.kf_epochs, "patience": st.kf_patience}
    build_kf = lambda mode: (lambda: NeuralKF(mode, config.HYB_KF_SIGMA_A, config.HYB_HIDDEN, doppler=dop,  # noqa: E731
                                              sigma_vr=sigma_vr))
    plain = build_kf("noise")().to(ctx.device)
    plain_fc = {n: [mask_forecast(kf_forecast(plain, rt, f, ctx.device), k)
                    for rt, f, k in zip(cd.eval(n)[1], evaluation.imm_forecasts(cd, n), cd.ks(n))] for n in names}
    rep_plain = evaluation.report(cd, "val", plain_fc["val"])
    kf_ref = reg.report("stage1/KF", "val")["ade"]
    summary.check("Фильтр на PyTorch без сети близок к фильтру Калмана этапа 1" + (" (с доплером)" if dop else ""),
                  abs(rep_plain["ade"] / kf_ref - 1) <= (0.10 if dop else 0.05),
                  f"ADE val {rep_plain['ade']:.2f} м против {kf_ref:.2f} м (2D против 3D, псевдолинейный доплер)")
    kf_lr = ctx.state("kf_lr", {})
    lr_lines = []
    for code, mode in (("B2", "gain"), ("B3", "noise")):
        if code not in kf_lr:           # выбор шага обучения по val на первом seed; результат сохраняется
            val_loss, skips = {}, {}
            for lr in st.kf_lrs:
                rng = make_rng("training", seeds[0])
                torch.manual_seed(seeds[0])
                hist, _ = train(build_kf(mode)(), xs_tr, ys_tr, xs_val, ys_val, {**kf_params, "lr": lr}, ctx.device, rng,
                                quiet=True, loss_fn=loss_kf)
                val_loss[lr], skips[lr] = hist["best_val_loss"], sum(e["skipped"] for e in hist["epochs"])
            stable = [lr for lr in st.kf_lrs if skips[lr] == 0 and np.isfinite(val_loss[lr])]
            kf_lr[code] = min(stable or st.kf_lrs, key=val_loss.get)
            lr_lines.append(f"{NAMES[code]}: " + ", ".join(f"lr {lr:g} -> {val_loss[lr]:.4f}"
                                                             + (f" (пропусков {skips[lr]})" if skips[lr] else "")
                                                             for lr in st.kf_lrs) + f"; выбран {kf_lr[code]:g}")
            ctx.save_state("kf_lr", kf_lr)
    diverged = {}
    kf_ms = None
    for code, mode in (("B2", "gain"), ("B3", "noise")):
        for s_i, seed in enumerate(seeds):
            model, hist, _ = fit_model(ctx, STAGE, NAMES[code], seed, build_kf(mode), xs_tr, ys_tr, xs_val, ys_val,
                                       {**kf_params, "lr": kf_lr[code]}, loss_fn=loss_kf,
                                       overfit={"batch": 32, "steps": 100 if not ctx.smoke else 20, "lr": kf_lr[code],
                                                "clip": 1.0, "ok": lambda f, l: l < f} if s_i == 0 else None)
            fcs = {n: [mask_forecast(kf_forecast(model, rt, f, ctx.device), k)
                       for rt, f, k in zip(cd.eval(n)[1], evaluation.imm_forecasts(cd, n), cd.ks(n))] for n in names}
            n_bad = sum(int(np.isnan(f[k]).any(axis=(1, 2)).sum()) for n in names for f, k in zip(fcs[n], cd.ks(n)))
            trained = hist["best_val_loss"] < hist["val_loss_initial"]
            summary.check(f"{NAMES[code]}, seed {seed}: прогноз конечен на всех обзорах", n_bad == 0,
                          f"обзоров с nan: {n_bad}" + ("" if n_bad == 0 else " (seed исключён из средних)"))
            if not trained or n_bad:
                diverged.setdefault(NAMES[code], []).append(seed)
                continue
            if kf_ms is None:
                kf_ms = ms_per_scan_sequence(model, xs_tr.shape[-1])
            record(NAMES[code], seed, fcs, {"n_params": latency.n_params(model), "imm_ms": 0.0,
                                            "total_cpu_ms": kf_ms, "kf_ms": kf_ms})

    # --- A + Б1 ---
    n_h = config.HORIZON_SCANS * n_models
    loss_ab1 = weight_residual_loss(config.HORIZON_SCANS, n_models)

    def ab1_correction(out, d, lw):
        z = lw + out[:, :n_h].reshape(lw.shape)
        w = np.exp(z - z.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        return (w[..., None] * d).sum(axis=2) + norm.y_inv(out[:, n_h:].reshape(-1, config.HORIZON_SCANS, 2))

    for s_i, seed in enumerate(seeds):
        model, _, _ = fit_model(ctx, STAGE, NAMES["AB1"], seed, lambda: WeightResidualGRU(
            n_feat, config.HORIZON_SCANS, n_models, **hp), x_tr, y1_tr, x_ev["val"], y1_val, params, loss_fn=loss_ab1)
        if s_i == 0:
            zero = np.abs(ab1_correction(predict(WeightResidualGRU(n_feat, config.HORIZON_SCANS, n_models, **hp),
                                                 x_ev["val"], ctx.device), *ev["val"])).max()
            summary.check("A + Б1 до обучения = IMM", zero < 1e-3, f"max |поправка| {zero:.2e} м")
        fcs = {n: corrected(n, ab1_correction(predict(model, x_ev[n], ctx.device), *ev[n])) for n in names}
        record(NAMES["AB1"], seed, fcs, net_cost(model))

    # --- Б4, Б4 + A ---
    lab_tr = maneuver_labels(ws_tr, rts_tr)[:, None]
    lab_ev = {n: maneuver_labels(cd.eval(n)[0], cd.eval(n)[1])[:, None] for n in names}
    ap_lines, alpha_lines = [], []
    with config_override(cd.override()):
        rt0 = cd.eval("val")[1][0]
        a_ = IMM().run(rt0).forecast
        b_ = IMM().run(rt0, mode_logit=np.zeros((len(rt0.t), n_models))).forecast
        c_ = IMM(extra_models=[NeuralMode(np.full((len(rt0.t), config.HORIZON_SCANS, 3), np.nan), np.eye(2))]).run(
            rt0).forecast
        d_ = IMM(model_names=tuple(config.IMM_MODELS) + ("CV",)).run(rt0).forecast
    summary.check("Б4: при нулевой добавке рабочий IMM = IMM", np.array_equal(np.nan_to_num(a_), np.nan_to_num(b_)),
                  "прогнозы совпали до бита")
    summary.check("Б5: режим сети без прогноза совпадает с моделью CV", float(np.nanmax(np.abs(c_ - d_))) < 1e-6,
                  f"IMM (CV, CT, сеть без прогноза) против IMM (CV, CT, CV): {float(np.nanmax(np.abs(c_ - d_))):.2e} м")
    for s_i, seed in enumerate(seeds):
        clf, _, _ = fit_model(ctx, STAGE, NAMES["B4"], seed, lambda: ModeGRU(n_feat, **hp), x_tr, lab_tr, x_ev["val"],
                              lab_ev["val"], params, loss_fn=bce_loss,
                              overfit={"ok": lambda f, l: l < config.OVERFIT_TARGET * f} if s_i == 0 else None)
        logit = {n: predict(clf, x_ev[n], ctx.device)[:, 0] for n in names}
        for n in names:
            ws, _, runs = cd.eval(n)
            mu_ct = np.array([runs[t].mu[k, config.IMM_MODELS.index("CT")] for t, k in zip(ws.track, ws.k)])
            ap_lines.append(f"{n}, seed {seed}: AP сети {metrics.average_precision(logit[n], lab_ev[n][:, 0]):.3f}, "
                            f"AP вероятности CT в IMM {metrics.average_precision(mu_ct, lab_ev[n][:, 0]):.3f}, "
                            f"доля маневра {lab_ev[n].mean():.3f}")
        ade_alpha = {}
        for alpha in st.alphas:
            fcs_v = run_imm("val", mode_logit=mode_logits(ws_val, n_scans["val"], logit["val"], alpha))
            ade_alpha[alpha] = evaluation.summarize_method(cd.eval("val")[1], fcs_v)["ade"]
        alpha = min(ade_alpha, key=ade_alpha.get)
        alpha_lines.append(f"seed {seed}: " + ", ".join(f"alpha {a:g} -> {v:.2f} м" for a, v in ade_alpha.items())
                           + f"; выбрано {alpha:g}")
        ml = {n: mode_logits(cd.eval(n)[0], n_scans[n], logit[n], alpha) for n in names}
        record(NAMES["B4"], seed, {n: run_imm(n, mode_logit=ml[n]) for n in names},
               {"n_params": latency.n_params(clf), **latency.response_times(imm_ms, clf, n_feat, window)},
               {"alpha": alpha})
        # Б4 + A: окна по IMM схемы Б4, корректор поверх
        logit_tr = predict(clf, x_tr, ctx.device)[:, 0]
        ml_tr = mode_logits(ws_tr, [len(rt.t) for rt in rts_tr], logit_tr, alpha)
        with config_override(cd.override()):
            ws_tr4, _, _ = dataset.build(tracks_tr, window=window, run_kwargs=[{"mode_logit": m} for m in ml_tr])
            b4 = {n: dataset.build(sets[n], window=window, run_kwargs=[{"mode_logit": m} for m in ml[n]]) for n in names}
        res4 = run_experiment(f"{NAMES['B4A']}, seed {seed}", "GRU", "hybrid", ws_tr4, b4["val"][0], params, ctx.device,
                              summary, log, ctx.smoke, ctx.out_dir, ctx.art_dir, window=window, seed=seed,
                              model_kwargs=hp, overfit=False, extra_sets={n: b4[n][0] for n in names[1:]},
                              store=ctx.store, store_key=(STAGE, case))
        preds4 = {"val": res4["pred"], **res4["extra"]}
        fcs4 = {n: corrected(n, preds4[n], runs=b4[n][2], ws=b4[n][0]) for n in names}
        a_extra = reg.data[A_METHOD][str(seed)]["extra"] if reg.has(A_METHOD, seed) else {}
        clf_ms = net_ms(clf, n_feat, window, torch.device("cpu"))
        record(NAMES["B4A"], seed, fcs4, {"n_params": latency.n_params(clf) + a_extra.get("n_params", 0),
                                          "imm_ms": 2 * imm_ms,
                                          "total_cpu_ms": 2 * imm_ms + clf_ms + a_extra.get("net_cpu_ms", 0.0)},
               {"alpha": alpha})
        del ws_tr4, b4

    # --- Б5: прогноз сети A - третий режим IMM ---
    for s_i, seed_out in enumerate(outs_a):
        seed = seed_out["seed"]
        fc_a = seed_out["fcs"]
        err = []
        for rt, fc in zip(cd.eval("val")[1], fc_a["val"]):
            k = np.flatnonzero(~np.isnan(fc[:-1, 0, 0]))
            err.append(rt.truth[k + 1, :2] - fc[k, 0, :2])
        err = np.concatenate(err)
        pos_cov = np.cov(err.T) + np.outer(err.mean(axis=0), err.mean(axis=0))
        fc5 = {}
        with config_override(cd.override()):
            for n in names:
                fc5[n] = [mask_forecast(IMM(extra_models=[NeuralMode(fc_a[n][i], pos_cov)]).run(rt).forecast, cd.ks(n)[i])
                          for i, rt in enumerate(cd.eval(n)[1])]
        a_costs = {k: reg.data[A_METHOD][str(seed)]["extra"].get(k) for k in ("n_params", "total_cpu_ms", "total_gpu_ms")
                   if reg.has(A_METHOD, seed)}
        record(NAMES["B5"], seed, fc5, {**a_costs, "total_cpu_ms": (a_costs.get("total_cpu_ms") or imm_ms) * 1.3})

    # --- сводка, основа ---
    order = ["stage1/IMM", A_METHOD] + [f"{STAGE}/{NAMES[c]}" for c in ("B1", "AB1", "B4", "B4A", "B5", "B2", "B3")
                                       if reg.has(f"{STAGE}/{NAMES[c]}")]
    for n in names:
        summary.section(f"{n}: ADE (м), схемы - среднее ± СКО по {len(seeds)} seed (без разошедшихся)",
                        method_table(reg, order, n, ref="stage1/IMM"))
    summary.section("Б2, Б3: выбор шага обучения по val", lr_lines or ["взят из сохранённого состояния kf_lr"])
    summary.section("Разошедшиеся seed (в средние не входят)", [f"{m}: seed {', '.join(map(str, s))}"
                                                                for m, s in diverged.items()] or ["нет"])
    summary.section("Б4: распознавание маневра (AP по обзорам окон)", ap_lines)
    summary.section("Б4: выбор alpha по ADE на val", alpha_lines)

    candidates = [m for m in reg.methods() if m.split("/", 1)[0] in ("stage2", "stage4", "stage5")]
    pick = selection.choose_base(reg, candidates, "val", st.boot_n)
    rows = sorted(pick["rows"], key=lambda r: r["ade"])
    summary.section("Выбор основы по val (разность ADE с лучшей, 95% интервал по полётам; не значимо хуже - «равна»)", [
        f"{'метод':48s} {'ADE val':>8s} {'разность':>9s} {'интервал':>16s} {'мс CPU':>7s}  равна лучшей"] + [
        f"{r['method']:48s} {r['ade']:8.2f} {r['diff']:+9.2f} [{r['lo']:+6.2f}, {r['hi']:+6.2f}] {r['ms']:7.2f}  "
        f"{'да' if r['equal'] else 'нет'}" for r in rows] + [
        f"лучшая по ADE: {pick['best']}; основа (самая быстрая из равных): {pick['base']}"])
    hybrid = [r for r in rows if r["method"].split("/", 1)[0] in ("stage2", "stage4")]
    base_corr = pick["base"] if pick["base"] in {r["method"] for r in hybrid} else hybrid[0]["method"]
    summary.check("Основа выбрана", pick["base"] in candidates, f"{pick['base']} (корректор для гипотез: {base_corr})")
    ctx.save_state("base", {"method": pick["base"], "best": pick["best"], "corrector_method": base_corr,
                            "rows": rows})
    plot_bars_ci([(r["method"].split("/", 1)[1],) + selection.ade_ci(reg, r["method"], "val", st.boot_n) for r in rows],
                 ctx.out_dir / "val_ade.png", f"ADE val (95% интервал по полётам): {cases.CASE_TITLES[case]}")
    (ctx.out_dir / "metrics.json").write_text(json.dumps(
        {m: {n: {k: v for k, v in reg.report(m, n).items() if k != "flights"} for n in names} for m in order},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 5: схемы встраивания", run_case, needs_gpu=True)


if __name__ == "__main__":
    sys.exit(main())
