"""
Этап 17: доплер в сочетаниях (решение пользователя 25.10.2026, после выбора итоговой модели).

Итоговая модель - IMM с доплером + GRU, обучение «+ все расширения» (этап 12, доплер только в
IMM; THESIS_NOTES.md, 4.24). Приёмы прежних этапов проверялись без доплера; новый компонент -
их сочетание с итоговой моделью:
  A. ансамбль (как этап 15) - среднее поправок сохранённых сетей этапа 12, без обучения:
     3 seed итоговой модели и 6 сетей (с ними - 3 seed «доплер в IMM и в сети», прогноз IMM тот же);
  B. больше синтетики (как этап 4в) - обучение с доплером на синтетике x4 (новые партии -
     поток "synth_extra", те же траектории, что на этапе 4в), без расширения и с ним;
  C. три гипотезы прямо / влево / вправо (как этап 14, семантические) - угадывается ли класс
     будущего движения на входе в маневр, когда IMM видит доплер;
  D. строб (как этап 6б) - покрытие и размер областей при том же заданном покрытии, IMM и гибрид
     без доплера и с доплером;
  E. пропуски отметок (как этап 6а) - P_d из config.PD_LEVELS: итоговая модель без переобучения и
     переобученная при каждом P_d; рядом IMM и гибрид «+ все расширения» без доплера (веса этапа 9).
Гиперпараметры - этапа 4, 3 seed. При пропуске отметки нет и радиальной скорости.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage17.py
Смоук (рабочий ПК, нужны смоуки этапов 4, 9 и 12): .venv\\Scripts\\python.exe run_stage17.py --smoke

Проверки: веса этапов 9 и 12 повторяют свои результаты на val; ансамбль не хуже средней сети;
новая синтетика не совпадает с исходной; гипотезы: лучшая не хуже каждой; ковариация не меняет
прогноз IMM, на val покрытие стробов не ниже заданного; доля пропусков соответствует P_d;
признаки окон с доплером не зависят от отметок и доплера в обзорах с пропуском; проверки
обучения из training/experiment.py.
"""

import argparse
import copy
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
from data.windows import apply_correction, build_windows  # noqa: E402
from eval import metrics  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from filters.imm import IMM  # noqa: E402
from models.multi import MultiHypGRU, semantic_loss  # noqa: E402
from radar.model import observe  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage4c import extra_synth  # noqa: E402
from run_stage6a import pd_tag, plot_levels  # noqa: E402
from run_stage6b import REGIONS, apply_region, gate_pairs, region_params  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage9 import KINDS  # noqa: E402
from run_stage11 import reference_ade  # noqa: E402
from run_stage12 import doppler_override  # noqa: E402
from run_stage14 import classes, decode, plot_fan  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment, slug  # noqa: E402
from training.trainer import overfit_one_batch, pick_device, predict, train  # noqa: E402
from tuning import search  # noqa: E402
from viz import train_plots  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43), "levels": (1.0, 0.8), "extra_step": 25}

STAGE = "stage17"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
SYNTH_X = 4                  # синтетика x4: исходная партия и SYNTH_X - 1 новых
DOP = "доплер в IMM"                                  # итоговая модель - доплер только в IMM
REF = "IMM с доплером + GRU, + все расширения"        # итоговая модель (веса этапа 12)
NODOP = f"{HYB}, + все расширения, без доплера"        # та же сеть без доплера (веса этапа 9)
# сохранённые сети этапа 12: метка здесь -> (имя на этапе 12, вариант доплера)
SAVED12 = {REF: (f"{HYB}, + все расширения, доплер в IMM", DOP),
           "доплер в IMM и в сети, + все расширения": (f"{HYB}, + все расширения, доплер в IMM и в сети",
                                                       "доплер в IMM и в сети")}
X4 = {"x4": "IMM с доплером + GRU, синтетика x4",
      "x4_aug": "IMM с доплером + GRU, + все расширения + синтетика x4"}
HYP = "IMM с доплером + GRU, + все расширения, гипотезы прямо / влево / вправо"
HYP_OUT = {"top": "наиболее вероятная", "mix": "смесь", "best": "лучшая из 3 задним числом"}
GATES = {"IMM, без доплера": ("imm", False), "IMM с доплером": ("imm", True),
         "гибрид без доплера": ("hyb", False), "гибрид с доплером": ("hyb", True)}
GATE_REGIONS = (REGIONS[0], REGIONS[1])              # для IMM; для гибрида - REGIONS[2], REGIONS[3]
GATE_COLORS = {"IMM, без доплера": "#d32f2f", "IMM с доплером": "#f48fb1", "гибрид без доплера": "#9ccc65",
               "гибрид с доплером": "#2e7d32"}


def ref12_val(name12: str, s_i: int, tag: str):
    """ADE val сети этапа 12 (seed с номером s_i)."""
    path = config.RESULTS_DIR / "stage12" / f"metrics{tag}.json"
    if not path.exists():
        return None
    reps = json.loads(path.read_text(encoding="utf-8"))["val"].get(name12, [])
    return reps[s_i]["ade"] if s_i < len(reps) else None


def check_no_leak_doppler(tracks: list, window: int, summary: StageSummary) -> None:
    """Отметки и доплер в обзорах с пропуском заменяются мусором - признаки окон не должны измениться."""
    worst, n_missed = 0.0, 0
    imm = IMM()
    for tr in tracks:
        rt = observe(tr)
        missed = ~rt.detected
        if not missed.any():
            continue
        n_missed += int(missed.sum())
        w = build_windows(rt, imm.run(rt), window=window)
        bad = copy.deepcopy(rt)
        bad.meas[missed] += 1e4
        bad.polar[missed] += 1e3
        bad.vr[missed] += 1e3
        w2 = build_windows(bad, imm.run(bad), window=window)
        if len(w["x"]) != len(w2["x"]):
            worst = float("inf")
            break
        worst = max(worst, float(np.abs(w["x"] - w2["x"]).max(initial=0.0)))
    summary.check("Доплер: признаки не зависят от отметок и доплера в обзорах с пропуском",
                  n_missed > 0 and worst == 0.0, f"пропусков в проверке {n_missed}, наибольшее изменение {worst:.3g}")


def plot_gate_radius(reports: dict, path: Path, title: str) -> None:
    """Медианный эквивалентный радиус строба по горизонтам (конформные области)."""
    hs = np.arange(1, config.HORIZON_SCANS + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, reps in reports.items():
        vals = np.array([[r[f"radius_h{h}"] for h in hs] for r in reps])
        ax.errorbar(hs * config.SCAN_PERIOD, vals.mean(axis=0), yerr=vals.std(axis=0), fmt="-o", ms=4, capsize=3,
                    color=GATE_COLORS[name.split(":")[0]], ls=":" if "круг" in name else "-", label=name)
    ax.set_xlabel("горизонт прогноза, с")
    ax.set_ylabel("медианный эквивалентный радиус, м")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 17: доплер в сочетаниях")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 17: доплер в сочетаниях" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 17 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache12 = config.CACHE_DIR / f"stage12{tag}"
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
    levels = SMOKE["levels"] if args.smoke else config.PD_LEVELS
    coverage = config.GATE_COVERAGE
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
    dop = {**imm, **doppler_override(DOP)}

    paths = {label: [config.ARTIFACTS_DIR / "stage12" / f"{slug(f'{name12}, seed {s}')}{tag}.pt" for s in seeds]
             for label, (name12, _) in SAVED12.items()}
    paths[NODOP] = [config.ARTIFACTS_DIR / "stage9" / f"{slug(f'+ все расширения, seed {s}')}{tag}.pt" for s in seeds]
    missing = [str(p) for ps in paths.values() for p in ps if not p.exists()]
    summary.check("Веса этапов 9 и 12 найдены", not missing, f"нет {missing}" if missing else "все")
    if missing:
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1

    sets = load_data(args.smoke)
    real, synth = sets["train"], sets["synth"]
    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    copies = [c for kind in KINDS for c in augment.augment(real, kind, limits)[0]]

    # Наборы оценки: без доплера, доплер в IMM, доплер в IMM и в сети (прогноз IMM - как у «в IMM»)
    built = {}
    for variant, override in (("без доплера", imm), (DOP, dop),
                              ("доплер в IMM и в сети", {**imm, **doppler_override("доплер в IMM и в сети")})):
        with search.config_override(override):
            for name in EVAL_SETS:
                built[(name, variant)] = dataset.build(sets[name], window=window)
    ks = {name: [np.array(sorted(built[(name, DOP)][0].k[built[(name, DOP)][0].track == i]), dtype=int)
                 for i in range(len(sets[name]))] for name in EVAL_SETS}
    rts_ev = {name: built[(name, "без доплера")][1] for name in EVAL_SETS}

    table = {name: {} for name in EVAL_SETS}
    forecasts = {name: {} for name in EVAL_SETS}
    for name in EVAL_SETS:
        for method, variant in (("IMM", "без доплера"), ("IMM с доплером", DOP)):
            fc = [mask_forecast(r.forecast, k) for r, k in zip(built[(name, variant)][2], ks[name])]
            table[name][method] = [summarize_method(rts_ev[name], fc)]
            forecasts[name][method] = fc

    def forecasts_of(name, variant, corr):
        ws, _, runs = built[(name, variant)]
        return [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                               corr[ws.track == i]), ks[name][i]) for i, run in enumerate(runs)]

    def record(method, s_i, variant, corr_by_set):
        for name in EVAL_SETS:
            fc = forecasts_of(name, variant, corr_by_set[name])
            table[name].setdefault(method, []).append(summarize_method(rts_ev[name], fc))
            if s_i == 0:
                forecasts[name][method] = fc

    # A. Сохранённые сети и ансамбли
    corr = {}                                       # (метка, seed) -> {набор: поправка (N, H, 2)}
    variant_of = {**{label: v for label, (_, v) in SAVED12.items()}, NODOP: "без доплера"}
    for label, ps in paths.items():
        v = variant_of[label]
        for s_i, (seed, path) in enumerate(zip(seeds, ps)):
            model, norm, _ = load_model(path, device)
            corr[(label, seed)] = {name: norm.y_inv(predict(model, norm.x(built[(name, v)][0].x), device))
                                   for name in EVAL_SETS}
            record(label, s_i, v, corr[(label, seed)])
            if label == NODOP and s_i > 0:              # у этапа 9 для проверки - только первый seed
                continue
            ref = ref12_val(SAVED12[label][0], s_i, tag) if label in SAVED12 else reference_ade(
                "stage9", "+ все расширения", tag, best_all)
            got = table["val"][label][-1]["ade"]
            summary.check(f"{label}, seed {seed}: веса повторяют свой результат на val",
                          ref is not None and abs(got / ref - 1) <= 0.005,
                          f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь {got:.2f} м")
    dd = list(SAVED12.values())[1][1]
    aligned = all(np.array_equal(built[(n, DOP)][0].k, built[(n, dd)][0].k) and
                  np.array_equal(built[(n, DOP)][0].psi, built[(n, dd)][0].psi) and
                  all(np.array_equal(a.forecast, b.forecast, equal_nan=True)
                      for a, b in zip(built[(n, DOP)][2], built[(n, dd)][2])) for n in EVAL_SETS)
    summary.check("Доплер в IMM и «в IMM и в сети»: те же обзоры, курс и прогноз IMM (поправки можно усреднять)",
                  aligned, "все наборы оценки")
    groups = {f"ансамбль {len(seeds)} seed: {REF}": [(REF, s) for s in seeds],
              f"ансамбль {2 * len(seeds)} сетей с доплером (в IMM; в IMM и в сети)":
                  [(label, s) for label in SAVED12 for s in seeds]}
    jensen = []
    for group, members in groups.items():
        corr_mean = {name: np.mean([corr[m][name] for m in members], axis=0) for name in EVAL_SETS}
        record(group, 0, DOP, corr_mean)
        for name in EVAL_SETS:
            mean_members = np.mean([table[name][label][seeds.index(s)]["ade"] for label, s in members])
            jensen.append(table[name][group][0]["ade"] <= mean_members + 1e-9)
    summary.check("Ансамбль не хуже средней сети (неравенство треугольника)", all(jensen),
                  f"ансамблей {len(groups)}, наборов {len(EVAL_SETS)}")

    # B. Синтетика x4 с доплером
    extra = extra_synth(SYNTH_X - 1, limits, SMOKE["extra_step"] if args.smoke else 1)
    new = [tr for b in extra for tr in b]
    clash = {tr.flight_id for tr in new} & {tr.flight_id for tr in synth}
    summary.check("Новая синтетика не совпадает с исходной", not clash and len(new) > 0,
                  f"новых траекторий {len(new)} в {len(extra)} партиях, совпадений имён {len(clash)}")
    with search.config_override(dop):
        ws_base = dataset.load_or_build(f"train{tag}_w{window}_d10", real + synth, cache12, window=window)
        ws_aug = dataset.load_or_build(f"aug_all{tag}_w{window}_d10", copies, cache12, window=window)
        ws_extra = [dataset.load_or_build(f"synth_extra{b + 1}_w{window}_d10", batch, cache, window=window)
                    for b, batch in enumerate(extra)]
    for key, method in X4.items():
        ws_tr = dataset.concat([ws_base] + ([ws_aug] if key == "x4_aug" else []) + ws_extra)
        log.info("%s: %d окон", method, len(ws_tr))
        for s_i, seed in enumerate(seeds):
            res = run_experiment(f"{method}, seed {seed}", "GRU", "hybrid", ws_tr, built[("val", DOP)][0], params,
                                 device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed,
                                 model_kwargs=hp, overfit=s_i == 0,
                                 extra_sets={n: built[(n, DOP)][0] for n in EVAL_SETS[1:]})
            record(method, s_i, DOP, {"val": res["pred"], **res["extra"]})
        del ws_tr
    del ws_extra

    # C. Три гипотезы с доплером (семантические)
    ws_hyp = dataset.concat([ws_base, ws_aug])
    rts_all = {tr.flight_id: observe(tr) for tr in real + synth + copies}
    rts_all.update({rt.flight_id: rt for name in EVAL_SETS for rt in rts_ev[name]})
    cls_tr = classes(ws_hyp, rts_all)
    cls_ev = {name: classes(built[(name, DOP)][0], rts_all) for name in EVAL_SETS}
    norm = dataset.Normalizer.fit(ws_hyp)

    def pack(ws, cl):
        return np.concatenate([norm.y(ws.y).reshape(len(ws), -1), cl[:, None]], axis=1).astype(np.float32)

    x_tr, y_tr = norm.x(ws_hyp.x), pack(ws_hyp, cls_tr)
    x_val, y_val = norm.x(built[("val", DOP)][0].x), pack(built[("val", DOP)][0], cls_ev["val"])
    n_hyp, loss_fn, best_ok, accuracy = config.MULTI_HYPOTHESES, semantic_loss(), True, {}
    for s_i, seed in enumerate(seeds):
        key = f"{HYP}, seed {seed}"
        if s_i == 0:
            rng = make_rng("training", seed)
            idx = rng.choice(len(x_tr), size=min(256, len(x_tr)), replace=False)
            torch.manual_seed(seed)
            m = MultiHypGRU(x_tr.shape[-1], config.HORIZON_SCANS, zero_init=True, **hp)
            first, last = overfit_one_batch(m, x_tr[idx], y_tr[idx], config.OVERFIT_STEPS, 3e-3, device,
                                            loss_fn=loss_fn)
            summary.check(f"{HYP}: overfit one batch (loss с энтропией, до нуля не обязан)", last < 0.5 * first,
                          f"loss {first:.4f} -> {last:.4f}")
        rng = make_rng("training", seed)
        torch.manual_seed(seed)
        model = MultiHypGRU(x_tr.shape[-1], config.HORIZON_SCANS, zero_init=True, **hp)
        history, _ = train(model, x_tr, y_tr, x_val, y_val, params, device, rng, loss_fn=loss_fn)
        ep = history["epochs"]
        summary.check(f"{key}: модель обучилась", history["best_val_loss"] < history["val_loss_initial"],
                      f"val loss {history['val_loss_initial']:.4f} -> {history['best_val_loss']:.4f}")
        summary.check(f"{key}: train loss падает", len(ep) >= 2 and ep[-1]["train_loss"] < ep[0]["train_loss"],
                      f"{ep[0]['train_loss']:.4f} -> {ep[-1]['train_loss']:.4f} за {len(ep)} эпох")
        file_key = f"hyp_aug_d10_seed_{seed}{tag}"
        train_plots.plot_history(history, OUT_DIR / f"loss_{file_key}.png", f"Обучение {key}")
        torch.save({"state_dict": model.state_dict(), "normalizer": norm.as_dict(), "variant": "semantic",
                    "window": window, "seed": seed, "model_kwargs": hp, "smoke": args.smoke}, ART / f"{file_key}.pt")
        outs = {o: {} for o in HYP_OUT}
        for name in EVAL_SETS:
            ws = built[(name, DOP)][0]
            hyps, probs = decode(predict(model, norm.x(ws.x), device), norm, n_hyp)
            err = np.linalg.norm(hyps - ws.y[:, None], axis=-1).mean(axis=-1)
            best = err.argmin(axis=1)
            outs["top"][name] = hyps[np.arange(len(ws)), probs.argmax(axis=1)]
            outs["mix"][name] = (probs[:, :, None, None] * hyps).sum(axis=1)
            outs["best"][name] = hyps[np.arange(len(ws)), best]
            best_ok &= bool((err[np.arange(len(ws)), best][:, None] <= err + 1e-9).all())
            hit = probs.argmax(axis=1) == cls_ev[name]
            entry = np.array([rts_ev[name][t].phase[k] == PHASE_CODE["entry"] for t, k in zip(ws.track, ws.k)],
                             dtype=bool)
            accuracy.setdefault(name, []).append((float(hit.mean()),
                                                  float(hit[entry].mean()) if entry.any() else float("nan")))
            if name == "test" and s_i == 0:
                rt_i = max(range(len(rts_ev[name])), key=lambda j: len(rts_ev[name][j].onsets))
                sel = ws.track == rt_i
                plot_fan(built[(name, DOP)][1][rt_i], built[(name, DOP)][2][rt_i],
                         SimpleNamespace(k=ws.k[sel], psi=ws.psi[sel]), hyps[sel], probs[sel],
                         OUT_DIR / f"fan_example{tag}.png")
        for o, label in HYP_OUT.items():
            record(f"{HYP}: {label}", s_i, DOP, outs[o])
    summary.check("Гипотезы: лучшая не хуже каждой (по ошибке поправки)", best_ok, "все окна, все наборы")
    del ws_hyp, x_tr, y_tr

    # D. Строб: IMM и гибрид без доплера и с доплером
    runs_cov = {}
    for variant, override in (("без доплера", imm), (DOP, dop)):
        with search.config_override(override):
            imm_cov = IMM()
            for name in EVAL_SETS:
                runs_cov[(name, variant)] = [imm_cov.run(rt, with_cov=True) for rt in built[(name, variant)][1]]
    same = max(float(np.nanmax(np.abs(a.forecast - b.forecast), initial=0.0))
               for key, runs in runs_cov.items() for a, b in zip(built[key][2], runs))
    summary.check("Строб: ковариация не меняет прогноз IMM (без доплера и с доплером)", same == 0.0,
                  f"наибольшее расхождение прогноза {same:.3g} м")
    gate_reports = {name: {} for name in EVAL_SETS}
    gate_ok = []
    for s_i, seed in enumerate(seeds):
        pairs = {}
        for gate, (kind, with_d) in GATES.items():
            variant = DOP if with_d else "без доплера"
            for name in EVAL_SETS:
                if kind == "imm":
                    fc = [mask_forecast(r.forecast, k) for r, k in zip(built[(name, variant)][2], ks[name])]
                else:
                    fc = forecasts_of(name, variant, corr[(REF if with_d else NODOP, seed)][name])
                pairs[(gate, name)] = gate_pairs(rts_ev[name], fc, runs_cov[(name, variant)])
        for with_d in (False, True):
            g_imm, g_hyb = ("IMM с доплером", "гибрид с доплером") if with_d else ("IMM, без доплера", "гибрид без доплера")
            thr = region_params(pairs[(g_imm, "val")], pairs[(g_hyb, "val")], coverage)
            for region in REGIONS:
                gate = g_imm if region.startswith("IMM,") else g_hyb
                label = f"{gate}: {region.split(', ', 1)[1]}"
                for name in EVAL_SETS:
                    inside, area = apply_region(region, thr, pairs[(gate, name)])
                    rep = metrics.gate_report(inside, area, pairs[(gate, name)]["phase"], pairs[(gate, name)]["h"])
                    gate_reports[name].setdefault(label, []).append(rep)
                    if name == "val" and s_i == 0 and region != REGIONS[0]:
                        gate_ok.append(rep["coverage"] >= coverage)
    summary.check("Строб: на калибровочном наборе (val) покрытие конформных областей не ниже заданного",
                  all(gate_ok), f"областей {len(gate_ok)}, заданное покрытие {coverage:.2f}")

    # E. Пропуски отметок
    table_pd = {name: {} for name in EVAL_SETS}
    labels_pd = {"IMM": ("без доплера", None), "IMM с доплером": (DOP, None),
                 f"{NODOP} (обучен при P_d = 1)": ("без доплера", NODOP),
                 f"{REF} (обучен при P_d = 1)": (DOP, REF)}
    for pd in levels:
        built_pd = {}
        for variant, override in (("без доплера", imm), (DOP, dop)):
            with search.config_override({**override, "RADAR_PD": pd}):
                if variant == DOP and pd == min(levels) and pd < 1.0:
                    check_no_leak_doppler(sets["test"][:4], window, summary)
                for name in EVAL_SETS:
                    built_pd[(name, variant)] = built[(name, variant)] if pd == 1.0 else dataset.build(
                        sets[name], window=window)
        ks_pd = {name: [np.array(sorted(built_pd[(name, DOP)][0].k[built_pd[(name, DOP)][0].track == i]), dtype=int)
                        for i in range(len(sets[name]))] for name in EVAL_SETS}
        miss = 1.0 - float(np.mean(np.concatenate([rt.detected for rt in built_pd[("test", DOP)][1]])))
        summary.check(f"P_d = {pd:.2f}: доля пропусков на test", abs(miss - (1.0 - pd)) <= config.PD_MISS_TOL,
                      f"{miss:.3f} при ожидаемой {1.0 - pd:.3f}")

        def fc_pd(name, variant, corr_set):
            ws, _, runs = built_pd[(name, variant)]
            if corr_set is None:
                return [mask_forecast(r.forecast, k) for r, k in zip(runs, ks_pd[name])]
            return [mask_forecast(apply_correction(run.forecast, ws.k[ws.track == i], ws.psi[ws.track == i],
                                                   corr_set[ws.track == i]), ks_pd[name][i]) for i, run in enumerate(runs)]

        for method, (variant, net) in labels_pd.items():
            for s_i in range(len(seeds) if net else 1):
                model, norm_n, _ = load_model(paths[net][s_i], device) if net else (None, None, None)
                for name in EVAL_SETS:
                    corr_set = None if net is None else norm_n.y_inv(
                        predict(model, norm_n.x(built_pd[(name, variant)][0].x), device))
                    table_pd[name].setdefault(method, {}).setdefault(pd, []).append(
                        summarize_method(built_pd[(name, variant)][1], fc_pd(name, variant, corr_set)))
        method = f"{REF} (обучен при этом P_d)"
        if pd == 1.0:
            for name in EVAL_SETS:
                table_pd[name].setdefault(method, {})[pd] = table_pd[name][f"{REF} (обучен при P_d = 1)"][pd]
            continue
        with search.config_override({**dop, "RADAR_PD": pd}):
            ws_tr = dataset.concat([dataset.load_or_build(f"train_w{window}_d10_{pd_tag(pd)}", real + synth, cache,
                                                          window=window),
                                    dataset.load_or_build(f"aug_all_w{window}_d10_{pd_tag(pd)}", copies, cache,
                                                          window=window)])
        for s_i, seed in enumerate(seeds):
            res = run_experiment(f"{REF}, P_d = {pd:.2f}, seed {seed}", "GRU", "hybrid", ws_tr,
                                 built_pd[("val", DOP)][0], params, device, summary, log, args.smoke, OUT_DIR, ART,
                                 window=window, seed=seed, model_kwargs=hp, overfit=False,
                                 extra_sets={n: built_pd[(n, DOP)][0] for n in EVAL_SETS[1:]})
            preds = {"val": res["pred"], **res["extra"]}
            for name in EVAL_SETS:
                table_pd[name].setdefault(method, {}).setdefault(pd, []).append(
                    summarize_method(built_pd[(name, DOP)][1], fc_pd(name, DOP, preds[name])))
        del ws_tr

    # Сводка
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
        summary.section(f"{name}: ADE (м), сети - среднее ± СКО по {len(seeds)} seed; «лучшая из 3» - не прогноз",
                        lines)
        source_section(summary, name, table[name])
        main_methods = ["IMM", "IMM с доплером", NODOP, REF, X4["x4_aug"]]       # остальные - в таблицах
        compare(rts_ev[name], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")
    summary.section("Гипотезы с доплером: доля верно угаданного класса (наиболее вероятная гипотеза), среднее по seed; "
                    "без доплера - этап 14 (THESIS_NOTES.md, 4.21)",
                    [f"{name:8s} все окна {np.mean([a for a, _ in v]):.1%}, вход в маневр {np.mean([b for _, b in v]):.1%}"
                     for name, v in accuracy.items()])
    for name in EVAL_SETS:
        lines = [f"{'область':58s} {'покрытие':>9s} {'прямая':>8s} {'вход':>8s} {'маневр':>8s} {'радиус, м':>10s}"]
        for label, reps in gate_reports[name].items():
            lines.append(f"{label:58s} {np.mean([r['coverage'] for r in reps]):9.3f} "
                         f"{np.mean([r['coverage_straight'] for r in reps]):8.3f} "
                         f"{np.mean([r['coverage_entry'] for r in reps]):8.3f} "
                         f"{np.mean([r['coverage_turn'] for r in reps]):8.3f} "
                         f"{np.mean([r['radius_median'] for r in reps]):10.1f}")
        summary.section(f"{name}: строб, заданное покрытие {coverage:.2f}; радиус - медиана эквивалентного, "
                        f"среднее по {len(seeds)} seed", lines)
        plot_gate_radius({k: v for k, v in gate_reports[name].items() if "хи-квадрат" not in k},
                         OUT_DIR / f"gate_radius_{name}{tag}.png", f"Размер строба (конформный), {name}")
        width = max(len(m) for m in table_pd[name]) + 1
        for key, what in (("ade", "ADE"), ("entry", "ADE на входе в маневр")):
            lines = [f"{'метод':{width}s}" + "".join(f" {'P_d = ' + format(pd, '.1f'):>12s}" for pd in levels)]
            for method, by_pd in table_pd[name].items():
                lines.append(f"{method:{width}s}" + "".join(
                    f" {cell(by_pd[pd], key) if pd in by_pd else '-':>12s}" for pd in levels))
            summary.section(f"{name}: пропуски отметок, {what} (м)", lines)
        plot_levels(table_pd[name], levels, "ade", OUT_DIR / f"ade_vs_pd_{name}{tag}.png",
                    f"ADE при пропусках отметок, с доплером и без, {name}")
        plot_levels(table_pd[name], levels, "entry", OUT_DIR / f"entry_vs_pd_{name}{tag}.png",
                    f"ADE на входе в маневр при пропусках, {name}")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps({
        "methods": table, "pd": {n: {m: {str(pd): r for pd, r in by.items()} for m, by in t.items()}
                                 for n, t in table_pd.items()},
        "gate": gate_reports, "hyp_accuracy": accuracy}, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
