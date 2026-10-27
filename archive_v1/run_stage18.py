"""
Этап 18: доплер по телеметрии (решение пользователя 26.10.2026).

На этапах 12-17 истинная скорость для доплера у реальных полётов считалась по трём последним
точкам GPS-траектории - на реальных трассах это шумно (оценка 1-1,5 м/с). Новый компонент -
истинная скорость из самого лога: скорость GPS-приёмника (GSpd, Hdg, VSpd группы GPS,
data/flight_logs, Track.vel_log), config.DOPPLER_TRUTH = "telemetry". Измерение РЛС - её проекция
на линию визирования плюс шум RADAR_SIGMA_VR, как раньше. Синтетика не меняется (скорость по
гладкой траектории точная). Телеметрия во входы моделей не идёт - только в модель измерения.

Сравнение на одних обзорах, итоговая модель - IMM с доплером + GRU, «+ все расширения» (доплер
только в IMM), гиперпараметры этапа 4, 3 seed:
  - IMM без доплера, IMM с доплером по траектории и по телеметрии;
  - гибрид без доплера (веса этапа 9);
  - итоговая модель этапа 12 (доплер по траектории) - на своих обзорах и на доплере по телеметрии;
  - итоговая модель, переобученная на доплере по телеметрии, и ансамбль её 3 seed.
Рядом - насколько доплер по телеметрии отличается от доплера по траектории на реальных полётах.
Скорость приёмника и GPS-траектория (истина для оценки) на поворотах расходятся на единицы м/с,
поэтому СКО доплера, которое закладывает фильтр (config.FILTER_SIGMA_VR), выбирается по ADE IMM на
val из config.DOPPLER_FILTER_SIGMAS; шум самого измерения остаётся RADAR_SIGMA_VR.

Треки этапа 0 собраны до появления Track.vel_log, поэтому скорость подклеивается заново из
логов (flight_logs.attach_log_velocity) с проверкой совпадения координат; этап 0 не
перезапускается.

Запуск (домашний ПК, в фоне - см. PLATFORM.md): .venv\\Scripts\\python.exe run_stage18.py
Смоук (рабочий ПК, нужны смоуки этапов 4, 9 и 12): .venv\\Scripts\\python.exe run_stage18.py --smoke

Проверки: скорость подклеена ко всем реальным трекам без несовпадений; курс и модуль скорости по
телеметрии совпадают со скоростью по координатам (медианы); известный ответ радиальной скорости
по заданной скорости; у синтетики доплер и отметки от источника скорости не меняются; веса
этапов 9 и 12 повторяют свои результаты; ансамбль не хуже средней сети; проверки обучения.
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
from data import augment, generator  # noqa: E402
from data.flight_logs import attach_log_velocity  # noqa: E402
from data.windows import apply_correction  # noqa: E402
from eval.checks import StageSummary  # noqa: E402
from eval.compare import compare, mask_forecast, source_section  # noqa: E402
from radar.model import observe, true_radial_velocity  # noqa: E402
from run_stage4 import load_data, summarize_method  # noqa: E402
from run_stage8 import load_model  # noqa: E402
from run_stage9 import KINDS  # noqa: E402
from run_stage11 import reference_ade  # noqa: E402
from run_stage12 import doppler_override  # noqa: E402
from run_stage17 import ref12_val  # noqa: E402
from training import dataset  # noqa: E402
from training.experiment import run_experiment, slug  # noqa: E402
from training.trainer import pick_device, predict  # noqa: E402
from tuning import search  # noqa: E402
from viz.plots import _save  # noqa: E402

# Настройки, зависящие от устройства
FINAL = {"epochs": 80, "patience": 10}
SMOKE = {"epochs": 2, "patience": 2, "seeds": (42, 43)}

STAGE = "stage18"
OUT_DIR = config.RESULTS_DIR / STAGE
ART = config.ARTIFACTS_DIR / STAGE
EVAL_SETS = config.EVAL_SETS
HYB = "IMM+GRU"
NAME12 = f"{HYB}, + все расширения, доплер в IMM"                    # итоговая модель на этапе 12
NODOP = f"{HYB}, + все расширения, без доплера"
REF_P = "итоговая (этап 12, доплер по траектории)"
REF_PT = "итоговая этапа 12 на доплере по телеметрии"
REF_T = "итоговая, обучена на доплере по телеметрии"
# вариант обзоров: переопределения config
VARIANTS = {"без доплера": {}, "траектория": {**doppler_override("доплер в IMM"), "DOPPLER_TRUTH": "positions"},
            "телеметрия": {**doppler_override("доплер в IMM"), "DOPPLER_TRUTH": "telemetry"}}


def velocity_agreement(tracks: list) -> dict:
    """По типам аппаратов: отношение модулей горизонтальной скорости (телеметрия / координаты), отличие
    курса, °, и модуль разности горизонтальной скорости, м/с - на точках со скоростью больше 8 м/с."""
    rows = {}
    for tr in tracks:
        if tr.vel_log is None or len(tr.t) < 5:
            continue
        v = (tr.pos[2:] - tr.pos[:-2]) / (tr.t[2:] - tr.t[:-2])[:, None]
        vl = tr.vel_log[1:-1]
        ok = np.isfinite(vl).all(axis=1) & (np.hypot(v[:, 0], v[:, 1]) > 8.0)
        if not ok.any():
            continue
        r = rows.setdefault(tr.aircraft, {"ratio": [], "course": [], "dv": []})
        r["ratio"].append(np.hypot(vl[ok, 0], vl[ok, 1]) / np.hypot(v[ok, 0], v[ok, 1]))
        d = np.arctan2(vl[ok, 0], vl[ok, 1]) - np.arctan2(v[ok, 0], v[ok, 1])
        r["course"].append(np.degrees(np.abs((d + np.pi) % (2 * np.pi) - np.pi)))
        r["dv"].append(np.linalg.norm(vl[ok, :2] - v[ok, :2], axis=1))
    return {a: {k: np.concatenate(v) for k, v in r.items()} for a, r in rows.items()}


def plot_vr_difference(diff: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lim = np.quantile(np.abs(diff), 0.99)
    ax.hist(diff, bins=80, range=(-lim, lim), color="#5c6bc0")
    ax.set_xlabel("радиальная скорость: по телеметрии минус по траектории, м/с")
    ax.set_ylabel("обзоров")
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 18: доплер по телеметрии")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tag = "_smoke" if args.smoke else ""
    log = setup_logging(f"{STAGE}{tag}")
    summary = StageSummary("Этап 18: доплер по телеметрии" + (" (СМОУК: не результат)" if args.smoke else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    if device.type != "cuda" and not args.smoke:
        log.error("GPU нет. Обучение этапа 18 - только на домашнем ПК; здесь запускать с --smoke")
        return 2
    cache4 = config.CACHE_DIR / ("stage4_smoke" if args.smoke else "stage4")
    cache = config.CACHE_DIR / f"{STAGE}{tag}"
    epochs = SMOKE["epochs"] if args.smoke else FINAL["epochs"]
    patience = SMOKE["patience"] if args.smoke else FINAL["patience"]
    seeds = SMOKE["seeds"] if args.smoke else config.FINAL_SEEDS
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
    paths = {REF_P: [config.ARTIFACTS_DIR / "stage12" / f"{slug(f'{NAME12}, seed {s}')}{tag}.pt" for s in seeds],
             NODOP: [config.ARTIFACTS_DIR / "stage9" / f"{slug(f'+ все расширения, seed {s}')}{tag}.pt" for s in seeds]}
    missing = [str(p) for ps in paths.values() for p in ps if not p.exists()]
    summary.check("Веса этапов 9 и 12 найдены", not missing, f"нет {missing}" if missing else "все")
    if missing:
        summary.write(OUT_DIR / f"summary{tag}.txt")
        return 1

    # Скорость по телеметрии - к реальным трекам до расширения (копии её наследуют)
    sets = load_data(args.smoke)
    real_all = [tr for name in ("train",) + EVAL_SETS for tr in sets[name] if tr.source != "synth"]
    attached, mismatch = attach_log_velocity(real_all)
    summary.check("Скорость по телеметрии подклеена ко всем реальным трекам", attached == len(real_all) and mismatch == 0,
                  f"подклеено {attached} из {len(real_all)}, несовпадений координат {mismatch}")
    agree = velocity_agreement(real_all)
    cat = {k: np.concatenate([a[k] for a in agree.values()]) for k in ("ratio", "course", "dv")}
    summary.check("Скорость по телеметрии согласована со скоростью по координатам (медианы)",
                  abs(np.median(cat["ratio"]) - 1.0) <= 0.05 and np.median(cat["course"]) <= 5.0,
                  f"отношение модулей {np.median(cat['ratio']):.3f}, отличие курса {np.median(cat['course']):.1f}°, "
                  f"точек {len(cat['ratio'])}")
    summary.section("Скорость по телеметрии против скорости по координатам (центральная разность), точки быстрее "
                    "8 м/с, медианы", [
                        f"{a:12s} отношение {np.median(r['ratio']):.3f}, курс {np.median(r['course']):4.1f}°, "
                        f"|разность| {np.median(r['dv']):.2f} м/с (90% - {np.quantile(r['dv'], 0.9):.2f}), "
                        f"точек {len(r['ratio'])}" for a, r in sorted(agree.items())])

    # Известный ответ: при постоянной скорости заданная скорость = формула назад по трём точкам
    t = np.arange(30) * config.SCAN_PERIOD
    truth = np.column_stack([-300.0 + 20.0 * t, np.full_like(t, 1000.0), 5.0 + 0.0 * t])
    vel = np.tile([20.0, 0.0, 0.0], (len(t), 1))
    a, b = true_radial_velocity(truth, np.zeros(3)), true_radial_velocity(truth, np.zeros(3), vel=vel)
    summary.check("Доплер по заданной скорости: известный ответ (прямолинейное движение мимо РЛС)",
                  float(np.abs(a[2:] - b[2:]).max()) < 1e-9 and np.isfinite(b[0]),
                  f"наибольшее отличие от формулы по траектории {np.abs(a[2:] - b[2:]).max():.1e} м/с")
    synth = sets["synth"][:5]
    with search.config_override(VARIANTS["траектория"]):
        p_rt = [observe(tr) for tr in synth]
    with search.config_override(VARIANTS["телеметрия"]):
        t_rt = [observe(tr) for tr in synth]
    same = all(np.array_equal(x.vr, y.vr, equal_nan=True) and np.array_equal(x.meas, y.meas) for x, y in zip(p_rt, t_rt))
    summary.check("У синтетики доплер и отметки от источника скорости не меняются", same, f"трасс {len(synth)}")

    with open(config.CACHE_DIR / "stage0" / "calibration.json", encoding="utf-8") as fh:
        limits = generator.Limits(**json.load(fh))
    real, synth_tr = sets["train"], sets["synth"]
    copies = [c for kind in KINDS for c in augment.augment(real, kind, limits)[0]]

    built = {}
    for variant, override in VARIANTS.items():
        with search.config_override({**imm, **override}):
            for name in EVAL_SETS:
                built[(name, variant)] = dataset.build(sets[name], window=window)
    # Насколько фильтр верит доплеру по телеметрии: СКО в фильтре (FILTER_SIGMA_VR) - по val у IMM
    sigma_key = {}
    for s in config.DOPPLER_FILTER_SIGMAS:
        sigma_key[s] = "телеметрия" if s == config.RADAR_SIGMA_VR else f"телеметрия, СКО в фильтре {s:g}"
        if s != config.RADAR_SIGMA_VR:
            with search.config_override({**imm, **VARIANTS["телеметрия"], "FILTER_SIGMA_VR": s}):
                for name in EVAL_SETS:
                    built[(name, sigma_key[s])] = dataset.build(sets[name], window=window)
    ks = {name: [np.array(sorted(built[(name, "без доплера")][0].k[built[(name, "без доплера")][0].track == i]),
                          dtype=int) for i in range(len(sets[name]))] for name in EVAL_SETS}
    rts_ev = {name: built[(name, "без доплера")][1] for name in EVAL_SETS}

    # Насколько доплер по телеметрии отличается от доплера по траектории (реальные трассы, без шума РЛС)
    diffs = {}
    for name in EVAL_SETS:
        d = [true_radial_velocity(b.truth, b.radar_pos, vel=np.column_stack(
                 [np.interp(b.t, tr.t, tr.vel_log[:, j]) for j in range(3)])) - true_radial_velocity(b.truth, b.radar_pos)
             for tr, b in zip(sets[name], built[(name, "телеметрия")][1]) if tr.vel_log is not None]
        diffs[name] = np.concatenate(d)[np.isfinite(np.concatenate(d))] if d else np.zeros(0)
    summary.section("Радиальная скорость на реальных трассах: по телеметрии минус по траектории (без шума РЛС), м/с", [
        f"{name:8s} СКО {np.std(v):.2f}, медиана |разности| {np.median(np.abs(v)):.2f}, 90% |разности| "
        f"{np.quantile(np.abs(v), 0.9):.2f}, обзоров {len(v)}" for name, v in diffs.items() if len(v)])
    if len(diffs["test"]):
        plot_vr_difference(diffs["test"], OUT_DIR / f"vr_difference_test{tag}.png",
                           "Доплер по телеметрии против доплера по траектории, реальные трассы test")

    table = {name: {} for name in EVAL_SETS}
    forecasts = {name: {} for name in EVAL_SETS}
    imm_rows = [("IMM", "без доплера"), ("IMM с доплером по траектории", "траектория")] + [
        (f"IMM с доплером по телеметрии, СКО в фильтре {s:g} м/с", sigma_key[s]) for s in config.DOPPLER_FILTER_SIGMAS]
    for name in EVAL_SETS:
        for method, variant in imm_rows:
            fc = [mask_forecast(r.forecast, k) for r, k in zip(built[(name, variant)][2], ks[name])]
            table[name][method] = [summarize_method(rts_ev[name], fc)]
            forecasts[name][method] = fc
    best_sigma = min(config.DOPPLER_FILTER_SIGMAS,
                     key=lambda s: table["val"][f"IMM с доплером по телеметрии, СКО в фильтре {s:g} м/с"][0]["ade"])
    tel = sigma_key[best_sigma]
    summary.section("СКО радиальной скорости в фильтре для доплера по телеметрии - по ADE IMM на val", [
        f"{s:g} м/с: ADE val {table['val'][f'IMM с доплером по телеметрии, СКО в фильтре {s:g} м/с'][0]['ade']:.1f} м"
        for s in config.DOPPLER_FILTER_SIGMAS] + [f"выбрано {best_sigma:g} м/с; на нём обучается сеть"])

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

    # Сохранённые сети: без доплера; итоговая этапа 12 - на своих обзорах и на доплере по телеметрии
    for method, label, variant in ((NODOP, NODOP, "без доплера"), (REF_P, REF_P, "траектория"),
                                   (REF_PT, REF_P, "телеметрия")):
        for s_i, (seed, path) in enumerate(zip(seeds, paths[label])):
            model, norm, _ = load_model(path, device)
            record(method, s_i, variant, {name: norm.y_inv(predict(model, norm.x(built[(name, variant)][0].x), device))
                                          for name in EVAL_SETS})
            if method == REF_PT or (method == NODOP and s_i > 0):
                continue
            ref = ref12_val(NAME12, s_i, tag) if method == REF_P else reference_ade("stage9", "+ все расширения", tag,
                                                                                    best_all)
            got = table["val"][method][-1]["ade"]
            summary.check(f"{method}, seed {seed}: веса повторяют свой результат на val",
                          ref is not None and abs(got / ref - 1) <= 0.005,
                          f"ADE val там {ref if ref is None else round(ref, 2)} м, здесь {got:.2f} м")

    # Переобучение на доплере по телеметрии с выбранным СКО в фильтре
    ref_t = f"{REF_T}, СКО в фильтре {best_sigma:g}"      # без «/»: имя идёт в файл весов
    with search.config_override({**imm, **VARIANTS["телеметрия"], "FILTER_SIGMA_VR": best_sigma}):
        ws_tr = dataset.concat([
            dataset.load_or_build(f"train_w{window}_d10_tel_s{best_sigma:g}", real + synth_tr, cache, window=window),
            dataset.load_or_build(f"aug_all_w{window}_d10_tel_s{best_sigma:g}", copies, cache, window=window)])
    log.info("%s: %d окон", ref_t, len(ws_tr))
    corr_t = []
    for s_i, seed in enumerate(seeds):
        res = run_experiment(f"{ref_t}, seed {seed}", "GRU", "hybrid", ws_tr, built[("val", tel)][0], params,
                             device, summary, log, args.smoke, OUT_DIR, ART, window=window, seed=seed, model_kwargs=hp,
                             overfit=s_i == 0, extra_sets={n: built[(n, tel)][0] for n in EVAL_SETS[1:]})
        corr_t.append({"val": res["pred"], **res["extra"]})
        record(ref_t, s_i, tel, corr_t[-1])
    del ws_tr
    ens = f"ансамбль {len(seeds)} seed: {ref_t}"
    record(ens, 0, tel, {name: np.mean([c[name] for c in corr_t], axis=0) for name in EVAL_SETS})
    summary.check("Ансамбль не хуже средней сети (неравенство треугольника)",
                  all(table[n][ens][0]["ade"] <= np.mean([r["ade"] for r in table[n][ref_t]]) + 1e-9 for n in EVAL_SETS),
                  f"наборов {len(EVAL_SETS)}")

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
        summary.section(f"{name}: ADE (м), сети - среднее ± СКО по {len(seeds)} seed", lines)
        source_section(summary, name, table[name])
        main_methods = ["IMM", f"IMM с доплером по телеметрии, СКО в фильтре {best_sigma:g} м/с", NODOP, REF_P, ref_t]
        compare(rts_ev[name], {m: forecasts[name][m] for m in main_methods if m in forecasts[name]}, summary,
                f"{name} (seed {seeds[0]})", OUT_DIR, f"{name}{tag}")

    (OUT_DIR / f"metrics{tag}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.section("Время", [f"{(time.time() - t0) / 60:.1f} мин на {device}"])
    summary.write(OUT_DIR / f"summary{tag}.txt")
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
