"""
Этап 3 (версия 2): длина окна, размер сети и объём данных.

Гибриды IMM + GRU и IMM + Transformer с гиперпараметрами этапа 2 (лучшая конфигурация); 3 seed, оценка на
одних обзорах (общих для всех длин окна - по самому длинному окну; в смоуке окна 20 и 60), val и test (holdout):
  окно    - WINDOW_VARIANTS обзоров;
  размер  - ширина SCALE_HIDDEN x число слоёв SCALE_LAYERS;
  данные  - только реальные; реальные + расширение; реальные + расширение + доля синтетики
            (SCALE_SYNTH от x1; доли выше объёма прогона не берутся: в прогоне x4 до 4, в x1 до 1).
Вопросы: помогает ли длинное окно, большая сеть, больше данных (в версии 1: нет, плато после x1).
Сети хранятся в model_store/<прогон>/stage3/<случай>/, прерванный запуск продолжается.

  .venv\\Scripts\\python.exe run_stage3.py [--case ...] [--only window,size,data] [--smoke]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from common import cases  # noqa: E402
from pipeline import data as pdata  # noqa: E402
from pipeline.runs import fit_and_eval  # noqa: E402
from pipeline.stage import Ctx, run_stage  # noqa: E402
from pipeline.tables import method_table  # noqa: E402
from tuning import corrector_search  # noqa: E402
from viz.tuning_plots import plot_series  # noqa: E402

STAGE = "stage3"
ARCHS = (("IMM+GRU", "GRU"), ("IMM+Transformer", "Transformer"))


def add_args(parser) -> None:
    parser.add_argument("--only", default="window,size,data", help="какие опыты: window,size,data")


def run_case(ctx: Ctx) -> bool:
    st, summary = ctx.st, ctx.summary
    only = {s.strip() for s in ctx.args.only.split(",")}
    state = ctx.state("corr")
    if state is None or any(label not in state for label, _ in ARCHS):
        summary.check("Результаты этапа 2 найдены", False, "нужны IMM+GRU и IMM+Transformer в состоянии corr")
        return False
    windows = config.WINDOW_VARIANTS[:2] if ctx.smoke else config.WINDOW_VARIANTS    # смоук: короче и меньше
    hiddens = config.SCALE_HIDDEN[:2] if ctx.smoke else config.SCALE_HIDDEN
    layer_list = config.SCALE_LAYERS[:1] if ctx.smoke else config.SCALE_LAYERS
    ks_window = max(windows)
    sets = ctx.sets
    tracks_main = pdata.train_tracks(sets)
    methods = {"window": [], "size": [], "data": []}
    point = {}      # метод -> (параметры, x для графика)

    def run(kind: str, label: str, corrector: str, params: dict, hp: dict, window: int, net: bool, tracks: list,
            key: str, x) -> None:
        cd = ctx.casedata(window=window, net_input=net).variant(ks_window=ks_window)
        out = fit_and_eval(ctx, cd, STAGE, label, corrector, "hybrid", cd.windows(key, tracks), params, hp, window)
        methods[kind].append(label)
        point[label] = x
        return out

    base_ade = {}
    for arch, corrector in ARCHS:
        window0, net, hp0, train_params = corrector_search.split(state[arch]["best"])
        if "window" in only:
            for w in windows:
                out = run("window", f"{arch} окно {w}", corrector, train_params, hp0, w, net, tracks_main, "main", w)
                if w == config.WINDOW_SCANS:
                    base_ade[arch] = out[0]["reports"]["val"]["ade"]
        if "size" in only:
            for hidden in hiddens:
                for layers in layer_list:
                    hp = {**hp0, "hidden": hidden, "layers": layers}
                    run("size", f"{arch} ширина {hidden} слоёв {layers}", corrector, train_params, hp, window0, net,
                        tracks_main, "main", None)
    if "data" in only:
        arch, corrector = ARCHS[0]
        window0, net, hp0, train_params = corrector_search.split(state[arch]["best"])
        variants = [("только реальные", False, 0.0, 0.0), ("реальные + расширение", True, 0.0, 0.0)]
        variants += [(f"реальные + расширение + синтетика x{f:g}", True, f / config.SYNTH_FACTOR, f)
                     for f in config.SCALE_SYNTH if 0 < f <= config.SYNTH_FACTOR]
        for name, aug, share, f in variants:
            run("data", f"{arch} данные: {name}", corrector, train_params, hp0, window0, net,
                pdata.train_tracks(sets, aug=aug, synth_share=share), f"data_{int(aug)}_{share:g}", f)

    reg = ctx.registry
    if base_ade:
        ga = state["IMM+GRU"]["ade_val"]
        got = base_ade["IMM+GRU"]
        summary.check("Окно 20 с параметрами этапа 2 повторяет его val (общие обзоры по окну 100)",
                      abs(got / ga - 1) <= 0.15, f"ADE val этап 2: {ga:.2f} м, здесь {got:.2f} м")
    for kind, title in (("window", "Длина окна"), ("size", "Размер сети"), ("data", "Объём данных")):
        if not methods[kind]:
            continue
        names = [f"{STAGE}/{m}" for m in methods[kind]]
        for name in config.EVAL_SETS:
            summary.section(f"{title}: {name}, ADE (м), среднее ± СКО по {len(st.seeds)} seed", method_table(
                reg, names, name, ref=names[0]))
    series = {}
    for arch, _ in ARCHS:
        ms = [m for m in methods["window"] if m.startswith(arch)]
        if ms:
            series[arch] = [reg.metric(f"{STAGE}/{m}", "test") for m in ms]
    if series:
        plot_series(list(windows), series, ctx.out_dir / "window.png",
                    f"ADE test от длины окна: {cases.CASE_TITLES[ctx.case]}", "окно, обзоров")
    series = {}
    for arch, _ in ARCHS:
        ms = [m for m in methods["size"] if m.startswith(arch)]
        if ms:
            series[arch] = [reg.metric(f"{STAGE}/{m}", "test") for m in ms]
    if series:
        xs = [reg.data[f"{STAGE}/{m}"][str(st.seeds[0])]["extra"]["n_params"] for m in methods["size"] if m.startswith(ARCHS[0][0])]
        order = sorted(range(len(xs)), key=xs.__getitem__)
        series = {a: [v[i] for i in order] for a, v in series.items() if len(v) == len(xs)}
        plot_series([xs[i] for i in order], series, ctx.out_dir / "size.png",
                    f"ADE test от числа параметров: {cases.CASE_TITLES[ctx.case]}", "параметров", logx=True)
    if methods["data"]:
        sy = [m for m in methods["data"] if "синтетика" in m or "расширение" in m or "реальные" in m]
        plot_series(list(range(len(sy))), {ARCHS[0][0]: [reg.metric(f"{STAGE}/{m}", "test") for m in sy]},
                    ctx.out_dir / "data.png", f"ADE test от объёма данных: {cases.CASE_TITLES[ctx.case]}",
                    "только реальные, + расширение, + синтетика (x0,25 ... )")
    return True


def main() -> int:
    return run_stage(STAGE, "Этап 3: окно, размер сети, объём данных", run_case, needs_gpu=True, add_args=add_args)


if __name__ == "__main__":
    sys.exit(main())
