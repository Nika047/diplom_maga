"""
Обучение и оценка корректора по seed для этапов версии 2.

fit_and_eval: для каждого seed - run_experiment через хранилище моделей (готовая модель не
обучается заново), прогнозы на val, test (holdout), отчёты с ошибками по полётам - в реестр
(pipeline/registry.py) вместе с числом параметров, временем обучения и временем отклика.
"""

import config
from models import build_corrector
from pipeline import evaluation, latency
from pipeline.casedata import CaseData
from pipeline.stage import Ctx
from training.experiment import run_experiment


def model_costs(cd: CaseData, corrector: str, mode: str, n_features: int, window: int, hp: dict, device) -> dict:
    """Параметры сети и время отклика (шаг IMM плюс проход сети): миллисекунды на обзор, процессор и видеокарта."""
    model = build_corrector(corrector, n_features, config.HORIZON_SCANS, window, **hp)
    imm_ms = latency.imm_ms_per_scan(cd, "val") if mode == "hybrid" else 0.0
    return {"n_params": latency.n_params(model), **latency.response_times(imm_ms, model, n_features, window)}


def fit_and_eval(ctx: Ctx, cd: CaseData, stage: str, label: str, corrector: str, mode: str, ws_tr, params: dict,
                 hp: dict, window: int, seeds: tuple = None, overfit: bool = True, weights: tuple = None,
                 record: bool = True) -> list:
    """
    Возвращает по seed {"seed", "history", "preds": {набор: прогноз сети}, "fcs": {набор: прогнозы по трассам},
    "reports": {...}}. Метод в реестре - "<этап>/<подпись>". params - lr, batch, weight_decay; число эпох и
    терпение берутся из настроек, если не заданы.
    """
    st = ctx.st
    seeds = seeds or st.seeds
    params = {"epochs": st.epochs, "patience": st.patience, **params}
    ws_val = cd.eval("val")[0]
    extra_sets = {n: cd.eval(n)[0] for n in config.EVAL_SETS[1:]}
    build = evaluation.hybrid_forecasts if mode == "hybrid" else evaluation.e2e_forecasts
    out, costs = [], None
    for s_i, seed in enumerate(seeds):
        res = run_experiment(f"{label}, seed {seed}", corrector, mode, ws_tr, ws_val, params, ctx.device, ctx.summary,
                             ctx.log, ctx.smoke, ctx.out_dir, ctx.art_dir, window=window, seed=seed, model_kwargs=hp,
                             overfit=overfit and s_i == 0, extra_sets=extra_sets, weights=weights, store=ctx.store,
                             store_key=(stage, ctx.case))
        preds = {"val": res["pred"], **res["extra"]}
        fcs = {n: build(cd, n, preds[n]) for n in config.EVAL_SETS}
        reps = evaluation.reports(cd, fcs)
        if record:
            if costs is None:
                costs = model_costs(cd, corrector, mode, ws_tr.view(mode).x.shape[-1], window, hp, ctx.device)
            ctx.registry.record(f"{stage}/{label}", seed, reps,
                                {**costs, "train_seconds": res["history"]["train_seconds"],
                                 "epochs": len(res["history"]["epochs"]), "window": window, "hp": hp,
                                 "params": params, "net_input": cd.net_input})
        out.append({"seed": seed, "history": res["history"], "preds": preds, "fcs": fcs, "reports": reps})
    return out
