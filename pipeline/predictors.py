"""
Загрузка обученных сетей из хранилища для повторных оценок этапа 9 (пропуски отметок, переобучение под шум,
сгенерированный test): поправка сети на окнах любого CaseData.

Поддержаны методы корректоров (этапы 2, 4) и сочетаний гипотез (этапы 6, 7); схемы Б этапа 5 в итоговые модели
не попадают (этап 7 строится на корректоре основы).
"""

import numpy as np

import config
from models import build_corrector
from pipeline import hypotheses as hyp
from pipeline.casedata import CaseData
from pipeline.stage import Ctx
from training import dataset
from training.trainer import predict


def corrections(ctx: Ctx, method: str, seed: int, cd: CaseData, names=None) -> dict:
    """{набор: поправка сети (N, H, 2) в метрах} для окон cd.eval(набор) метода method (запись реестра), seed."""
    names = names or config.EVAL_SETS
    stage, label = method.split("/", 1)
    ex = ctx.registry.data[method][str(seed)]["extra"]
    out = {}
    if ex.get("spec"):
        spec = frozenset(ex["spec"])
        bc = hyp.BaseConfig(ex["corrector"], ex["window"], ex["net_input"], ex["hp"], ex["params"])
        nets, norm = {}, None
        roles = ("recognizer", "straight", "turn") if "modular" in spec else ("main",)
        for role in roles:
            payload, _ = ctx.store.load(stage, ctx.case, f"{label}|{role}", seed)
            norm = dataset.Normalizer.from_dict(payload["normalizer"])
            n_feat = len(payload["normalizer"]["x_mean"])
            net = hyp._builders(bc, ex["hp"], n_feat, spec)[role]()
            net.load_state_dict(payload["state_dict"])
            nets[role] = net
        sn = hyp.SpecNets(spec, nets, norm, n_feat, bc.window)
        for name in names:
            out[name] = sn.predict(norm.x(cd.eval(name)[0].x), ctx.device)[0]
        return out
    payload, _ = ctx.store.load(stage, ctx.case, f"{label}, seed {seed}", seed)
    norm = dataset.Normalizer.from_dict(payload["normalizer"])
    model = build_corrector(payload["corrector"], len(payload["normalizer"]["x_mean"]), config.HORIZON_SCANS,
                            payload["window"], **payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    for name in names:
        out[name] = norm.y_inv(predict(model, norm.x(cd.eval(name)[0].x), ctx.device))
    return out


def forecast_lists(cd: CaseData, corr: dict) -> dict:
    """Поправки -> прогнозы гибрида по трассам {набор: [(K, H, 3)]} на обзорах cd."""
    from pipeline import evaluation

    return {name: evaluation.hybrid_forecasts(cd, name, c) for name, c in corr.items()}


def mean_corr(corrs: list) -> dict:
    """Ансамбль seed: среднее поправок."""
    return {name: np.mean([c[name] for c in corrs], axis=0) for name in corrs[0]}
