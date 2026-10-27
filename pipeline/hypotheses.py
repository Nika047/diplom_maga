"""
Гипотезы этапов 6-7 (версия 2): четыре способа улучшить основу и их сочетания.

  weight  - вес ошибкам на входе в маневр: примеры «вход» (участок entry) и «около начала маневра» при обучении
            получают вес W из config.ENTRY_WEIGHTS (остальные - 1);
  multi   - три гипотезы прогноза с вероятностями («прямо / влево / вправо» по будущему курсу истины, как
            этап 14 версии 1); как одиночный прогноз берётся самая вероятная гипотеза;
  onset   - прогноз начала маневра: вторая голова (метка data/windows.onset_labels) рядом с поправкой;
  modular - модульная схема: распознаватель «в горизонте прогноза поворот» и два эксперта («прямая»,
            «поворот»), жёсткое переключение по порогу config.MODULAR_THRESHOLD.
Сочетание = множество гипотез; multi и modular несовместимы (обе меняют выход сети). Сеть - корректор основы
(та же архитектура и окно), гиперпараметры основы или найденные генетическим подбором для сочетания.

Цели (класс будущего движения, метка начала, вес) строятся по истине и во входы сети не идут.
"""

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import optuna
import torch

import config
from common.rng import make_rng
from data.phases import PHASE_CODE
from data.windows import onset_labels, turn_class
from eval.latency import net_ms
from models.heads import HeadedCorrector, TARGET_EXTRA, parts, spec_loss
from pipeline import evaluation, latency
from pipeline.custom import fit_model
from pipeline.stage import Ctx
from training import dataset
from training.trainer import predict, train
from tuning import search

HYPS = ("weight", "multi", "onset", "modular")
TITLES = {"weight": "вес ошибкам на входе в маневр", "multi": "три гипотезы прогноза",
          "onset": "прогноз начала маневра", "modular": "модульная схема"}
SHORT = {"weight": "вес", "multi": "гипотезы", "onset": "начало маневра", "modular": "модульная"}
STRUCTURAL = ("multi", "onset", "modular")      # меняют устройство сети - для них генетический подбор


def spec_name(spec) -> str:
    return "+".join(SHORT[h] for h in HYPS if h in spec)


def compatible(spec) -> bool:
    return not ({"multi", "modular"} <= set(spec))


def combinations(survivors: list) -> list:
    """Все сочетания выживших из двух и более гипотез без несовместимой пары, по возрастанию размера."""
    out = []
    for size in range(2, len(survivors) + 1):
        out += [frozenset(c) for c in itertools.combinations(sorted(survivors, key=HYPS.index), size)
                if compatible(c)]
    return out


# ============================================================================
# Цели по истине
# ============================================================================

@dataclass
class Targets:
    important: np.ndarray       # вход в маневр или около его начала - вес
    cls: np.ndarray             # 0 прямо, 1 влево, 2 вправо
    onset: np.ndarray           # метка начала маневра
    turn: np.ndarray            # в горизонте прогноза поворот (cls > 0)


def targets_for(ws, rts: list) -> Targets:
    """Цели для окон ws; rts - треки РЛС (observe) в порядке ws.flight_ids: сетка обзоров может отличаться от сетки трека."""
    n = len(ws)
    imp, cls, ons = np.zeros(n, bool), np.zeros(n, np.int64), np.zeros(n, np.float32)
    for i, rt in enumerate(rts):
        sel = ws.track == i
        if not sel.any():
            continue
        ks = ws.k[sel]
        ons[sel] = onset_labels(rt, ks)
        imp[sel] = (rt.phase[ks] == PHASE_CODE["entry"]) | (ons[sel] > 0)
        cls[sel] = turn_class(rt, ks)
    return Targets(imp, cls, ons, cls > 0)


def pack(norm, ws, tg: Targets, w_value: float) -> np.ndarray:
    n = len(ws)
    w = np.where(tg.important, w_value, 1.0).astype(np.float32)
    return np.concatenate([norm.y(ws.y).reshape(n, -1), w[:, None], tg.cls[:, None].astype(np.float32),
                           tg.onset[:, None], tg.turn[:, None].astype(np.float32)], axis=1).astype(np.float32)


# ============================================================================
# Сети сочетания
# ============================================================================

@dataclass
class BaseConfig:
    corrector: str
    window: int
    net_input: bool
    hp: dict
    params: dict            # lr, batch, weight_decay (без числа эпох)

    @classmethod
    def from_registry(cls, ctx: Ctx, method: str) -> "BaseConfig":
        ex = ctx.registry.data[method][str(ctx.registry.seeds(method)[0])]["extra"]
        params = {k: v for k, v in ex["params"].items() if k in ("lr", "batch", "weight_decay")}
        return cls(ex["corrector"], ex["window"], ex["net_input"], ex["hp"], params)


def n_hyp_of(spec) -> int:
    return config.MULTI_HYPOTHESES if "multi" in spec else 1


@dataclass
class SpecNets:
    spec: frozenset
    nets: dict
    norm: object
    n_feat: int
    window: int
    aux: dict = field(default_factory=dict)

    def predict(self, x_norm: np.ndarray, device) -> tuple:
        """Поправка в метрах (N, H, 2) и вспомогательные выходы."""
        h = config.HORIZON_SCANS
        spec = self.spec
        aux = {}
        if "modular" in spec:
            rec = parts(predict(self.nets["recognizer"], x_norm, device), 0, h, "onset" in spec, True)
            p_turn = 1.0 / (1.0 + np.exp(-rec["turn"]))
            s = predict(self.nets["straight"], x_norm, device)[:, :2 * h].reshape(-1, h, 2)
            t = predict(self.nets["turn"], x_norm, device)[:, :2 * h].reshape(-1, h, 2)
            corr = np.where((p_turn >= config.MODULAR_THRESHOLD)[:, None, None], t, s)
            aux["turn_prob"] = p_turn
            if "onset" in spec:
                aux["onset"] = rec["onset"]
        else:
            k = n_hyp_of(spec)
            out = predict(self.nets["main"], x_norm, device)
            p = parts(torch.from_numpy(out), k, h, "onset" in spec, False)
            hyps = p["hyps"].numpy().reshape(len(out), k, h, 2)
            if k > 1:
                idx = p["logits"].numpy().argmax(axis=1)
                corr = hyps[np.arange(len(out)), idx]
                aux["hyps"], aux["probs"] = hyps, torch.softmax(p["logits"], dim=1).numpy()
            else:
                corr = hyps[:, 0]
            if "onset" in spec:
                aux["onset"] = p["onset"].numpy()
        return self.norm.y_inv(corr), aux

    def cost(self, imm_ms: float) -> dict:
        """Параметры и время отклика (процессор / видеокарта): модульная схема - распознаватель и один эксперт."""
        nets = self.nets
        n_par = int(sum(latency.n_params(m) for m in nets.values()))
        if "modular" in self.spec:
            t = [net_ms(nets["recognizer"], self.n_feat, self.window, torch.device("cpu")),
                 max(net_ms(nets["straight"], self.n_feat, self.window, torch.device("cpu")),
                     net_ms(nets["turn"], self.n_feat, self.window, torch.device("cpu")))]
            out = {"net_cpu_ms": t[0] + t[1]}
            if torch.cuda.is_available():
                g = torch.device("cuda")
                out["net_gpu_ms"] = net_ms(nets["recognizer"], self.n_feat, self.window, g) + max(
                    net_ms(nets["straight"], self.n_feat, self.window, g), net_ms(nets["turn"], self.n_feat, self.window, g))
        else:
            out = {"net_cpu_ms": net_ms(nets["main"], self.n_feat, self.window, torch.device("cpu"))}
            if torch.cuda.is_available():
                out["net_gpu_ms"] = net_ms(nets["main"], self.n_feat, self.window, torch.device("cuda"))
        res = {"n_params": n_par, "imm_ms": imm_ms, "total_cpu_ms": imm_ms + out["net_cpu_ms"], **out}
        if "net_gpu_ms" in out:
            res["total_gpu_ms"] = imm_ms + out["net_gpu_ms"]
        return res


def _builders(bc: BaseConfig, hp: dict, n_feat: int, spec) -> dict:
    h = config.HORIZON_SCANS
    onset = "onset" in spec
    if "modular" in spec:
        return {"recognizer": lambda: HeadedCorrector(bc.corrector, n_feat, h, bc.window, hp, 0, onset, True),
                "straight": lambda: HeadedCorrector(bc.corrector, n_feat, h, bc.window, hp, 1),
                "turn": lambda: HeadedCorrector(bc.corrector, n_feat, h, bc.window, hp, 1)}
    return {"main": lambda: HeadedCorrector(bc.corrector, n_feat, h, bc.window, hp, n_hyp_of(spec), onset)}


def fit_spec(ctx: Ctx, spec, bc: BaseConfig, hp: dict, params: dict, seed: int, ws_tr, tg_tr: Targets, ws_val,
             tg_val: Targets, norm, w_value: float, stage: str = None, label: str = "", first: bool = True,
             epochs: int = None, patience: int = None) -> SpecNets:
    """
    Обучает сети сочетания spec на одном seed. stage задан - через хранилище моделей (проверки в сводку,
    подпись label); None - подбор: без хранилища и сводки, короткое обучение (epochs, patience).
    """
    spec = frozenset(spec)
    h = config.HORIZON_SCANS
    n_feat = ws_tr.x.shape[-1]
    x_tr, x_val = norm.x(ws_tr.x), norm.x(ws_val.x)
    y_tr, y_val = pack(norm, ws_tr, tg_tr, w_value), pack(norm, ws_val, tg_val, w_value)
    weight, onset = "weight" in spec, "onset" in spec
    builders = _builders(bc, hp, n_feat, spec)
    tp = {**params, "epochs": epochs or ctx.st.epochs, "patience": patience or ctx.st.patience}
    jobs = {}
    if "modular" in spec:
        jobs["recognizer"] = (x_tr, y_tr, x_val, y_val, spec_loss(h, 0, True, False, onset, True))
        for role, flag in (("straight", False), ("turn", True)):
            m_tr, m_va = tg_tr.turn == flag, tg_val.turn == flag
            jobs[role] = (x_tr[m_tr], y_tr[m_tr], x_val[m_va], y_val[m_va], spec_loss(h, 1, True, weight, False, False))
    else:
        jobs["main"] = (x_tr, y_tr, x_val, y_val, spec_loss(h, n_hyp_of(spec), True, weight, onset, False))
    nets = {}
    for role, (xt, yt, xv, yv, loss) in jobs.items():
        if stage is not None:
            net, _, _ = fit_model(ctx, stage, f"{label}|{role}", seed, builders[role], xt, yt, xv, yv, tp, loss_fn=loss,
                                  overfit={"ok": lambda f, l: l < f} if first else None, plot=first)
        else:
            rng = make_rng("training", seed)
            torch.manual_seed(seed)
            net = builders[role]()
            train(net, xt, yt, xv, yv, tp, ctx.device, rng, quiet=True, loss_fn=loss)
        nets[role] = net
    return SpecNets(spec, nets, norm, n_feat, bc.window)


# ============================================================================
# Генетический подбор гиперпараметров сочетания
# ============================================================================

class SpecObjective:
    """Цели NSGA-II: ADE val (по поправкам, м) и время сети (мс). Первая попытка - параметры основы."""

    def __init__(self, ctx: Ctx, spec, bc: BaseConfig, data: dict, w_value: float):
        self.ctx, self.spec, self.bc, self.data, self.w_value = ctx, spec, bc, data, w_value

    def __call__(self, trial: optuna.Trial):
        hp = {"hidden": trial.suggest_categorical("hidden", list(config.TUNE_HIDDEN)),
              "layers": trial.suggest_categorical("layers", list(config.TUNE_LAYERS)),
              "dropout": trial.suggest_float("dropout", *config.TUNE_DROPOUT)}
        params = {"batch": trial.suggest_categorical("batch", list(config.TUNE_BATCH)),
                  "lr": trial.suggest_float("lr", *config.TUNE_LR, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", *config.TUNE_WEIGHT_DECAY, log=True)}
        d = self.data
        t0 = time.time()
        nets = fit_spec(self.ctx, self.spec, self.bc, hp, params, config.RANDOM_SEED, d["ws_tr"], d["tg_tr"], d["ws_val"],
                        d["tg_val"], d["norm"], self.w_value, epochs=self.ctx.st.tune_epochs,
                        patience=self.ctx.st.tune_patience)
        corr, _ = nets.predict(d["norm"].x(d["ws_val"].x), self.ctx.device)
        ade = float(np.linalg.norm(corr - d["ws_val"].y, axis=-1).mean())
        cost = nets.cost(0.0)
        trial.set_user_attr("seconds", time.time() - t0)
        trial.set_user_attr("n_params", cost["n_params"])
        return ade, cost["net_cpu_ms"]


def tune_spec(ctx: Ctx, spec, bc: BaseConfig, data: dict, w_value: float, n_trials: int) -> dict:
    """Лучшая по ADE конфигурация (hp, params) сочетания; результат хранится в cache/<прогон>/<этап>/<случай>/optuna.db."""
    name = spec_name(spec)
    study = search.open_study(f"hyp_{name}{'_smoke' if ctx.smoke else ''}", ctx.cache_dir / "optuna.db", n_objectives=2)
    def snap(value, choices):           # значение основы должно лежать в сетке подбора
        return min(choices, key=lambda c: abs(c - value))

    first = {"hidden": snap(bc.hp.get("hidden", config.CORRECTOR_HIDDEN), config.TUNE_HIDDEN),
             "layers": snap(bc.hp.get("layers", config.CORRECTOR_LAYERS), config.TUNE_LAYERS),
             "dropout": bc.hp.get("dropout", config.CORRECTOR_DROPOUT),
             "batch": snap(bc.params["batch"], config.TUNE_BATCH), "lr": bc.params["lr"],
             "weight_decay": bc.params["weight_decay"]}
    search.run_budget(study, SpecObjective(ctx, spec, bc, data, w_value), n_trials, first)
    done = search.completed(study)
    best = min(done, key=lambda t: t.values[0])
    ctx.log.info("Подбор %s: лучшее ADE val %.2f м (основа %.2f), %d попыток", name, best.values[0],
                 search.default_trial(study).values[0], len(done))
    return {"hp": {k: best.params[k] for k in ("hidden", "layers", "dropout")},
            "params": {k: best.params[k] for k in ("batch", "lr", "weight_decay")},
            "ade_val": best.values[0], "ade_val_base": search.default_trial(study).values[0], "trials": len(done)}


def hp_for(bc: BaseConfig, tuned: dict = None) -> tuple:
    """(hp сети, параметры обучения): основы или найденные подбором."""
    if tuned:
        return {**bc.hp, **tuned["hp"]}, tuned["params"]
    return bc.hp, bc.params


def evaluate_spec(ctx: Ctx, cd, spec, bc: BaseConfig, hp: dict, params: dict, ws_tr, tg_tr: Targets, norm,
                  w_value: float, stage: str, label: str) -> list:
    """Обучение по seed, оценка на val, test (holdout), запись в реестр под "<stage>/<label>". Возвращает по seed результаты."""
    st = ctx.st
    ws_val = cd.eval("val")[0]
    tg_val = targets_for(ws_val, cd.eval("val")[1])
    imm_ms = latency.imm_ms_per_scan(cd, "val")
    out, costs = [], None
    for s_i, seed in enumerate(st.seeds):
        nets = fit_spec(ctx, spec, bc, hp, params, seed, ws_tr, tg_tr, ws_val, tg_val, norm, w_value, stage, label,
                        first=s_i == 0)
        preds, auxes, fcs = {}, {}, {}
        for name in config.EVAL_SETS:
            preds[name], auxes[name] = nets.predict(norm.x(cd.eval(name)[0].x), ctx.device)
            fcs[name] = evaluation.hybrid_forecasts(cd, name, preds[name])
        costs = costs or nets.cost(imm_ms)
        reps = evaluation.reports(cd, fcs)
        extra = {**costs, "spec": sorted(spec), "hp": hp, "params": params, "window": bc.window,
                 "net_input": bc.net_input, "corrector": bc.corrector, "mode": "hybrid", "w_value": w_value}
        extra.update(aux_metrics(cd, spec, auxes))
        ctx.registry.record(f"{stage}/{label}", seed, reps, extra, fcs=fcs)
        out.append({"seed": seed, "nets": nets, "preds": preds, "aux": auxes, "fcs": fcs, "reports": reps})
    return out


def aux_metrics(cd, spec, auxes: dict) -> dict:
    """Числа гипотез кроме ADE: угадывание класса и охват лучшей из трёх (multi), AP начала маневра (onset), распознаватель."""
    from eval import metrics

    res = {}
    for name in ("val", "test"):
        if name not in auxes:
            continue
        ws, rts, _ = cd.eval(name)
        tg = targets_for(ws, rts)
        aux = auxes[name]
        entry = np.array([rts[t].phase[k] == PHASE_CODE["entry"] for t, k in zip(ws.track, ws.k)], bool)
        if "probs" in aux:
            guess = aux["probs"].argmax(axis=1) == tg.cls
            hyps = aux["hyps"]
            best = np.linalg.norm(hyps - ws.y[:, None], axis=-1).mean(axis=-1).min(axis=1)
            res[f"{name}_class_acc"] = float(guess.mean())
            res[f"{name}_class_acc_entry"] = float(guess[entry].mean()) if entry.any() else float("nan")
            res[f"{name}_best_of_k_ade"] = float(best.mean())
        if "onset" in aux:
            res[f"{name}_onset_ap"] = float(metrics.average_precision(aux["onset"], tg.onset))
        if "turn_prob" in aux:
            hit = (aux["turn_prob"] >= config.MODULAR_THRESHOLD) == tg.turn
            res[f"{name}_recognizer_acc"] = float(hit.mean())
            res[f"{name}_recognizer_acc_entry"] = float(hit[entry].mean()) if entry.any() else float("nan")
    return res
