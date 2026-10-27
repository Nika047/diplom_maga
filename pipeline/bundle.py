"""
Итоговая модель случая как самостоятельный пакет (этап 11) и предсказатель по нему (predict.py).

Пакет artifacts/<прогон>/stage11/<случай>/: config.json (подмены config - параметры IMM, флаги доплера,
доверие к доплеру; окно, тип сети, гиперпараметры, сочетание гипотез, время отклика) и member_<seed>.pt
(веса и нормализатор). Пакет не зависит от реестра и хранилища моделей: по нему программа работает на
любом компьютере с проектом.
"""

import json
from pathlib import Path

import numpy as np
import torch

import config
from common.cases import config_override
from data.windows import apply_correction, build_windows
from filters.imm import IMM
from models import build_corrector
from pipeline import hypotheses as hyp
from training import dataset
from training.trainer import predict


def export_member(ctx, method: str, seed: int, out: Path) -> dict:
    """Пишет member_<seed>.pt для метода реестра (корректор или сочетание гипотез); возвращает описание."""
    stage, label = method.split("/", 1)
    ex = ctx.registry.data[method][str(seed)]["extra"]
    if ex.get("spec"):
        spec = sorted(ex["spec"])
        roles = ("recognizer", "straight", "turn") if "modular" in spec else ("main",)
        states, norm = {}, None
        for role in roles:
            payload, _ = ctx.store.load(stage, ctx.case, f"{label}|{role}", seed)
            states[role], norm = payload["state_dict"], payload["normalizer"]
        torch.save({"kind": "spec", "spec": spec, "states": states, "normalizer": norm, "corrector": ex["corrector"],
                    "window": ex["window"], "hp": ex["hp"], "n_feat": len(norm["x_mean"])}, out / f"member_{seed}.pt")
    else:
        payload, _ = ctx.store.load(stage, ctx.case, f"{label}, seed {seed}", seed)
        torch.save({"kind": "corrector", "state_dict": payload["state_dict"], "normalizer": payload["normalizer"],
                    "corrector": payload["corrector"], "window": payload["window"],
                    "model_kwargs": payload["model_kwargs"]}, out / f"member_{seed}.pt")
    return {"seed": seed, "file": f"member_{seed}.pt"}


class _Member:
    def __init__(self, path: Path, device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.kind, self.device = ck["kind"], device
        self.norm = dataset.Normalizer.from_dict(ck["normalizer"])
        self.window = ck["window"]
        n_feat = len(ck["normalizer"]["x_mean"])
        if self.kind == "corrector":
            self.model = build_corrector(ck["corrector"], n_feat, config.HORIZON_SCANS, self.window, **ck["model_kwargs"])
            self.model.load_state_dict(ck["state_dict"])
            self.model.to(device).eval()
        else:
            bc = hyp.BaseConfig(ck["corrector"], self.window, True, ck["hp"], {})
            spec = frozenset(ck["spec"])
            nets = {}
            for role, state in ck["states"].items():
                net = hyp._builders(bc, ck["hp"], n_feat, spec)[role]()
                net.load_state_dict(state)
                nets[role] = net
            self.spec_nets = hyp.SpecNets(spec, nets, self.norm, n_feat, self.window)

    def correction(self, x: np.ndarray) -> np.ndarray:
        xn = self.norm.x(x)
        if self.kind == "corrector":
            return self.norm.y_inv(predict(self.model, xn, self.device))
        return self.spec_nets.predict(xn, self.device)[0]


class FinalPredictor:
    """IMM случая + итоговая сеть (или среднее поправок seed при ensemble=True) по пакету bundle_dir."""

    def __init__(self, bundle_dir: Path, device: torch.device, ensemble: bool = False):
        cfg = json.loads((Path(bundle_dir) / "config.json").read_text(encoding="utf-8"))
        self.cfg, self.device = cfg, device
        self.override = cfg["override"]
        self.window = cfg["window"]
        self.case = cfg["case"]
        files = [m["file"] for m in cfg["members"]]
        self.members = [_Member(Path(bundle_dir) / f, device) for f in (files if ensemble else files[:1])]
        label = cfg["label"]
        self.name = (label if label.startswith("IMM") else f"IMM+{label}") + (f" (ансамбль {len(self.members)} seed)" if ensemble else "")

    @property
    def sensor_override(self) -> dict:
        """Подмены модели РЛС (доплер и его источник скорости) для моделирования отметок из лога."""
        keys = ("DOPPLER_IN_IMM", "DOPPLER_IN_NET", "DOPPLER_TRUTH")
        return {k: self.override[k] for k in keys if k in self.override}

    def run(self, rt) -> dict:
        """Прогнозы (K, H, 3): imm - IMM, hyb - гибрид (nan, пока окно не набралось)."""
        with config_override(self.override):
            run = IMM().run(rt)
            w = build_windows(rt, run, window=self.window, with_target=False)
        hyb = np.full_like(run.forecast, np.nan)
        if len(w["k"]):
            corr = np.mean([m.correction(w["x"]) for m in self.members], axis=0)
            hyb[w["k"]] = apply_correction(run.forecast, w["k"], w["psi"], corr)[w["k"]]
        return {"imm": run.forecast, "hyb": hyb, "ks": w["k"]}
