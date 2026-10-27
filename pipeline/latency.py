"""
Время отклика метода (версия 2): миллисекунд на один обзор при прогнозе по одной трассе, то есть шаг IMM
плюс прямой проход сети по окну (батч 1). Процессор и видеокарта замеряются отдельно; числа с CPU за
числа с GPU не выдаются (AGENTS.md). Шаг IMM считается по прогону одной длинной трассы набора.
"""

import time

import numpy as np
import torch

import config
from common.cases import config_override
from eval.latency import net_ms
from filters.imm import IMM
from pipeline.casedata import CaseData


def imm_ms_per_scan(cd: CaseData, name: str = "val") -> float:
    """Время шага IMM в случае cd (с доплером дороже), мс на обзор, процессор."""
    rt = max(cd.eval(name)[1], key=lambda r: len(r.t))
    with config_override(cd.override()):
        imm = IMM()
        t0 = time.perf_counter()
        imm.run(rt)
        return (time.perf_counter() - t0) / len(rt.t) * 1e3


def response_times(imm_ms: float, model: torch.nn.Module = None, n_features: int = 0,
                   window: int = config.WINDOW_SCANS) -> dict:
    """{"imm_ms", "net_cpu_ms", "net_gpu_ms" (если есть видеокарта), "total_cpu_ms", "total_gpu_ms"}."""
    out = {"imm_ms": imm_ms}
    if model is None:
        out["total_cpu_ms"] = imm_ms
        return out
    out["net_cpu_ms"] = net_ms(model, n_features, window, torch.device("cpu"))
    out["total_cpu_ms"] = imm_ms + out["net_cpu_ms"]
    if torch.cuda.is_available():
        out["net_gpu_ms"] = net_ms(model, n_features, window, torch.device("cuda"))
        out["total_gpu_ms"] = imm_ms + out["net_gpu_ms"]
    return out


def n_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
