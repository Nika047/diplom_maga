"""
Данные одного случая доплера для этапов версии 2: окна и прогоны IMM под параметрами случая.

Случай задаёт флаги доплера (common/cases.py), параметры IMM (подобраны на этапе 1 в этом случае) и,
в случаях с доплером, доверие фильтра к доплеру FILTER_SIGMA_VR. Всё, что строится по трассам (прогон
РЛС и IMM, окна), выполняется внутри override(): параметры читаются из config в момент вызова.

Обзоры оценки (ks) одни и те же у всех случаев - по случаю без доплера: сравнение «доплер против
его отсутствия» идёт на одинаковых парах (k, h).
"""

import pickle

import numpy as np

import config
from common.cases import CASES, config_override, overrides
from eval.compare import mask_forecast
from filters.imm import IMM
from pipeline.paths import RunPaths
from training import dataset


class CaseData:
    def __init__(self, paths: RunPaths, sets: dict, case: str, imm: dict = None, sigma_vr: float = None,
                 net_input: bool = True, window: int = config.WINDOW_SCANS):
        if case not in CASES:
            raise ValueError(case)
        self.paths, self.sets, self.case = paths, sets, case
        self.imm, self.sigma_vr, self.net_input, self.window = imm or {}, sigma_vr, net_input, window
        self._built = {}
        self._ks = {}

    def variant(self, **kw) -> "CaseData":
        """Тот же случай с другими параметрами (окно, доплер во входе сети, параметры IMM)."""
        args = dict(imm=self.imm, sigma_vr=self.sigma_vr, net_input=self.net_input, window=self.window)
        args.update(kw)
        return CaseData(self.paths, self.sets, self.case, **args)

    @property
    def doppler(self) -> bool:
        return self.case != "nodop"

    def override(self, **extra) -> dict:
        net = self.net_input if self.doppler else False
        return {**self.imm, **overrides(self.case, net, self.sigma_vr), **extra}

    @property
    def cache_dir(self):
        return self.paths.cache("windows", self.case)

    def windows(self, key: str, tracks: list) -> dataset.WindowSet:
        """Окна обучения (кэш на диске): key - имя набора, например "main"."""
        with config_override(self.override()):
            return dataset.load_or_build(f"{key}_w{self.window}_n{int(self.net_input)}", tracks, self.cache_dir,
                                         window=self.window)

    def build(self, tracks: list, run_kwargs: list = None) -> tuple:
        with config_override(self.override()):
            return dataset.build(tracks, window=self.window, run_kwargs=run_kwargs)

    def eval(self, name: str) -> tuple:
        """(окна, трассы РЛС, прогоны IMM) набора оценки; строятся один раз."""
        if name not in self._built:
            self._built[name] = self.build(self.sets[name])
        return self._built[name]

    def ks(self, name: str) -> list:
        """Обзоры оценки набора по трассам - общие для всех случаев (по случаю без доплера)."""
        if name not in self._ks:
            path = self.paths.cache("state") / f"ks_{name}_w{self.window}.pkl"
            if path.exists():
                with open(path, "rb") as fh:
                    self._ks[name] = pickle.load(fh)
            else:
                ref = self if self.case == "nodop" else CaseData(self.paths, self.sets, "nodop", window=self.window)
                ws, rts, _ = ref.eval(name)
                self._ks[name] = [np.array(sorted(ws.k[ws.track == i]), dtype=int) for i in range(len(rts))]
                with open(path, "wb") as fh:
                    pickle.dump(self._ks[name], fh)
        return self._ks[name]

    def mask(self, name: str, forecasts: list) -> list:
        return [mask_forecast(fc, k) for fc, k in zip(forecasts, self.ks(name))]

    def imm_runner(self) -> IMM:
        """Рабочий IMM под параметрами случая (создавать внутри override())."""
        return IMM()
