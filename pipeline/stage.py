"""
Каркас этапа версии 2: разбор аргументов, журнал, сводка и перебор случаев доплера.

Скрипт этапа определяет функцию run_case(ctx) и вызывает run_stage(...). Запуск:
  python run_stageN.py [--case nodop|dop_traj|dop_tele|all] [--smoke] [свои ключи этапа]
Этап по каждому случаю пишет results/<прогон>/stageN/<случай>/summary.txt; run_all.py считает этап
готовым, когда есть сводки всех случаев. Повторный запуск продолжает: обученные модели берутся из
хранилища (training/store.py), попытки подбора - из базы Optuna.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

import config
from common import lock
from common.cases import CASES, CASE_TITLES
from common.logging_setup import setup_logging
from eval.checks import StageSummary
from pipeline import data as pdata
from pipeline import settings
from pipeline.casedata import CaseData
from pipeline.paths import RunPaths
from pipeline.registry import Registry
from training.trainer import pick_device


@dataclass
class Ctx:
    stage: str
    case: str
    paths: RunPaths
    st: settings.Settings
    summary: StageSummary
    log: object
    device: torch.device
    args: argparse.Namespace
    sets: dict
    registry: Registry = None
    t0: float = field(default_factory=time.time)

    @property
    def smoke(self) -> bool:
        return self.st.smoke

    @property
    def out_dir(self) -> Path:
        return self.paths.results(self.stage, self.case)

    @property
    def art_dir(self) -> Path:
        return self.paths.artifacts(self.stage, self.case)

    @property
    def cache_dir(self) -> Path:
        return self.paths.cache(self.stage, self.case)

    @property
    def store(self):
        return self.paths.store

    def imm_state(self) -> dict:
        """Результат этапа 1 для случая: {"params": параметры IMM, "sigma_vr": доверие к доплеру или None}."""
        state = self.paths.load_state(f"imm_{self.case}")
        if state is None:
            raise FileNotFoundError(f"Нет подобранного IMM случая {self.case}: сначала run_stage1.py")
        return state

    def casedata(self, window: int = config.WINDOW_SCANS, net_input: bool = True, **kw) -> CaseData:
        s = self.imm_state()
        return CaseData(self.paths, self.sets, self.case, imm=s["params"], sigma_vr=s["sigma_vr"],
                        net_input=net_input, window=window, **kw)

    def state(self, name: str, default=None):
        return self.paths.load_state(f"{name}_{self.case}", default)

    def save_state(self, name: str, obj) -> None:
        self.paths.save_state(f"{name}_{self.case}", obj)


def run_stage(stage: str, title: str, run_case, needs_gpu: bool = False, add_args=None, needs_sets: bool = True,
              doc: str = "") -> int:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--case", default="all", choices=("all",) + CASES, help="случай доплера")
    parser.add_argument("--smoke", action="store_true", help="смоук: подвыборка и крошечные бюджеты, не результат")
    if add_args:
        add_args(parser)
    args = parser.parse_args()
    st = settings.get(args.smoke)
    paths = RunPaths(args.smoke)
    tag = "_smoke" if args.smoke else ""
    device = pick_device()
    cases = CASES if args.case == "all" else (args.case,)
    sets = None
    code = 0
    for case in cases:
        held = lock.acquire(paths.cache("state") / f".lock_{stage}_{case}")
        if held is None:
            print(f"[отказ] {stage} {case} уже выполняется другим процессом ({config.RUN_NAME}); второй запуск испортил бы общие "
                  f"файлы", flush=True)
            return 3
        log = setup_logging(f"{stage}_{case}{tag}")
        log.info("Прогон %s, случай: %s (%s), %s", config.RUN_NAME, case, CASE_TITLES[case],
                 "СМОУК" if args.smoke else "полный запуск")
        log.info("Устройство: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        summary = StageSummary(f"{title} - {CASE_TITLES[case]}" + (" (СМОУК: не результат)" if args.smoke else ""))
        out = paths.results(stage, case)
        if needs_gpu and device.type != "cuda" and not args.smoke:
            log.error("GPU нет. Обучение этого этапа - только на домашнем ПК; здесь запускать с --smoke")
            return 2
        if needs_sets and sets is None:
            sets = pdata.load_sets(paths, st)
        ctx = Ctx(stage, case, paths, st, summary, log, device, args, sets, Registry(paths, case))
        if sets is not None and case == cases[0]:
            summary.section("Данные", pdata.describe(sets))
        ok = run_case(ctx)
        summary.section("Время", [f"{(time.time() - ctx.t0) / 60:.1f} мин на {device}"])
        summary.write(out / "summary.txt")
        code |= 0 if (ok is not False and summary.all_passed) else 1
    return code


def main_guard(fn) -> None:
    sys.exit(fn())
