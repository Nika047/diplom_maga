"""
Реестр результатов случая (версия 2): метрики каждого метода по наборам оценки и ошибки по полётам для
бутстрепа. Хранится в cache/<прогон>/state/registry_<случай>.json; этапы дописывают, этап отбора читает.

Метод = строка "<этап>/<подпись>"; на каждый seed своя запись (сети). Необучаемые методы - seed 0.
"""

import numpy as np

from pipeline.paths import RunPaths


class Registry:
    def __init__(self, paths: RunPaths, case: str):
        self.paths, self.case = paths, case
        self.name = f"registry_{case}"
        self.data = paths.load_state(self.name, {})

    def record(self, method: str, seed: int, reports: dict, extra: dict = None) -> None:
        """reports: {набор: отчёт (pipeline/evaluation.report) с ключом "flights"}; extra - прочие числа метода."""
        entry = self.data.setdefault(method, {})
        entry[str(seed)] = {"reports": reports, "extra": extra or {}}
        self.paths.save_state(self.name, self.data)

    def set_report(self, method: str, seed: int, name: str, report: dict, extra: dict = None) -> None:
        """Дописывает отчёт метода по одному набору (запись на диск - flush)."""
        entry = self.data.setdefault(method, {}).setdefault(str(seed), {"reports": {}, "extra": {}})
        entry["reports"][name] = report
        if extra:
            entry["extra"].update(extra)

    def flush(self) -> None:
        self.paths.save_state(self.name, self.data)

    def has(self, method: str, seed: int = None) -> bool:
        return method in self.data and (seed is None or str(seed) in self.data[method])

    def methods(self) -> list:
        return list(self.data)

    def seeds(self, method: str) -> list:
        return sorted(int(s) for s in self.data[method])

    def report(self, method: str, name: str, seed: int = None) -> dict:
        """Отчёт метода на наборе: при seed=None - первый seed."""
        seeds = self.seeds(method)
        return self.data[method][str(seeds[0] if seed is None else seed)]["reports"][name]

    def metric(self, method: str, name: str, key: str = "ade") -> tuple:
        """(среднее, СКО) показателя по seed."""
        vals = np.array([self.data[method][str(s)]["reports"][name][key] for s in self.seeds(method)], float)
        return float(vals.mean()), float(vals.std())

    def flights(self, method: str, name: str) -> tuple:
        """Суммы и числа по полётам, усреднённые по seed (для парного бутстрепа): (ключи, суммы, числа)."""
        parts = [self.data[method][str(s)]["reports"][name]["flights"] for s in self.seeds(method)]
        keys = parts[0]["keys"]
        for p in parts[1:]:
            if p["keys"] != keys:
                raise ValueError(f"Полёты seed метода {method} не совпадают")
        sums = np.mean([p["sums"] for p in parts], axis=0)
        return keys, sums, np.array(parts[0]["counts"], float)

    def extra(self, method: str, key: str, default=None):
        seeds = self.seeds(method)
        return self.data[method][str(seeds[0])]["extra"].get(key, default)
