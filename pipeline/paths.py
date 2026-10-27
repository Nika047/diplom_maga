"""
Пути и состояние прогона (версия 2). Все каталоги - от config.*_DIR (у каждого из четырёх прогонов свои),
смоук пишет в подпапку _smoke и не трогает полный запуск.

Состояние между этапами - JSON в cache/<прогон>/state/: подобранные параметры IMM и корректоров, основа,
выжившие гипотезы. Этап читает состояние предыдущих и пишет своё; повторный запуск того же этапа
перезаписывает только то, что пересчитал.
"""

import json
from pathlib import Path

import config
from training.store import ModelStore


class RunPaths:
    def __init__(self, smoke: bool = False):
        self.smoke = smoke

    def _root(self, base: Path) -> Path:
        return base / "_smoke" if self.smoke else base

    def results(self, stage: str, case: str = None) -> Path:
        path = self._root(config.RESULTS_DIR) / stage / (case or "")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache(self, stage: str, case: str = None) -> Path:
        path = self._root(config.CACHE_DIR) / stage / (case or "")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts(self, stage: str, case: str = None) -> Path:
        path = self._root(config.ARTIFACTS_DIR) / stage / (case or "")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def store(self) -> ModelStore:
        return ModelStore(smoke=self.smoke)

    @property
    def stage0(self) -> Path:
        """Кэш этапа 0 общий для смоука и полного запуска."""
        return config.CACHE_DIR / "stage0"

    def _state(self, name: str) -> Path:
        return self.cache("state") / f"{name}.json"

    def load_state(self, name: str, default=None):
        path = self._state(name)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def save_state(self, name: str, obj) -> None:
        path = self._state(name)
        tmp = path.with_suffix(".tmp")      # запись целиком: прерывание посреди записи не портит прежнее состояние
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        tmp.replace(path)
