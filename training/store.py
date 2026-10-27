"""
Хранилище обученных моделей версии 2 (решение пользователя 27.10.2026): эксперименты
воспроизводятся без повторного обучения, прерванный запуск продолжается без потери.

Одна модель = одна папка model_store/<прогон>/<этап>/<случай>/<подпись>_s<seed>/:
  model.pt  - веса и всё, что нужно для прогноза (нормализатор, настройки сети);
  meta.json - метрики обучения (история, время, число параметров), результаты проверок;
  ckpt.pt   - контрольная точка прерванного обучения (удаляется после сохранения модели).
Модель считается готовой, когда есть meta.json (он пишется последним). Смоук-модели лежат
отдельно (_smoke/) и в основном хранилище не появляются.
"""

import json
import shutil
from pathlib import Path

import torch

import config


def _slug(label: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label.replace("+", "_"))
    return "_".join(part for part in keep.split("_") if part)


class ModelStore:
    def __init__(self, root: Path = None, smoke: bool = False):
        root = Path(root) if root is not None else config.STORE_DIR
        self.root = root / "_smoke" if smoke else root

    def folder(self, stage: str, case: str, label: str, seed: int) -> Path:
        return self.root / stage / case / f"{_slug(label)}_s{seed}"

    def has(self, stage: str, case: str, label: str, seed: int) -> bool:
        return (self.folder(stage, case, label, seed) / "meta.json").exists()

    def checkpoint(self, stage: str, case: str, label: str, seed: int) -> Path:
        """Путь контрольной точки прерванного обучения (папка создаётся)."""
        folder = self.folder(stage, case, label, seed)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "ckpt.pt"

    def save(self, stage: str, case: str, label: str, seed: int, payload: dict, meta: dict) -> Path:
        folder = self.folder(stage, case, label, seed)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(payload, folder / "model.pt")
        (folder / "ckpt.pt").unlink(missing_ok=True)
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
        return folder

    def load(self, stage: str, case: str, label: str, seed: int) -> tuple:
        """(payload, meta) готовой модели."""
        folder = self.folder(stage, case, label, seed)
        payload = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return payload, meta

    def remove(self, stage: str, case: str, label: str, seed: int) -> None:
        shutil.rmtree(self.folder(stage, case, label, seed), ignore_errors=True)
