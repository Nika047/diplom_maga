"""
Проверки этапа со статусом PASS / FAIL и запись сводки results/stageN/summary.txt.

Каждая проверка сразу пишется в журнал; сводка содержит разделы со статистикой
и итоговый вердикт.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common.logging_setup import get_logger


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class StageSummary:
    stage: str
    checks: list = field(default_factory=list)
    sections: list = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Записывает проверку; возвращает её результат."""
        passed = bool(passed)
        self.checks.append(Check(name, passed, detail))
        status = "PASS" if passed else "FAIL"
        log = get_logger()
        (log.info if passed else log.error)("[%s] %s: %s", status, name, detail)
        return passed

    def section(self, title: str, lines: list) -> None:
        """Добавляет в сводку раздел со статистикой и дублирует его в журнал."""
        self.sections.append((title, list(lines)))
        log = get_logger()
        log.info("--- %s ---", title)
        for line in lines:
            log.info("%s", line)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n_fail = sum(not c.passed for c in self.checks)
        verdict = "ВСЕ ПРОВЕРКИ PASS" if n_fail == 0 else f"FAIL: {n_fail} из {len(self.checks)}"
        lines = [f"Сводка: {self.stage}",
                 f"Время: {datetime.now():%Y-%m-%d %H:%M:%S}",
                 f"Итог: {verdict}", ""]
        for title, body in self.sections:
            lines.append(f"== {title} ==")
            lines.extend(body)
            lines.append("")
        lines.append("== Проверки ==")
        for c in self.checks:
            lines.append(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        get_logger().info("Сводка: %s (%s)", path, verdict)
