"""
Единая настройка журнала: каждое сообщение идёт в консоль и в файл
log/<запуск>_<ГГГГММДД-ЧЧММСС>.log (UTF-8).

Необработанные исключения и предупреждения тоже попадают в файл.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import config

LOGGER_NAME = "diplom"
_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(run_name: str, log_dir: Path = config.LOG_DIR) -> logging.Logger:
    """Настраивает журнал запуска и возвращает его. Повторный вызов заменяет обработчики."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{run_name}_{stamp}.log"

    # Консоль Windows и Git Bash могут не уметь UTF-8 - не падать на символах
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers = [console, file_handler]

    def _log_uncaught(exc_type, exc, tb):
        logger.critical("Необработанное исключение", exc_info=(exc_type, exc, tb))

    sys.excepthook = _log_uncaught
    logger.info("Журнал запуска: %s", log_path)
    return logger


def get_logger() -> logging.Logger:
    """Журнал для модулей: сообщения уходят в обработчики, настроенные setup_logging."""
    return logging.getLogger(LOGGER_NAME)
