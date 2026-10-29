"""
Блокировка от двойного запуска (версия 2): два процесса, пишущие в одни и те же хранилище моделей, базу Optuna и
реестр, портят результаты друг другу и искажают замер времени. Замок держится открытым файлом, пока жив процесс; при
его гибели (в том числе taskkill) ОС освобождает замок сама, остатка после остановки нет.
"""

import os
from pathlib import Path


def acquire(path: Path):
    """Файл-замок (держать ссылку до конца работы) или None, если замок уже у другого процесса."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh
