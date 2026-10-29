"""
Время отклика (версия 2): сколько миллисекунд занимает обработка одного обзора при
прогнозе по одной трассе (шаг IMM плюс сеть). Медиана по повторам, процессор и видеокарта
замеряются отдельно; числа с процессора за числа с GPU не выдавать (AGENTS.md).
"""

import statistics
import time

import torch

import config


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_ms(fn, device: torch.device = torch.device("cpu"), warmup: int = config.LATENCY_WARMUP,
              repeats: int = config.LATENCY_REPEATS) -> float:
    """
    Время одного вызова fn() в миллисекундах: наименьшая из LATENCY_BLOCKS медиан по блокам повторов (с прогревом
    и синхронизацией GPU). На процессоре замер идёт в одном потоке: прогон по одной трассе не распараллеливается.
    Причина: на прогоне base_x1 две сети одной архитектуры дали 0,51 и 0,80 мс на процессоре (замеры по медиане 30
    повторов); влияние числа потоков не проверялось, разброс отсекается наименьшей из медиан блоков.
    """
    threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(1)
    try:
        for _ in range(warmup):
            fn()
        _sync(device)
        blocks = []
        for _ in range(config.LATENCY_BLOCKS):
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                _sync(device)
                times.append((time.perf_counter() - t0) * 1e3)
            blocks.append(statistics.median(times))
        return min(blocks)
    finally:
        torch.set_num_threads(threads)


def net_ms(model: torch.nn.Module, n_features: int, window: int, device: torch.device,
           repeats: int = config.LATENCY_REPEATS) -> float:
    """Прямой проход сети по одному окну (батч 1), мс."""
    model = model.to(device).eval()
    x = torch.zeros(1, window, n_features, device=device)

    def fn():
        with torch.no_grad():
            model(x)

    return median_ms(fn, device, repeats=repeats)
