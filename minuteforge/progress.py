"""Сообщения о ходе работы.

Общее для распознавания и для сборки протокола: оба этапа идут долго —
первый десятки минут, второй по десятку секунд на фрагмент, — и оба должны
показывать, что живы. Механизм один, чтобы интерфейс и командная строка
рисовали их одинаково.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from loguru import logger


@dataclass
class Step:
    """Событие о ходе работы."""

    name: str
    #: Человеческое название — то, что видит человек.
    title: str
    #: Шаг закончен. До этого приходит событие о начале.
    done: bool = False
    #: Шаг не считался, а взят из сохранённого.
    reused: bool = False
    elapsed_s: float = 0.0
    #: Доля выполненного, от 0 до 1.
    share: float = 0.0


#: Кому сообщать о ходе работы.
Progress = Callable[[Step], None]


def report(progress: Progress | None, step: Step) -> None:
    """Сообщает о шаге, не роняя расчёт из-за ошибки в показе прогресса.

    Сорок минут работы не должны пропасть из-за того, что не нарисовалась
    полоска.
    """
    if progress is None:
        return
    try:
        progress(step)
    except Exception as exc:  # рисование не должно ломать работу
        logger.warning("Ошибка при показе прогресса: {}", exc)
