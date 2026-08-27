"""Звук из видео: извлечение и подготовка к распознаванию.

Whisper ждёт моно 16 кГц. Отдать ему стереодорожку с записи совещания —
значит заставить его пересчитывать её самому на каждом запуске и потерять
время там, где его и так не хватает.

Работа делается ffmpeg, а не библиотекой: он уже стоит у всех, кто имеет дело
с видео, читает любой контейнер и не тянет за собой зависимостей в питон.
Вызов вынесен за параметр ``run`` — так подготовку можно проверить, не
раскодировав ни одного файла.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from loguru import logger

#: Частота дискретизации, на которой обучен Whisper. Другая означает лишний
#: пересчёт внутри модели.
SAMPLE_RATE = 16_000

#: Один канал: голоса совещания в стерео не разнесены, вторая дорожка только
#: удваивает объём.
CHANNELS = 1


def parse_time(value: str | float | None) -> float | None:
    """Разбирает «1:05:30», «65:30» и «3930» в секунды.

    Человек указывает место в записи так, как видит его в плеере, и требовать
    от него секунд — значит заставить считать в уме.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)

    parts = str(value).strip().split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise AudioError(f"Не разобрать время: {value!r}. Ожидается 1:05:30 или 65:30") from exc

    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


class AudioError(RuntimeError):
    """Звук не удалось подготовить."""


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True)


def ffmpeg_available(which: Callable[[str], str | None] = shutil.which) -> bool:
    return which("ffmpeg") is not None


def extract_audio(
    source: str | Path,
    target: str | Path | None = None,
    *,
    overwrite: bool = False,
    normalize: bool = True,
    start: float | None = None,
    duration: float | None = None,
    run: Runner = _run,
    which: Callable[[str], str | None] = shutil.which,
) -> Path:
    """Извлекает звук из видео в wav, пригодный для распознавания.

    :param target: куда положить; по умолчанию рядом с исходником, с тем же
        именем и расширением ``.wav``.
    :param overwrite: пересоздавать ли готовый файл. По умолчанию нет:
        раскодирование часовой записи занимает минуты, и повторять его при
        каждом запуске незачем — но это же значит, что подменённое видео с
        прежним именем останется незамеченным.
    :param start: с какой секунды записи брать звук. Нужно чаще, чем
        кажется: на совещании с десятками подключений первый час уходит на
        перекличку, и распознавать его — час работы видеокарты впустую.
    :param duration: сколько секунд взять, считая от ``start``.
    :param normalize: выравнивать ли громкость. На записях совещаний разброс
        огромный: председатель у микрофона и участник в другом конце зала
        отличаются на десятки децибел, и тихого распознавание не слышит.
    """
    source = Path(source)
    if not source.exists():
        raise AudioError(f"Файл не найден: {source}")

    target = Path(target) if target else source.with_suffix(".wav")
    if target.exists() and not overwrite:
        logger.info("Звук уже извлечён: {}", target)
        return target

    if not ffmpeg_available(which):
        raise AudioError(
            "ffmpeg не найден. Он извлекает звук из видео; поставьте его "
            "(macOS: brew install ffmpeg, Ubuntu: apt install ffmpeg) "
            "и повторите."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y"]
    if start:
        # Перед -i, а не после: так ffmpeg перематывает по ключевым кадрам и
        # не раскодирует пропускаемый час.
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(source)]
    if duration:
        command += ["-t", f"{duration:.3f}"]
    command += [
        "-vn",                       # видеодорожка не нужна
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
    ]
    if normalize:
        # loudnorm приводит запись к общему уровню громкости. Одного прохода
        # достаточно: точная двухпроходная нормализация здесь роскошь, речь
        # распознаётся одинаково.
        command += ["-af", "loudnorm"]
    command.append(str(target))

    logger.info("Извлекаю звук: {} -> {}", source.name, target.name)
    result = run(command)
    if result.returncode != 0:
        raise AudioError(
            f"ffmpeg не смог обработать {source.name}: "
            f"{_tail(getattr(result, 'stderr', ''))}"
        )
    if not target.exists():
        raise AudioError(f"ffmpeg отработал, но файла нет: {target}")

    return target


def _tail(text: str, lines: int = 3) -> str:
    """Последние строки вывода ffmpeg.

    Он пишет в stderr десятки строк о кодеках, а причина ошибки всегда в
    конце — показывать всё значит спрятать её в шуме.
    """
    parts = [line for line in (text or "").strip().splitlines() if line.strip()]
    return " / ".join(parts[-lines:]) if parts else "без объяснений"
