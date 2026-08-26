"""Весь путь одним вызовом: запись — стенограмма — поручения — протокол.

Модуль только расставляет шаги по порядку и передаёт между ними данные.
Вся содержательная работа живёт в соседних модулях, и это не педантизм:
распознавание требует видеокарты, извлечение поручений — запущенной модели,
а сборка протокола не требует ничего. Разделение позволяет остановиться на
любом шаге и продолжить на другой машине.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger

from .audio import extract_audio
from .blocks import Transcript, blocks_from_segments, consolidate, rename_speakers
from .chunking import split_into_chunks
from .config import Settings
from .llm import LLMClient
from .protocol import Protocol, build_protocol
from .tasks import extract_tasks
from .transcribe import Backend, recognize

#: Расширения, которые считаем видео: из них сперва извлекается звук.
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class Meeting:
    """Что известно о совещании до расчёта.

    Ничего из этого распознавание не даёт: дату и место знает только тот, кто
    запускает расчёт. Поэтому поля пустые по умолчанию — выдуманная дата в
    протоколе хуже пустой строки.
    """

    title: str = "Протокол совещания"
    date: str = ""
    place: str = ""
    chair: str = ""
    #: Имена по меткам говорящих: ``{"SPEAKER_00": "Орлов В.П."}``.
    names: dict[str, str] | None = None
    #: Полный список участников. Если не задан, берётся из стенограммы —
    #: но тогда в нём не будет тех, кто присутствовал и промолчал.
    attendees: Sequence[str] | None = None


def transcribe_meeting(
    source: str | Path,
    settings: Settings | None = None,
    *,
    backend: Backend | None = None,
    work_dir: str | Path | None = None,
    diarize: bool = True,
) -> Transcript:
    """Из записи — стенограмма. Самый долгий шаг, и единственный с видеокартой."""
    settings = settings or Settings()
    source = Path(source)
    work_dir = Path(work_dir) if work_dir else source.parent

    audio = source
    if source.suffix.lower() in VIDEO_SUFFIXES:
        audio = extract_audio(source, work_dir / f"{source.stem}.wav")

    recognition = recognize(
        audio, settings, backend=backend, cache_dir=work_dir, diarize=diarize
    )
    blocks = consolidate(blocks_from_segments(recognition.segments))
    logger.info(
        "Стенограмма: {} реплик, говорящих {}", len(blocks), len(Transcript(blocks).speakers)
    )
    return Transcript(blocks)


def protocol_from_transcript(
    transcript: Transcript,
    settings: Settings | None = None,
    *,
    meeting: Meeting | None = None,
    client: LLMClient | None = None,
) -> Protocol:
    """Из стенограммы — протокол с поручениями.

    Видеокарта здесь не нужна: шаг работает и на ноутбуке, лишь бы рядом
    отвечала модель. Именно поэтому стенограмму стоит сохранять — пересобрать
    протокол с другими именами участников можно, не распознавая заново.
    """
    settings = settings or Settings()
    meeting = meeting or Meeting()
    client = client or LLMClient(settings)

    blocks = transcript.blocks
    if meeting.names:
        blocks = rename_speakers(blocks, meeting.names)
    named = Transcript(blocks)

    chunks = split_into_chunks(
        blocks,
        max_tokens=settings.chunk_max_tokens,
        overlap_blocks=settings.chunk_overlap_blocks,
    )
    logger.info("Стенограмма разбита на {} фрагментов", len(chunks))

    tasks = extract_tasks(chunks, client)
    logger.info("Найдено поручений: {}", len(tasks))

    return build_protocol(
        tasks,
        named,
        title=meeting.title,
        date=meeting.date,
        place=meeting.place,
        chair=meeting.chair,
        attendees=meeting.attendees,
    )


def process(
    source: str | Path,
    settings: Settings | None = None,
    *,
    meeting: Meeting | None = None,
    backend: Backend | None = None,
    client: LLMClient | None = None,
    work_dir: str | Path | None = None,
) -> Protocol:
    """Полный путь: запись на входе, протокол на выходе."""
    settings = settings or Settings()
    transcript = transcribe_meeting(source, settings, backend=backend, work_dir=work_dir)
    return protocol_from_transcript(transcript, settings, meeting=meeting, client=client)


def save(protocol: Protocol, out_dir: str | Path, *, stem: str = "protocol") -> dict[str, Path]:
    """Сохраняет протокол и таблицу поручений.

    Два файла, а не один: протокол читают, а таблицу разбирают по
    исполнителям и ставят на контроль.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    document = out_dir / f"{stem}.md"
    document.write_text(protocol.as_markdown(with_transcript=True), encoding="utf-8")

    table = out_dir / f"{stem}_tasks.csv"
    # BOM: без него Excel в русской локали открывает таблицу кракозябрами.
    table.write_text(protocol.tasks_csv(), encoding="utf-8-sig")

    logger.info("Сохранено: {} и {}", document.name, table.name)
    return {"protocol": document, "tasks": table}
