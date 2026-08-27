"""Весь путь одним вызовом: запись — стенограмма — поручения — протокол.

Модуль только расставляет шаги по порядку и передаёт между ними данные.
Вся содержательная работа живёт в соседних модулях, и это не педантизм:
распознавание требует видеокарты, извлечение поручений — запущенной модели,
а сборка протокола не требует ничего. Разделение позволяет остановиться на
любом шаге и продолжить на другой машине.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger

from .audio import extract_audio, parse_time
from .blocks import (
    Block,
    Transcript,
    blocks_from_segments,
    consolidate,
    drop_soundcheck,
    rename_speakers,
)
from .chunking import split_into_chunks
from .config import Settings
from .llm import LLMClient
from .people import Person
from .protocol import Protocol, build_protocol
from .tasks import extract_tasks
from .transcribe import (
    STEP_TITLES,
    RecognitionError,
    Backend,
    Progress,
    Step,
    recognize,
    report,
)

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
    #: Реквизиты протокола как документа. Распознавание их не даёт: их знает
    #: тот, кто ведёт делопроизводство.
    secretary: str = ""
    number: str = ""
    #: Имена по меткам говорящих: ``{"SPEAKER_00": "Орлов В.П."}``.
    names: dict[str, str] | None = None
    #: Полный список участников. Если не задан, берётся из стенограммы —
    #: но тогда в нём не будет тех, кто присутствовал и промолчал.
    attendees: Sequence[str] | None = None
    #: Список участников с должностями. Не заменяет число говорящих для
    #: диаризации: в списке все приглашённые, а голосов в записи обычно
    #: втрое меньше.
    people: Sequence[Person] | None = None


def cache_dir_for(
    source: Path,
    work_dir: Path,
    *,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Своя папка промежуточных шагов для каждой записи.

    Шаги распознавания сохраняются, чтобы сбой на последнем не стоил сорока
    минут работы. Но складывать их под одними и теми же именами в общую
    папку нельзя: следующая запись подхватит чужую расшифровку и выдаст её
    за свою — молча, за секунду и с полной уверенностью.

    Отпечаток берётся из имени, размера и времени изменения файла. Читать
    гигабайты ради хеша незачем: подменить запись, сохранив всё три
    признака, можно только нарочно.
    """
    stat = source.stat()
    # Отрезок — часть приметы: распознав первый час, нельзя выдать его за
    # второй только потому, что файл тот же.
    mark = f"{source.name}|{stat.st_size}|{int(stat.st_mtime)}|{start or 0}|{end or 0}"
    short = hashlib.sha1(mark.encode("utf-8")).hexdigest()[:8]
    return work_dir / f"{source.stem}-{short}"


def transcribe_meeting(
    source: str | Path,
    settings: Settings | None = None,
    *,
    backend: Backend | None = None,
    work_dir: str | Path | None = None,
    diarize: bool = True,
    progress: Progress | None = None,
    fresh: bool = False,
    start: str | float | None = None,
    end: str | float | None = None,
) -> Transcript:
    """Из записи — стенограмма. Самый долгий шаг, и единственный с видеокартой.

    :param progress: куда сообщать о ходе работы: часовая запись считается
        десятки минут, и человеку нужно видеть, что она считается.
    :param fresh: считать заново, не заглядывая в сохранённые шаги.
    :param start, end: какой кусок записи распознавать — «1:05:30» или
        секундами. Нужно чаще, чем кажется: на совещании с десятками
        подключений первый час уходит на перекличку, и распознавать его —
        час работы видеокарты впустую. Время в стенограмме при этом остаётся
        от начала записи, а не от начала куска: иначе по нему не найти место
        в исходном видео.
    """
    settings = settings or Settings()
    source = Path(source)
    if not source.exists():
        raise RecognitionError(f"Файл не найден: {source}")

    from_second = parse_time(start)
    to_second = parse_time(end)
    if from_second is not None and to_second is not None and to_second <= from_second:
        raise RecognitionError(
            f"Конец отрезка ({end}) не позже начала ({start}) — распознавать нечего"
        )

    work_dir = Path(work_dir) if work_dir else source.parent
    cache = cache_dir_for(source, work_dir, start=from_second, end=to_second)
    if fresh and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)

    audio = source
    if source.suffix.lower() in VIDEO_SUFFIXES:
        title = STEP_TITLES["audio"]
        report(progress, Step(name="audio", title=title, share=0.0))
        audio = extract_audio(
            source, cache / f"{source.stem}.wav",
            start=from_second,
            duration=(to_second - (from_second or 0)) if to_second else None,
        )
        report(progress, Step(name="audio", title=title, done=True, share=0.05))

    recognition = recognize(
        audio, settings, backend=backend, cache_dir=cache,
        diarize=diarize, progress=progress,
    )
    blocks = consolidate(blocks_from_segments(recognition.segments))
    if from_second:
        # Время сдвигается обратно к началу записи: по нему возвращаются к
        # спорному месту в исходном видео, и «12:30» должно значить 12:30
        # видео, а не отрезка.
        blocks = [
            Block(
                b.speaker, b.text,
                None if b.start is None else b.start + from_second,
                None if b.end is None else b.end + from_second,
            )
            for b in blocks
        ]
    logger.info(
        "Стенограмма: {} реплик, говорящих {}", len(blocks), len(Transcript(blocks).speakers)
    )
    transcript = Transcript(blocks)
    # Уступки прикладываются к стенограмме: расшифровка, сделанная моделью
    # поменьше, — другая расшифровка, и знать об этом нужно и в интерфейсе,
    # и в командной строке.
    transcript.notes = list(recognition.fallbacks)
    return transcript


def protocol_from_transcript(
    transcript: Transcript,
    settings: Settings | None = None,
    *,
    meeting: Meeting | None = None,
    client: LLMClient | None = None,
    progress: Progress | None = None,
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

    # Перекличка отбрасывается только здесь, на входе в модель: стенограмма
    # остаётся полной. Расшифровка должна быть точной — что прозвучало, то и
    # записано; отбирать, чему место в документе, а чему нет, дело того, кто
    # ведёт протокол, а не инструмента.
    for_model, skipped = drop_soundcheck(blocks) if settings.drop_soundcheck else (blocks, 0)
    if skipped:
        logger.info("Перекличка в начале записи пропущена: {} реплик", skipped)

    chunks = split_into_chunks(
        for_model,
        max_tokens=settings.chunk_budget,
        overlap_blocks=settings.chunk_overlap_blocks,
    )
    logger.info("Стенограмма разбита на {} фрагментов", len(chunks))

    answers: list[str] = []
    tasks = extract_tasks(
        chunks, client, progress=progress, answers=answers,
        corpus=named.as_text(),
    )
    logger.info("Найдено поручений: {}", len(tasks))

    return build_protocol(
        tasks,
        named,
        title=meeting.title,
        date=meeting.date,
        place=meeting.place,
        chair=meeting.chair,
        secretary=meeting.secretary,
        number=meeting.number,
        attendees=meeting.attendees,
        people=meeting.people,
        answers=answers,
    )


def process(
    source: str | Path,
    settings: Settings | None = None,
    *,
    meeting: Meeting | None = None,
    backend: Backend | None = None,
    client: LLMClient | None = None,
    work_dir: str | Path | None = None,
    progress: Progress | None = None,
) -> Protocol:
    """Полный путь: запись на входе, протокол на выходе."""
    settings = settings or Settings()
    transcript = transcribe_meeting(
        source, settings, backend=backend, work_dir=work_dir, progress=progress
    )
    return protocol_from_transcript(
        transcript, settings, meeting=meeting, client=client, progress=progress
    )


def save_transcript(
    transcript: Transcript,
    out_dir: str | Path,
    *,
    stem: str = "transcript",
) -> dict[str, Path]:
    """Сохраняет стенограмму текстом и в json.

    Два формата, потому что читают их разные люди. Текст уходит в другое
    подразделение и читается глазами. json нужен, чтобы вернуться к этой же
    записи и пересобрать протокол, не распознавая заново.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = out_dir / f"{stem}.txt"
    text.write_text(transcript.as_text(with_time=True), encoding="utf-8")

    data = out_dir / f"{stem}.json"
    data.write_text(
        json.dumps(
            [
                {"speaker": b.speaker, "text": b.text, "start": b.start, "end": b.end}
                for b in transcript.blocks
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Стенограмма сохранена: {} и {}", text.name, data.name)
    return {"text": text, "json": data}


def save(
    protocol: Protocol,
    out_dir: str | Path,
    *,
    stem: str = "протокол",
    template: str | Path | None = None,
    with_transcript: bool = True,
) -> dict[str, Path]:
    """Сохраняет протокол, стенограмму и таблицу поручений.

    Три файла, а не один, и это не мелочь. Протокол — документ с заведённой
    формой, его подписывают и рассылают. Стенограмма — сырьё, по которому
    проверяют спорное место. Подшитая внутрь протокола, она превращает
    документ в расшифровку на сорок страниц, которую никто не читает.

    :param template: форма протокола. Задана — документ собирается по ней,
        нет — по встроенной разметке. Своя форма есть в каждой организации,
        и снаружи её не угадать.
    :param with_transcript: сохранять ли стенограмму. Если её уже сохранили
        на шаге распознавания, второй раз не нужно: одна и та же расшифровка
        под двумя именами в одной папке только сбивает с толку.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if template is not None:
        text = Path(template).read_text(encoding="utf-8-sig")
        document_text = protocol.render(text)
        suffix = Path(template).suffix or ".md"
    else:
        document_text = protocol.as_markdown()
        suffix = ".md"

    document = out_dir / f"{stem}{suffix}"
    document.write_text(document_text, encoding="utf-8")

    table = out_dir / f"{stem}_поручения.csv"
    # BOM: без него Excel в русской локали открывает таблицу кракозябрами.
    table.write_text(protocol.tasks_csv(), encoding="utf-8-sig")

    saved = {"protocol": document, "tasks": table}

    if with_transcript and protocol.transcript is not None:
        transcript = out_dir / f"{stem}_стенограмма.txt"
        transcript.write_text(
            protocol.transcript.as_text(with_time=True), encoding="utf-8"
        )
        saved["transcript"] = transcript

    logger.info("Сохранено: {}", ", ".join(path.name for path in saved.values()))
    return saved
