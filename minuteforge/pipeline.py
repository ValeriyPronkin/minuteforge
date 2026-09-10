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
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from loguru import logger

from .audio import extract_audio, parse_time, recorded_at
from .blocks import (
    Block,
    Transcript,
    blocks_from_segments,
    consolidate,
    drop_soundcheck,
    rename_speakers,
)
from .checks import Suspicion, suspicious
from .chunking import estimate_tokens, split_into_chunks, split_into_windows
from .config import Settings
from . import dates
from .directory import read_directory, at as unit_at
from .journal import Journal
from .vocabulary import read_vocabulary
from .llm import LLMClient, same_model
from .people import Person
from .protocol import Protocol, build_protocol
from .tasks import (
    COLLECTIVE_NAMES,
    asks_for_work,
    extract_tasks,
    use_words,
    most_addressed,
    worth_showing,
)
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


def recorded_when(source: str | Path) -> dates.Recorded | None:
    """Когда состоялось совещание — по самой записи.

    Дату совещания приходится вписывать руками, а она нужна не только в шапке:
    от неё считаются сроки, и без неё «через две недели» так и остаётся словами.
    Спрашивать её у человека, когда сама запись её знает, — лишняя работа и
    лишний повод ошибиться.

    Три источника, по убыванию доверия:

    1. **Имя файла.** Его даёт человек или программа записи, и оно переживает
       копирование: «Совещание 07.09.2026.mp4».
    2. **Метаданные контейнера.** Точный момент начала записи, но его теряет
       перекодирование, а в некоторых контейнерах его нет вовсе.
    3. **Время файла на диске.** Последнее средство, и самое ненадёжное: у
       скопированного файла это может оказаться день копирования.

    Первые два дополняют друг друга: имя чаще знает день, метаданные — час.
    Если они говорят об одном дне, берётся день из имени и час из свойств.

    Ничего не нашлось — ``None``. Дата совещания не то, что можно угадать:
    пусть лучше её впишут руками, чем документ уйдёт с выдуманной.
    """
    source = Path(source)
    named = dates.date_from_name(source.name)
    moment = recorded_at(source) if source.exists() else None

    if named is not None:
        if moment is not None and moment.date() == named.day and not named.clock:
            return replace(
                named,
                clock=moment.strftime("%H:%M"),
                source="из имени файла и свойств записи",
            )
        return named
    if moment is not None:
        return dates.Recorded(moment.date(), moment.strftime("%H:%M"), "из свойств записи")

    try:
        changed = datetime.fromtimestamp(source.stat().st_mtime)
    except OSError:
        return None
    return dates.Recorded(changed.date(), source="по времени файла на диске")


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
    transcript.model = recognition.asr_model
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

    # Свои слова — до первого обращения к правилу: по нему собираются окна,
    # и словарь, применённый позже, на отбор уже не повлияет.
    own_words = read_vocabulary(settings.words_file)
    use_words(own_words.orders, own_words.not_orders)

    blocks = transcript.blocks
    if meeting.names:
        blocks = rename_speakers(blocks, meeting.names)
    named = Transcript(blocks)

    # Перекличка отбрасывается только здесь, на входе в модель: стенограмма
    # остаётся полной. Расшифровка должна быть точной — что прозвучало, то и
    # записано; отбирать, чему место в документе, а чему нет, дело того, кто
    # ведёт протокол, а не инструмента.
    #
    # Реплика, в которой поручение слышно, остаётся, даже если похожа на
    # перекличку: «Слышно. Иванов, подготовьте справку» сказано вперемешку,
    # и терять вторую половину из-за первой нельзя. Одной вежливости для
    # этого мало: «подскажите, пожалуйста, Калмыкию слышно?» — перекличка,
    # и в протоколе она стояла пунктом «Подключить Республику Калмыкию».
    for_model, skipped = (
        drop_soundcheck(blocks, keep=asks_for_work)
        if settings.drop_soundcheck else (blocks, 0)
    )
    if skipped:
        logger.info("Перекличка пропущена: {} реплик из {}", skipped, len(blocks))

    if settings.extract_by_phrase:
        # Модель читает не совещание, а места, где поручение слышно. Кусок в
        # четыре тысячи токенов ей не по силам: поручений в нём полтора
        # десятка, а выписывает она первые несколько.
        chunks = split_into_windows(for_model, directive=worth_showing)
        logger.info(
            "Отобрано окон: {} (в них ~{} токенов из ~{} во всей стенограмме)",
            len(chunks),
            sum(estimate_tokens(c.text) for c in chunks),
            sum(estimate_tokens(f"{b.speaker}: {b.text}") for b in for_model),
        )
    else:
        chunks = split_into_chunks(
            for_model,
            max_tokens=settings.chunk_budget,
            overlap_blocks=settings.chunk_overlap_blocks,
        )
        logger.info("Стенограмма разбита на {} фрагментов", len(chunks))

    warm_up(client, progress)

    answers: list[str] = []
    # Ведущий — тот, к кому обращаются чаще всех. Поручений ему не дают:
    # обращением к нему докладчик открывает свою речь.
    chair = most_addressed(for_model)
    if chair:
        logger.info("Совещание ведёт {} — в исполнители не пойдёт", chair)
    # Справочник того, по кому идёт разбор. Нет файла — графа останется
    # пустой: услышанное название инструмент не берёт, потому что
    # распознавание их коверкает, а неверный адресат хуже пустого.
    units = read_directory(settings.directory_file, label=settings.directory_label)
    if not units:
        logger.warning(
            "Справочник направлений не задан — графа «{}» останется пустой. "
            "Файл указывается настройкой directory_file.",
            settings.directory_label,
        )

    # Выписывать и проверять — разные задачи. Щедрая модель находит
    # больше, строгая реже ошибается; когда это две разные модели, каждая
    # делает своё. Меняются они в памяти один раз: проверка идёт после всей
    # выписки.
    checker = None
    if settings.llm_verify_model and not same_model(
        settings.llm_verify_model, settings.llm_model
    ):
        checker = LLMClient(replace(settings, llm_model=settings.llm_verify_model))
        logger.info("Проверять поручения будет {}", settings.llm_verify_model)

    record = Journal()
    tasks = extract_tasks(
        chunks, client, progress=progress, answers=answers,
        corpus=named.as_text(), chair=chair, journal=record, directory=units,
        verifier=checker,
    )
    logger.info("Найдено поручений: {}", len(tasks))

    # Чей вопрос разбирали. Считается по всей стенограмме, а не по окну:
    # направление объявляют один раз, а поручают потом четверть часа.
    marks = units.follow(for_model)
    # Поручению, данному всему залу, направление не приписывается: «обращайтесь
    # в головную организацию — все регионы — Пермский край» сужает адресата
    # до одного субъекта, хотя сказано было всем.
    tasks = [
        task if task.who in COLLECTIVE_NAMES
        else replace(task, unit=unit_at(marks, task.at))
        for task in tasks
    ]
    filled = sum(1 for task in tasks if task.unit)
    logger.info(
        "Разбор шёл по {} направлениям, у {} поручений графа «{}» заполнена",
        len({name for _, name in marks if name}), filled, units.label,
    )

    protocol = build_protocol(
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
        unit_label=units.label,
        addressees=units.addressees(),
        decision_formula=settings.decision_formula,
    )
    protocol.journal = record
    return protocol


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


def cache_size(work_dir: str | Path) -> int:
    """Сколько байт занято промежуточным.

    Гигабайты копятся незаметно: извлечённый звук остаётся после каждой
    записи, а нужен он только чтобы не считать заново. Пока размер не видно,
    его никто и не чистит.
    """
    folder = Path(work_dir).expanduser()
    if not folder.exists():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def clear_cache(work_dir: str | Path) -> int:
    """Убирает промежуточное. Возвращает, сколько байт освободилось.

    Готовые файлы это не трогает: они лежат в другой папке. Потеряется
    только возможность пересобрать протокол без повторного распознавания —
    но стенограмма к тому времени уже сохранена, и пересобирать можно из
    неё.
    """
    folder = Path(work_dir).expanduser()
    freed = cache_size(folder)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    return freed


def check_writable(out_dir: str | Path) -> str:
    """Можно ли туда писать. Пустая строка — можно.

    Спрашивается до расчёта, а не после. Папка для результатов часто сетевая,
    а сетевой диск отваливается: буква не подключена в этом сеансе, сервер
    недоступен, прав на запись нет. Узнавать об этом после сорока минут
    распознавания — терять сорок минут.
    """
    folder = Path(out_dir).expanduser()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".minuteforge-проверка"
        probe.write_text("проверка", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return (
            f"Писать в «{folder}» не получается: {exc.strerror or exc}.\n"
            "Если это сетевой диск — доступен ли он сейчас и есть ли права "
            "на запись?"
        )
    return ""


def warm_up(client, progress=None) -> None:
    """Заставляет сервер загрузить модель до начала разбора.

    Загрузка семимиллиардной модели в видеопамять занимает десятки секунд, а
    после распознавания память ещё занята torch, и Ollama может перезапустить
    свой рантайм — на эти секунды сервер отказывает в соединении.

    Раньше в это окно попадал первый же рабочий запрос, и человек видел
    «модель не отвечает» на записи, которую только что сорок минут
    распознавали. Здесь то же ожидание проходит один раз, отдельным шагом, и
    неудача его не останавливает: настоящие запросы всё равно повторяются.
    """
    from .transcribe import Step, report

    report(progress, Step(name="warmup", title="Гружу модель в память", share=0.0))
    try:
        client.complete("Ответь одним словом.", "Готов?", max_tokens=8)
    except Exception as exc:
        logger.warning("Прогрев не удался, продолжаю: {}", exc)
    report(progress, Step(name="warmup", title="Модель загружена", done=True, share=0.0))


def models_tag(*names: str) -> str:
    """Короткие имена моделей для имени прогона: `mistral+qwen14-protocol`.

    Тег после двоеточия отбрасывается: «mistral» и «mistral:latest» — одна и
    та же модель, а в имени файла разница только мешает. Повторы снимаются:
    когда проверяет та же модель, что выписывает, писать её дважды незачем.
    """
    short = [name.split(":")[0].strip() for name in names if name and name.strip()]
    return "+".join(dict.fromkeys(short))


def run_tag(
    meeting: str = "",
    stamp: datetime | None = None,
    models: str = "",
) -> str:
    """Метка прогона: `2026-09-03_1720_mistral+qwen14-protocol`.

    День совещания, час расчёта и модели, которыми считали.

    Дата именно совещания, а не расчёта. За один день считают несколько
    записей разных дат, и папки, названные днём обработки, выходят
    одинаковыми: разобрать, где какое заседание, можно только заглянув
    внутрь. Час расчёта остаётся вторым: одну и ту же запись пересчитывают
    по нескольку раз с разными настройками, и прогоны надо различать.

    Модели — потому что прогоны сейчас и различаются ими: ту же запись гоняют
    с разной проверочной моделью, и по числам сравнивают, какая лучше. Когда
    набор устоится, они станут шумом — тогда `name_with_models: false`.

    Дату совещания не разобрали — берётся день расчёта: пустое место в имени
    хуже неточного.
    """
    stamp = stamp or datetime.now()
    day = dates.parse_meeting_date(meeting) if meeting else None
    when = f"{day:%Y-%m-%d}_{stamp:%H%M}" if day else f"{stamp:%Y-%m-%d_%H%M}"
    return f"{when}_{models}" if models else when


def run_dir(
    out_dir: str | Path,
    stem: str,
    *,
    stamp: datetime | None = None,
    meeting: str = "",
    tag: str = "",
) -> Path:
    """Своя папка на каждый разбор: `2026-09-03_1720_совещание`.

    Иначе второй прогон той же записи молча затирает первый, и сравнить
    настройки не с чем — а сравнивать приходится постоянно, потому что
    «стало лучше» на глаз не определяется.

    Папка, а не длинные имена файлов: её целиком отдают в другое
    подразделение, и в ней лежит всё по одному совещанию сразу. Дата первой
    в имени — так проводник сортирует прогоны по порядку.
    """
    # Готовая метка важнее: имена файлов внутри папки берутся из неё же, и
    # посчитанная второй раз она разошлась бы с именем папки на минуту.
    return Path(out_dir).expanduser().resolve() / f"{tag or run_tag(meeting, stamp)}_{stem}"


def _free_name(path: Path) -> Path:
    """Не затирает то, что уже лежит рядом.

    Внутри одного прогона протокол пересобирают по нескольку раз — с другой
    моделью, с другим окном. Затирать предыдущий нельзя: сравнить будет не с
    чем, а какой из них лучше, решает человек, а не последний запуск.
    """
    if not path.exists():
        return path
    for number in range(2, 100):
        nearby = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not nearby.exists():
            return nearby
    return path


def _text_with_head(transcript: Transcript, marks: Sequence[Suspicion] | None = None) -> str:
    """Стенограмма с шапкой: чем распознано, что уступили, где приврано.

    :param marks: уже посчитанные подозрительные места. Передаются, чтобы не
        считать их дважды там, где о них ещё и в журнал пишут.
    """
    head = []
    if transcript.model:
        head.append(f"# Распознано моделью {transcript.model}")
    head.extend(f"# {note}" for note in transcript.notes)
    # Подозрительные места — тут же, в шапке, а не отдельным файлом: тот, кто
    # читает стенограмму, должен наткнуться на них прежде, чем поверит тексту.
    if marks is None:
        marks = suspicious(transcript.blocks)
    head.extend(f"# ПРОВЕРИТЬ. {item.as_line()}" for item in marks)
    body = transcript.as_text(with_time=True)
    return "\n".join([*head, "", body]) if head else body


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
    # Полный путь, а не тот, что ввели: относительный «data/output» ничего
    # не говорит человеку, который потом ищет файлы в проводнике.
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    text = _free_name(out_dir / f"{stem}.txt")
    # Шапка с моделью — первой строкой файла. Стенограмма уходит в другое
    # подразделение отдельным файлом, без интерфейса и без лога, и там
    # должно быть видно, чем она сделана: спор о том, «Ессентуки» или
    # «Исинтуки» сказал докладчик, решается именно этим.
    marks = suspicious(transcript.blocks)
    text.write_text(_text_with_head(transcript, marks), encoding="utf-8")

    data = _free_name(out_dir / f"{stem}.json")
    # Не голый список реплик, а список с историей: чем распознано и на что
    # пришлось пойти. Из этого файла потом пересобирают протокол, не
    # распознавая заново, — и на этом пути терялась строка «Распознано
    # моделью». Документ выходил без реквизита, которым решается спор,
    # «Ессентуки» или «Исинтуки» сказал докладчик.
    data.write_text(
        json.dumps(
            {
                "model": transcript.model,
                "notes": list(transcript.notes),
                "segments": [
                    {"speaker": b.speaker, "text": b.text, "start": b.start, "end": b.end}
                    for b in transcript.blocks
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Счёт подозрительных мест — в журнал: без него «в файле ничего нет»
    # значит сразу две разные вещи — не нашлось или не записалось.
    logger.info(
        "Стенограмма сохранена: {} и {}; шапка: модель {}, подозрительных мест {}",
        text.name, data.name, transcript.model or "не указана", len(marks),
    )
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
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if template is not None:
        text = Path(template).read_text(encoding="utf-8-sig")
        document_text = protocol.render(text)
        suffix = Path(template).suffix or ".md"
    else:
        document_text = protocol.as_markdown()
        suffix = ".md"

    document = _free_name(out_dir / f"{stem}{suffix}")
    document.write_text(document_text, encoding="utf-8")

    table = _free_name(out_dir / f"{stem}_поручения.csv")
    # BOM: без него Excel в русской локали открывает таблицу кракозябрами.
    table.write_text(protocol.tasks_csv(), encoding="utf-8-sig")

    saved = {"protocol": document, "tasks": table}

    if protocol.journal is not None:
        # Отдельным файлом и с говорящим именем: это не документ и не для
        # рассылки, а рабочая записка для того, кто настраивает разбор.
        report = _free_name(out_dir / f"{stem}_разбор.md")
        report.write_text(protocol.journal.as_markdown(), encoding="utf-8")
        saved["journal"] = report

    if with_transcript and protocol.transcript is not None:
        transcript = out_dir / f"{stem}_стенограмма.txt"
        transcript.write_text(
            protocol.transcript.as_text(with_time=True), encoding="utf-8"
        )
        saved["transcript"] = transcript

    logger.info("Сохранено: {}", ", ".join(path.name for path in saved.values()))
    return saved
