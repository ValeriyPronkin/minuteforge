"""Командная строка.

    python -m minuteforge recognize meeting.mp4
    python -m minuteforge protocol segments.json --name SPEAKER_00="Орлов В.П."
    python -m minuteforge run meeting.mp4

Три команды, а не одна, по той же причине, по которой в интерфейсе три шага:
распознавание требует видеокарты, сборка протокола — нет, а между ними
человек говорит, кто есть кто. Разделение позволяет распознать запись на
машине с видеокартой, унести стенограмму и собрать протокол где угодно.

Интерфейс удобнее для одной записи, командная строка — когда записей
десяток или когда всё это должно идти по расписанию, без человека у экрана.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from loguru import logger

from .blocks import Transcript, blocks_from_segments, consolidate
from .checks import suspicious
from .config import HF_TOKEN_ENV, Settings
from .journal import setup_file_log
from .llm import LLMClient, free_the_card
from .audio import AudioError, ffmpeg_available
from .meetings import meeting_for, runs_root
from .people import merge_suggestions, mentioned_people, read_people
from .pipeline import (
    Meeting,
    check_writable,
    keep_as_latest,
    lay_out_transcript,
    models_tag,
    protocol_from_transcript,
    recorded_when,
    run_dir,
    run_tag,
    save,
    transcribe_meeting,
)
from .transcribe import GATED_MODELS, MissingToken, RecognitionError, check_model_access


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minuteforge",
        description="Протокол видеосовещания с поручениями.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser(
        "check", help="проверить доступ к моделям и связь с локальной моделью"
    )
    check.add_argument("--llm-url", default=None, help="адрес модели поручений")
    check.add_argument("--llm-model", default=None, help="имя модели поручений")

    recognize = commands.add_parser(
        "recognize", help="запись -> стенограмма (нужна видеокарта)"
    )
    recognize.add_argument("source", type=Path, help="видео или аудио")
    recognize.add_argument(
        "--out", type=Path, default=None,
        help="куда положить стенограмму. Не задан — папка заседания, та же, "
        "откуда взяли запись; заседания не видно — output_dir из настроек",
    )
    recognize.add_argument("--model", default=None, help="модель Whisper: tiny…large-v3")
    recognize.add_argument("--language", default=None)
    recognize.add_argument("--speakers", type=int, default=None, help="сколько человек говорит")
    recognize.add_argument(
        "--from", dest="start", default=None, metavar="ВРЕМЯ",
        help="распознавать с этого места: 1:05:30 или 3930",
    )
    recognize.add_argument(
        "--to", dest="end", default=None, metavar="ВРЕМЯ",
        help="и до этого; пусто — до конца записи",
    )
    recognize.add_argument(
        "--fresh", action="store_true",
        help="пересчитать с нуля шаги выбранной модели; шаги других моделей остаются",
    )
    recognize.add_argument(
        "--no-diarize",
        action="store_true",
        help="без разделения по говорящим: быстрее и не нужен токен, но "
        "стенограмма выйдет без авторов реплик",
    )

    protocol = commands.add_parser(
        "protocol", help="стенограмма -> протокол (видеокарта не нужна)"
    )
    protocol.add_argument("segments", type=Path, help="файл стенограммы json")
    _add_meeting_arguments(protocol)

    compare = commands.add_parser(
        "compare",
        help="прогнать одну стенограмму через несколько моделей и сравнить",
    )
    compare.add_argument("segments", type=Path, help="файл стенограммы json")
    compare.add_argument(
        "--model", action="append", default=[], required=True,
        help="какую модель проверить. Можно несколько раз.",
    )
    compare.add_argument("--llm-url", default=None)
    compare.add_argument("--out", type=Path, default=None, help="куда сложить результаты")

    run = commands.add_parser("run", help="запись -> протокол, всё разом")
    run.add_argument("source", type=Path, help="видео или аудио")
    run.add_argument("--model", default=None, help="модель Whisper")
    run.add_argument("--language", default=None)
    run.add_argument("--speakers", type=int, default=None)
    _add_meeting_arguments(run)

    return parser


def _add_meeting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out", type=Path, default=None,
        help="куда положить результат. Не задан — папка заседания, та же, "
        "откуда взяли запись; заседания не видно — output_dir из настроек",
    )
    parser.add_argument("--title", default="Протокол совещания")
    parser.add_argument(
        "--date", default="",
        help="дата и время совещания; от неё считаются сроки вида «через две "
        "недели». В команде run пусто — взять из самой записи",
    )
    parser.add_argument("--place", default="")
    parser.add_argument("--chair", default="")
    parser.add_argument("--secretary", default="", help="секретарь совещания")
    parser.add_argument("--number", default="", help="номер протокола")
    parser.add_argument(
        "--people", type=Path, default=None,
        help="список участников: csv или txt с ФИО и должностями",
    )
    parser.add_argument(
        "--template", type=Path, default=None,
        help="форма протокола с местами вида {{date}}; без неё — встроенная разметка",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        metavar="МЕТКА=ИМЯ",
        help='кто есть кто: --name SPEAKER_00="Орлов В.П.". Можно несколько раз. '
        "Метка без имени останется в протоколе как есть.",
    )
    parser.add_argument("--llm-url", default=None)
    parser.add_argument("--llm-model", default=None)


def results_dir(args: argparse.Namespace, settings: Settings, beside: Path | None) -> Path:
    """Куда ложится результат: заданное ключом, папка заседания, настройки.

    Папка заседания — та, в которой лежит то, с чего начали: запись для
    ``run`` и ``recognize``, стенограмма для ``protocol``. Так же считает её
    и интерфейс, и правило у них общее — ``meetings.meeting_for``.

    Прежде ключ ``--out`` молча указывал на ``data/output``, и стенограмма
    уходила туда независимо от того, где лежит запись: папку под заседание
    заводили заранее, складывали в неё присланное секретарём, а результат
    находили в перевалочной папке приложения и разносили руками.

    Заседания не видно — остаётся ``output_dir`` из настроек: так бывает,
    когда запись лежит в перевалочной или рабочей папке.
    """
    if getattr(args, "out", None):
        return Path(args.out).expanduser().resolve()
    folder, _ = meeting_for(
        Path(beside).expanduser().resolve() if beside else None,
        meetings_dir=_dir(settings.meetings_dir),
        not_meetings=(_dir(settings.input_dir), _dir(settings.work_dir)),
    )
    return folder or _dir(settings.output_dir)


def _dir(where: str | Path) -> Path:
    """Папка из настроек полным путём.

    Пути в настройках пишут и относительными — ``data/vks``, — а сравнивать
    их приходится с полными: путь к записи приходит от того, кто запускает.
    Относительный считается от текущей папки, как и сам ``config.yaml``.
    """
    return Path(where).expanduser().resolve()


def parse_names(pairs: list[str]) -> dict[str, str]:
    """Разбирает ``--name SPEAKER_00=Орлов В.П.``.

    Имя может содержать что угодно, включая знак равенства, поэтому строка
    делится по первому вхождению, а не по всем.
    """
    names: dict[str, str] = {}
    for pair in pairs:
        label, sep, value = pair.partition("=")
        if not sep or not value.strip():
            raise ValueError(
                f"не разобрать имя участника: {pair!r}. Ожидается вид "
                'SPEAKER_00="Фамилия И.О."'
            )
        names[label.strip()] = value.strip()
    return names


def settings_from(args: argparse.Namespace, base: Settings | None = None) -> Settings:
    """Настройки: файл и окружение, поверх — то, что задано в командной строке."""
    settings = base or Settings.load()
    overrides = {
        "asr_model": getattr(args, "model", None),
        "language": getattr(args, "language", None),
        "speakers": getattr(args, "speakers", None),
        "llm_base_url": getattr(args, "llm_url", None),
        "llm_model": getattr(args, "llm_model", None),
    }
    for field, value in overrides.items():
        if value is not None:
            setattr(settings, field, value)
    return settings


def load_transcript(path: Path) -> Transcript:
    """Читает стенограмму, сохранённую распознаванием или интерфейсом."""
    if not path.exists():
        raise FileNotFoundError(f"Файл стенограммы не найден: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    # Голый список — стенограмма прежних версий, и такие файлы никуда не
    # делись. Словарь — наш нынешний вид, с историей распознавания.
    segments = body.get("segments", []) if isinstance(body, dict) else body
    transcript = Transcript(consolidate(blocks_from_segments(segments)))
    if isinstance(body, dict):
        transcript.model = str(body.get("model") or "")
        transcript.notes = [str(note) for note in body.get("notes") or []]
        transcript.hints = str(body.get("hints") or "")
    return transcript


def describe_environment() -> list[tuple[bool, str]]:
    """Что установлено и увидит ли расчёт видеокарту.

    Проверять это нужно до первого запуска, а не после: без CUDA всё
    выглядит исправным и просто считает в двадцать раз дольше — человек
    решает, что так и должно быть, и ждёт часами.
    """
    checks: list[tuple[bool, str]] = []

    checks.append((
        ffmpeg_available(),
        "ffmpeg — им извлекается звук из видео"
        if ffmpeg_available()
        else "ffmpeg не найден: Windows — winget install ffmpeg, "
        "macOS — brew install ffmpeg, Ubuntu — apt install ffmpeg",
    ))

    try:
        import torch
    except ImportError:
        checks.append((
            False,
            "torch не установлен — распознавание работать не будет: "
            "pip install -r requirements-asr.txt",
        ))
    else:
        cuda = bool(torch.cuda.is_available())
        if cuda:
            checks.append((True, f"видеокарта видна: {torch.cuda.get_device_name(0)}"))
        else:
            checks.append((
                False,
                "torch не видит видеокарту — считать будет процессор, "
                "час записи вместо минут. Чаще всего это сборка без CUDA: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121",
            ))

    try:
        import whisperx  # noqa: F401
    except ImportError:
        checks.append((False, "whisperx не установлен: pip install -r requirements-asr.txt"))
    else:
        checks.append((True, "whisperx на месте"))

    return checks


def command_check(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    ok = True

    # Проверка сама рассказывает о каждой находке понятным языком, и
    # служебные предупреждения поверх этого только сбивают: одно и то же будет
    # сказано дважды, второй раз — трассировкой про Connection refused.
    logger.disable("minuteforge")
    try:
        return _run_checks(settings, ok)
    finally:
        logger.enable("minuteforge")


def _run_checks(settings: Settings, ok: bool) -> int:

    for good, note in describe_environment():
        print(f"  {'ок  ' if good else 'НЕТ '} {note}")
        ok = ok and good

    # Отдельной строкой: «нет токена» и «токен не принят» лечатся по-разному,
    # а в списке моделей выглядят одинаково.
    token = settings.hf_token
    if token:
        print(f"  ок   {HF_TOKEN_ENV} виден: {token[:4]}…{token[-2:]}, длина {len(token)}")
    else:
        print(f"  НЕТ  {HF_TOKEN_ENV} не виден этому процессу")
        print("       Проверить: echo %HF_TOKEN% (cmd), $env:HF_TOKEN (PowerShell).")
        print('       Задать: setx HF_TOKEN "hf_..." — и открыть НОВОЕ окно')
        print("       терминала: в текущем переменной не появится.")
        print("       Токен нужен только разметке голосов. Без неё стенограмма")
        print("       будет без имён, а поручения извлекутся: --no-diarize.")

    for model, why in check_model_access(token, models=GATED_MODELS).items():
        name = model.split("/")[-1]
        print(f"  {'ок  ' if not why else 'НЕТ '} {name}{'' if not why else ': ' + why}")
        ok = ok and not why

    where = settings.output_dir
    cannot = check_writable(where)
    print(f"  {'ок  ' if not cannot else 'НЕТ '} запись в {where}")
    if cannot:
        print(f"       {cannot.splitlines()[0]}")
        ok = False

    client = LLMClient(settings)
    trouble = client.diagnose()
    print(f"  {'ок  ' if not trouble else 'НЕТ '} модель {settings.llm_model}")
    if trouble:
        print(f"       {trouble}")
        ok = False

    installed = client.available_models()
    if installed:
        print(f"  установлены: {', '.join(installed)}")

    window, limit = client.context_window()
    if window:
        print(f"  окно модели {settings.llm_model}: {window} токенов")
    if limit and limit < settings.chunk_budget:
        print(
            f"  НЕТ  сервер обрежет запрос до {limit} токенов, а фрагмент у нас "
            f"{settings.chunk_budget}: конца фрагмента модель не увидит.\n"
            f"       Либо поднимите num_ctx у модели, либо поставьте окно "
            f"{limit} в настройках."
        )
        ok = False
    elif limit:
        print(f"  ок   предел сервера: {limit} токенов")
    elif window:
        print(
            f"  ?    сервер не сообщает свой предел (num_ctx). По умолчанию "
            f"Ollama берёт 2048–4096 и молча отрезает остальное, а фрагмент у "
            f"нас {settings.chunk_budget}.\n"
            f"       Задайте предел явно: PARAMETER num_ctx 8192 в Modelfile."
        )

    return 0 if ok else 1


def print_step(step) -> None:
    """Ход работы строчками. В терминале это единственный признак жизни:
    распознавание часовой записи идёт десятки минут молча."""
    if not step.done:
        # Начало шага и его ход — тоже строчкой: иначе между «Извлекаю звук»
        # и концом распознавания терминал молчал десятки минут, и было не
        # понять, идёт работа или всё повисло.
        tail = f"{step.percent:.0f}% шага" if step.percent is not None else "начат"
        print(f"  [{step.share:>4.0%}] {step.title} — {tail}", flush=True)
        return
    mark = "не считался, взят из сохранённого" if step.reused else f"{step.elapsed_s} с"
    print(f"  [{step.share:>4.0%}] {step.title} — {mark}", flush=True)



def _say_latest(written: dict, out: Path, settings: Settings) -> None:
    """Свежий протокол в корень папки заседания — и сказать об этом.

    Иначе за ним ходят в «runs» и выбирают папку по дате, а нужен он обычно
    именно свежий.
    """
    latest = keep_as_latest(
        written, out, meetings_dir=_dir(settings.meetings_dir)
    )
    if latest is not None:
        print(
            f"Свежий протокол положен в {out} как {latest.name} — "
            "подписанный «протокол» рядом не затронут."
        )


def _say_reference(laid, out: Path) -> None:
    """Про стенограмму заседания — вслух.

    Легла она в корень папки заседания или там уже лежала, видно только по
    этой строчке: заменить лежащую может лишь человек, и молчать о том, что
    расшифровка никуда не попала, нельзя.
    """
    if not laid.meeting:
        return
    if laid.reference:
        print(f"Стенограмма заседания положена в {out}")
    else:
        print(
            f"Стенограмма заседания в {out} уже была — оставлена как есть. "
            "Заменить её может только человек: вторая расшифровка бывает "
            "хуже первой."
        )


def command_recognize(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    out = results_dir(args, settings, args.source)
    # Проверяется до расчёта, а не при сохранении: папка часто сетевая, а
    # узнавать о недоступном диске после сорока минут распознавания —
    # терять сорок минут.
    trouble = check_writable(out)
    if trouble:
        print(trouble)
        return 1
    print(f"Результат ляжет в {out}")
    free_the_card(settings)
    # Промежуточное — в рабочую папку, а не к результату: из часовой записи
    # извлекается больше сотни мегабайт звука, и в папке заседания ему не
    # место. Прежде рабочей папкой служила папка вывода, и звук ложился
    # рядом со стенограммой — а убирают его как промежуточное.
    transcript = transcribe_meeting(
        args.source, settings, work_dir=_dir(settings.work_dir),
        diarize=not args.no_diarize, progress=print_step, fresh=args.fresh,
        start=args.start, end=args.end,
    )
    # Дата совещания в имени папки прогона — не украшение: проводник
    # сортирует прогоны по ней, а у recognize своего ключа --date нет. Знает
    # её сама запись, как и в команде run; не знает — останется день
    # расчёта, пустое место в имени хуже неточного.
    heard_on = recorded_when(args.source)
    laid = lay_out_transcript(
        transcript, out, args.source.stem, settings,
        meeting=heard_on.as_text() if heard_on else "",
        meetings_dir=_dir(settings.meetings_dir),
    )
    paths = laid.files

    for note in getattr(transcript, "notes", []):
        print(f"  ВНИМАНИЕ: не хватило видеопамяти — {note}")
    if transcript.model:
        print(f"Распознано моделью: {transcript.model}")
    for item in suspicious(transcript.blocks, transcript.hints):
        print(f"  ПРОВЕРИТЬ. {item.as_line()}")
    print(f"Реплик: {len(transcript.blocks)}, говорящих: {len(transcript.speakers)}")
    print(f"Метки: {', '.join(transcript.speakers)}")
    print(f"Стенограмма текстом: {paths['text']}")
    print(f"Стенограмма для расчёта: {paths['json']}")
    _say_reference(laid, out)
    print("Дальше: minuteforge protocol", paths["json"], '--name SPEAKER_00="Фамилия И.О."')
    return 0


def command_compare(args: argparse.Namespace) -> int:
    """Сравнивает модели на одной и той же стенограмме.

    Спорить о том, какая модель лучше, бессмысленно: на извлечении поручений
    из русского совещания они ведут себя по-разному, и решает это замер на
    вашем материале, а не общие соображения. Стенограмма берётся одна, чтобы
    разница была только в модели.
    """
    transcript = load_transcript(args.segments)
    settings = settings_from(args)
    rows = []

    for name in args.model:
        model_settings = replace(settings, llm_model=name)
        client = LLMClient(model_settings)
        trouble = client.diagnose()
        if trouble:
            print(f"{name}: пропущена — {trouble.splitlines()[0]}")
            continue

        print(f"{name}: считаю…", flush=True)
        started = time.perf_counter()
        protocol = protocol_from_transcript(
            transcript, model_settings, client=client
        )
        elapsed = time.perf_counter() - started
        rows.append((name, protocol, elapsed))

        if args.out:
            save(protocol, args.out, stem=f"protocol_{name.replace(':', '_')}")

    if not rows:
        print("Ни одна модель не ответила.", file=sys.stderr)
        return 1

    print()
    print(f"{'Модель':<22} {'Поручений':>9} {'С исполн.':>9} {'Время, с':>9}")
    for name, protocol, elapsed in rows:
        print(
            f"{name:<22} {len(protocol.tasks):>9} "
            f"{len(protocol.actionable):>9} {elapsed:>9.0f}"
        )

    print()
    print("Числа — не оценка: больше найденных поручений не значит «лучше».")
    print("Смотрите сами протоколы: модель могла принять обсуждение за поручение.")
    if args.out:
        print(f"Протоколы для сверки: {args.out}")
    return 0


def command_protocol(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    transcript = load_transcript(args.segments)
    # Заседание задаёт стенограмма: она и есть предмет разбора, а лежит она
    # в папке своего заседания — хоть в корне, хоть в папке прогона.
    out = results_dir(args, settings, args.segments)
    trouble = check_writable(out)
    if trouble:
        print(trouble)
        return 1
    folder, stem = _run_place(out, args, settings, args.segments.stem)
    print(f"Результат ляжет в {folder}")
    protocol = _build(transcript, args, settings, run_name=stem)
    written = save(
        protocol, folder, stem=f"{stem}_протокол",
        template=args.template or settings.form_file,
    )
    _report(protocol, written)
    _say_latest(written, out, settings)
    return 0


def _run_place(
    out: Path, args: argparse.Namespace, settings: Settings, stem: str
) -> tuple[Path, str]:
    """Папка прогона и метка, которой подписываются файлы в ней.

    Своя папка на каждый разбор: иначе второй прогон той же стенограммы
    молча затирает первый. Метка несёт дату совещания, час расчёта и модель
    — из папки файлы вынимают, пересылают, кладут рядом с чужими, и
    «протокол.md» от трёх разных заседаний в одной папке загрузок
    неразличимы.

    Модель в метке — та, что собирает поручения, а не распознаёт: протокол
    пересобирают по нескольку раз разными моделями из одной стенограммы.
    """
    made_by = (
        models_tag(settings.llm_model, settings.llm_verify_model)
        if settings.name_with_models else ""
    )
    tag = run_tag(getattr(args, "date", "") or "", models=made_by)
    place = run_dir(
        runs_root(out, meetings_dir=_dir(settings.meetings_dir)), stem, tag=tag
    )
    return place, tag


def command_run(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    # Дата совещания — не украшение шапки: от неё считаются сроки, а знает
    # её обычно сама запись. Спрашивать её отдельным ключом там, где она
    # лежит в имени файла, значит получать протоколы без сроков.
    if not args.date:
        found = recorded_when(args.source)
        if found is None:
            print(
                "Дата совещания неизвестна: ни в имени файла, ни в свойствах "
                "записи её нет. Сроки вида «через две недели» останутся "
                "словами — задайте --date."
            )
        else:
            args.date = found.as_text()
            print(f"Дата совещания {found.source}: {args.date}")
    out = results_dir(args, settings, args.source)
    trouble = check_writable(out)
    if trouble:
        print(trouble)
        return 1
    print(f"Результат ляжет в {out}")
    free_the_card(settings)
    # Промежуточное — в рабочую папку: см. command_recognize.
    transcript = transcribe_meeting(
        args.source, settings, work_dir=_dir(settings.work_dir),
        progress=print_step,
    )
    laid = lay_out_transcript(
        transcript, out, args.source.stem, settings,
        meeting=args.date, meetings_dir=_dir(settings.meetings_dir),
    )
    print(f"Стенограмма: {laid.files['text']}")
    _say_reference(laid, out)
    # Протокол ложится в ту же папку прогона, что и стенограмма: всё по
    # одному совещанию сразу, папку целиком отдают в другое подразделение.
    # А подписан он своей меткой — с моделью, которая собрала поручения.
    _, stem = _run_place(out, args, settings, args.source.stem)
    protocol = _build(transcript, args, settings, run_name=stem)
    written = save(
        protocol, laid.folder, stem=f"{stem}_протокол",
        with_transcript=False,
        template=args.template or settings.form_file,
    )
    _report(protocol, written)
    _say_latest(written, out, settings)
    return 0


def _build(
    transcript: Transcript, args: argparse.Namespace, settings: Settings,
    *, run_name: str = "",
):
    roster = read_people(args.people) if args.people else []
    people = merge_suggestions(roster, mentioned_people(transcript.as_text()))
    meeting = Meeting(
        title=args.title,
        date=args.date,
        place=args.place,
        chair=args.chair,
        secretary=args.secretary,
        number=args.number,
        names=parse_names(args.name) or None,
        people=people or None,
    )
    return protocol_from_transcript(
        transcript, settings, meeting=meeting,
        client=LLMClient(settings), progress=print_step, run_name=run_name,
    )


def _report(protocol, paths: dict[str, Path]) -> None:
    unclear = len(protocol.needs_clarification)
    print(f"Поручений: {len(protocol.tasks)}, из них без исполнителя: {unclear}")
    if not protocol.tasks and protocol.answers:
        print("  Модель не вернула ни одного поручения. Вот её ответ целиком —")
        print("  если это проза или пересказ задания, дело в модели:")
        for number, answer in enumerate(protocol.answers, 1):
            short = (answer or "(пустой ответ)").strip().replace("\n", " ")
            print(f"    [{number}] {short[:200]}")
    if unclear:
        print("  Пункты без исполнителя не выброшены — см. раздел «Требуют уточнения».")
    for label, key in (("Протокол", "protocol"), ("Стенограмма", "transcript"), ("Поручения", "tasks")):
        if key in paths:
            print(f"{label}: {paths[key]}")


COMMANDS = {
    "check": command_check,
    "compare": command_compare,
    "recognize": command_recognize,
    "protocol": command_protocol,
    "run": command_run,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # До разбора команды: журнал должен застать и её саму, и всё, что
    # случится дальше. Недоступная папка работу не остановит.
    base = Settings.load()
    setup_file_log(base.log_dir, base.log_level)
    try:
        return COMMANDS[args.command](args)
    except MissingToken as exc:
        # Не поломка, а незаконченная настройка: печатаем как есть, без
        # трассировки, которая тут ничего не объясняет.
        print(str(exc), file=sys.stderr)
        return 2
    except (RecognitionError, AudioError, FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
