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
from .config import Settings
from .llm import LLMClient
from .audio import ffmpeg_available
from .people import merge_suggestions, mentioned_people, read_people
from .pipeline import Meeting, protocol_from_transcript, save, transcribe_meeting
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
    recognize.add_argument("--out", type=Path, default=None, help="куда положить стенограмму")
    recognize.add_argument("--model", default=None, help="модель Whisper: tiny…large-v3")
    recognize.add_argument("--language", default=None)
    recognize.add_argument("--speakers", type=int, default=None, help="сколько человек говорит")
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
    parser.add_argument("--out", type=Path, default=Path("data/output"))
    parser.add_argument("--title", default="Протокол совещания")
    parser.add_argument("--date", default="", help="дата и время совещания")
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
    segments = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(segments, dict):  # на случай, если сохранили целиком результат
        segments = segments.get("segments", [])
    return Transcript(consolidate(blocks_from_segments(segments)))


def save_transcript(transcript: Transcript, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"speaker": b.speaker, "text": b.text, "start": b.start, "end": b.end}
        for b in transcript.blocks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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

    for model, why in check_model_access(settings.hf_token, models=GATED_MODELS).items():
        name = model.split("/")[-1]
        print(f"  {'ок  ' if not why else 'НЕТ '} {name}{'' if not why else ': ' + why}")
        ok = ok and not why

    client = LLMClient(settings)
    trouble = client.diagnose()
    print(f"  {'ок  ' if not trouble else 'НЕТ '} модель {settings.llm_model}")
    if trouble:
        print(f"       {trouble}")
        ok = False

    installed = client.available_models()
    if installed:
        print(f"  установлены: {', '.join(installed)}")

    return 0 if ok else 1


def print_step(step) -> None:
    """Ход работы строчками. В терминале это единственный признак жизни:
    распознавание часовой записи идёт десятки минут молча."""
    if not step.done:
        return
    mark = "взят готовым" if step.reused else f"{step.elapsed_s} с"
    print(f"  [{step.share:>4.0%}] {step.title} — {mark}", flush=True)



def command_recognize(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    work_dir = args.out or args.source.parent
    transcript = transcribe_meeting(
        args.source, settings, work_dir=work_dir,
        diarize=not args.no_diarize, progress=print_step,
    )
    path = save_transcript(transcript, Path(work_dir) / f"{args.source.stem}_segments.json")

    for note in getattr(transcript, "notes", []):
        print(f"  ВНИМАНИЕ: не хватило видеопамяти — {note}")
    print(f"Реплик: {len(transcript.blocks)}, говорящих: {len(transcript.speakers)}")
    print(f"Метки: {', '.join(transcript.speakers)}")
    print(f"Стенограмма: {path}")
    print("Дальше: minuteforge protocol", path, '--name SPEAKER_00="Фамилия И.О."')
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
    protocol = _build(transcript, args, settings)
    _report(protocol, save(protocol, args.out, template=args.template))
    return 0


def command_run(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    transcript = transcribe_meeting(
        args.source, settings, work_dir=args.out, progress=print_step
    )
    protocol = _build(transcript, args, settings)
    _report(protocol, save(protocol, args.out, template=args.template))
    return 0


def _build(transcript: Transcript, args: argparse.Namespace, settings: Settings):
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
        client=LLMClient(settings), progress=print_step,
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
    try:
        return COMMANDS[args.command](args)
    except MissingToken as exc:
        # Не поломка, а незаконченная настройка: печатаем как есть, без
        # трассировки, которая тут ничего не объясняет.
        print(str(exc), file=sys.stderr)
        return 2
    except (RecognitionError, FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
