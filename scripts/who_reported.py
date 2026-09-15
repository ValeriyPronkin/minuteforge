"""Кто о чём доложил: отрезки записи под раздел «Отметили».

    python scripts/who_reported.py стенограмма.json --directory справочник.csv --noted отметили.md

В подписанном протоколе два раздела, а инструмент собирает один. «Решили»
складывается из поручений — их ищут фразой в двухчасовой записи, и это
иголка в стоге. «Отметили» устроено иначе: один пункт — один доклад, «Информацию
такого-то о том-то», и пункты идут по кругу разбора, а не по алфавиту.

Единица здесь не фраза, а связный отрезок записи, и границы у него берутся не
из суждения модели, а из двух готовых вещей: смены голоса и объявления
очередного направления по справочнику. Значит, потолок полноты у этого раздела
считается без модели и без видеокарты — ровно то, что делает скрипт.

Считается три величины:

*   **сколько отрезков набирается** и сколько речи остаётся вне разбора. Вне
    разбора идут общие доклады, у которых направления нет вовсе: у них ключ
    другой, и это видно сразу, а не после прогона.
*   **полнота по направлениям** — для скольких пунктов эталона нашёлся свой
    отрезок. Печатается вместе со случайным уровнем: назвать половину
    справочника и отчитаться о полноте легко, и мерка уже льстила однажды.
*   **что лишнее** — направления с отрезком, которых в эталоне нет.

Эталон необязателен. Без него скрипт просто показывает раскладку записи по
направлениям — этого хватает, чтобы понять, есть ли вообще из чего собирать
раздел.

Файлы с записями совещаний в репозиторий не попадают: пути передаются
аргументами, ничего не читается по умолчанию.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.blocks import (  # noqa: E402
    Block,
    Transcript,
    blocks_from_segments,
    consolidate,
    drop_soundcheck,
    normalize_speaker,
    timestamp,
    within,
)
from minuteforge.directory import Directory, read_directory  # noqa: E402
# Раскладка по направлениям берётся у самого приложения, а не пишется здесь
# заново: скрипт меряет то, что пойдёт в документ, а не свою копию этого.
from minuteforge.notes import REPORT_SECONDS, Report, split_into_reports  # noqa: E402
from minuteforge.tasks import asks_for_work  # noqa: E402

#: Когда отрезок похож не на доклад, а на перекличку. Признака два, и нужны
#: оба: голосов на минуту больше, чем бывает у одного выступления, и внутри
#: названы чужие направления. Поодиночке они врут — короткий доклад «вопросов
#: нет, спасибо» даёт те же два голоса на минуту, а длинный разбор кроме
#: своего региона поминает десяток соседних. Это подсказка отчёта, а не
#: решение: помеченный отрезок смотрят глазами.
NOISY_VOICES = 1.0
NOISY_OTHERS = 3

#: Строка стенограммы в текстовом виде: «[00:12:30] SPEAKER_00: текст».
_LINE = re.compile(r"^(?:\[(\d{1,2}):(\d{2}):(\d{2})\]\s*)?([^:]{1,60}?)\s*:\s*(.+)$")

#: Пункт нумерованного списка в эталоне: «12. Информацию такого-то…».
_ITEM = re.compile(r"^(\d{1,3})[.)]\s+(.*)$")


def others(found: Report, units: Directory) -> list[str]:
    """Чужие направления, названные внутри отрезка."""
    said = " ".join(block.text for block in found.blocks)
    return [name for name in units.find_all(said) if name != found.unit]


def looks_like_roll_call(found: Report, units: Directory) -> bool:
    """Похож ли отрезок на перекличку, а не на чей-то доклад."""
    minutes = found.seconds / 60
    if not minutes:
        return False
    return (
        len(found.voices) / minutes >= NOISY_VOICES
        and len(others(found, units)) >= NOISY_OTHERS
    )


def read_blocks(path: Path) -> Transcript:
    """Читает стенограмму: json со сегментами или текст построчно."""
    if path.suffix.lower() == ".json":
        body = json.loads(path.read_text(encoding="utf-8-sig"))
        # WhisperX отдаёт {"segments": [...]}, наш save_transcript — список.
        segments = body.get("segments", []) if isinstance(body, dict) else body
        return Transcript(consolidate(blocks_from_segments(segments)))
    return Transcript(consolidate(_from_text(path.read_text(encoding="utf-8-sig"))))


def _from_text(text: str) -> list[Block]:
    """Разбирает стенограмму, сохранённую текстом."""
    blocks: list[Block] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            if blocks:
                blocks[-1].text = f"{blocks[-1].text} {line}".strip()
            continue
        hours, minutes, seconds, speaker, said = match.groups()
        start = None
        if hours is not None:
            start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        blocks.append(Block(normalize_speaker(speaker), said.strip(), start, start))
    return blocks


def named_in(text: str, units: Directory) -> list[str]:
    """Все направления, названные в тексте."""
    return units.find_all(text)


def read_noted(path: Path) -> list[tuple[str, str]]:
    """Читает эталонный раздел «Отметили»: номер пункта и его текст.

    Ждёт нумерованный список. Если в файле есть заголовок со словом
    «отметили», читается только его раздел: подписанный протокол приносят
    целиком, и пункты раздела «Решили» здесь ни при чём.

    Продолжение пункта приписывается к нему: у пунктов с подпунктами
    название направления стоит не в первой строке.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    heading = next(
        (i for i, line in enumerate(lines)
         if line.startswith("#") and "отметили" in line.lower()),
        None,
    )
    if heading is not None:
        end = next(
            (i for i in range(heading + 1, len(lines)) if lines[i].startswith("#")),
            len(lines),
        )
        lines = lines[heading + 1:end]

    items: list[list[str]] = []
    numbers: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ITEM.match(line)
        if match:
            numbers.append(match.group(1))
            items.append([match.group(2)])
        elif items:
            items[-1].append(line)
    return [(number, " ".join(parts)) for number, parts in zip(numbers, items)]


def _share(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "—"


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} мин"


def _plural(count: int, one: str, few: str, many: str) -> str:
    if count % 100 // 10 == 1:
        return f"{count} {many}"
    last = count % 10
    if last == 1:
        return f"{count} {one}"
    return f"{count} {few}" if 2 <= last <= 4 else f"{count} {many}"


def report_layout(
    spans: Sequence[Report], outside: Report, units: Directory, *, floor: float,
) -> list[Report]:
    """Печатает раскладку записи по направлениям. Возвращает отрезки-доклады."""
    print("Разбор по направлениям")
    print(f"{'направление':<32}{'начало':>10}{'речи':>9}{'реплик':>8}{'голосов':>9}")
    reports: list[Report] = []
    short: list[Report] = []
    noisy: list[Report] = []
    for span in spans:
        where = reports if span.seconds >= floor else short
        where.append(span)
        mark = "" if span.seconds >= floor else "  ·"
        back = f" (возвратов {span.runs - 1})" if span.runs > 1 else ""
        if span.guessed:
            back += " — не объявлен, узнан по содержанию"
        if looks_like_roll_call(span, units):
            noisy.append(span)
            back += " — похоже на перекличку"
        print(
            f"{span.unit[:31]:<32}{timestamp(span.start):>10}"
            f"{_minutes(span.seconds):>9}{len(span.blocks):>8}"
            f"{len(span.voices):>9}{mark}{back}"
        )
    print()
    if short:
        marked = "помечен" if len(short) % 10 == 1 and len(short) % 100 != 11 else "помечены"
        print(
            f"Точкой {marked} {_plural(len(short), 'отрезок', 'отрезка', 'отрезков')} "
            f"короче {floor / 60:.0f} мин — доклада там нет: подключились, "
            "отчитались одной фразой или название прозвучало мимоходом."
        )
    if noisy:
        print(
            "Перекличкой помечены отрезки, где на минуту приходится больше "
            "голосов, чем бывает у одного доклада, и названы чужие направления. "
            "Слово там передают не к докладу, а на проверку связи — оклик «Амурская "
            "область» звучит одинаково в обоих случаях, и различить их по одной "
            "фразе нельзя."
        )
    print(
        f"Вне разбора: {_minutes(outside.seconds)}, "
        f"{_plural(len(outside.blocks), 'реплика', 'реплики', 'реплик')}. "
        "Это повестка, перекличка и общие доклады — у них направления нет, "
        "и ключ для них нужен другой."
    )
    return reports


def report_against_noted(
    reports: Sequence[Report],
    noted: Sequence[tuple[str, str]],
    units: Directory,
) -> None:
    """Сверяет отрезки с эталонным разделом «Отметили»."""
    print()
    wanted: dict[str, list[str]] = {}
    unkeyed: list[str] = []
    for number, text in noted:
        names = named_in(text, units)
        if not names:
            unkeyed.append(number)
        for name in names:
            wanted.setdefault(name, []).append(number)

    print(
        f"Эталон «Отметили»: {_plural(len(noted), 'пункт', 'пункта', 'пунктов')}, "
        f"направление узнаётся в {len(noted) - len(unkeyed)}, "
        f"направлений названо {len(wanted)}"
    )
    if unkeyed:
        print(
            f"Без направления — пункты {', '.join(unkeyed)}: доклад не по кругу, "
            "а об общем. Такой пункт по региону не соберётся никогда, и это не "
            "промах разбора, а другой ключ."
        )

    found = [name for name in wanted if any(s.unit == name for s in reports)]
    missed = [name for name in wanted if name not in found]
    extra = [s for s in reports if s.unit not in wanted]

    print()
    # Случайный уровень: если назвать столько же направлений наугад из
    # справочника, столько эталонных в среднем и накроешь. Без этой строки
    # любая полнота выглядит достижением.
    chance = _share(len(reports), len(units))
    print(
        f"Полнота по направлениям: {len(found)} из {len(wanted)} "
        f"({_share(len(found), len(wanted))}) при случайном уровне {chance}"
    )
    if missed:
        print(f"Не нашлось: {', '.join(missed)}")
        print(
            "Пропуск здесь значит и то, что направление названо в чужом докладе, "
            "а своей очереди у него не было: мониторинг и статистика поминают "
            "регионы, которые на связь не выходили."
        )
    if extra:
        names = ", ".join(f"{s.unit} ({_minutes(s.seconds)})" for s in extra)
        print(f"Лишние отрезки: {names}")
    if not missed and not extra:
        print("Отрезки и пункты эталона совпали один в один.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Отрезки записи под раздел «Отметили» и сверка их с эталоном",
    )
    parser.add_argument("transcript", type=Path, help="стенограмма: json или txt")
    parser.add_argument(
        "--directory", type=Path, required=True,
        help="справочник направлений: чей вопрос разбирают по кругу",
    )
    parser.add_argument(
        "--noted", type=Path, default=None,
        help="эталонный раздел «Отметили» нумерованным списком",
    )
    parser.add_argument(
        "--min-seconds", type=float, default=REPORT_SECONDS,
        help=f"сколько речи делает отрезок докладом (по умолчанию {REPORT_SECONDS})",
    )
    parser.add_argument(
        "--label", default="Направление", help="как называется графа разбора",
    )
    parser.add_argument(
        "--from-min", type=float, default=None,
        help="с какой минуты записи идёт совещание (до неё — подключение)",
    )
    parser.add_argument(
        "--to-min", type=float, default=None,
        help="на какой минуте записи совещание кончается",
    )
    args = parser.parse_args(argv)

    units = read_directory(args.directory, label=args.label)
    if not units:
        print("Справочник пуст — раскладывать запись не по чему.")
        return 1

    transcript = read_blocks(args.transcript)
    blocks = transcript.blocks
    outside = 0
    if args.from_min is not None or args.to_min is not None:
        blocks, outside = within(
            blocks,
            since=args.from_min * 60 if args.from_min is not None else None,
            until=args.to_min * 60 if args.to_min is not None else None,
        )
    blocks, dropped = drop_soundcheck(blocks, keep=asks_for_work)
    print(
        f"Стенограмма: {_plural(len(transcript.blocks), 'реплика', 'реплики', 'реплик')}, "
        f"голосов {len(transcript.speakers)}, "
        f"длительность {transcript.duration_min} мин, "
        f"перекличкой убрано {dropped}"
        + (f", вне отрезка совещания {outside}" if outside else "")
    )
    print(f"Справочник: {_plural(len(units), 'направление', 'направления', 'направлений')}")
    print()

    # Порог здесь нулевой нарочно: короткие отрезки приложение убирает само,
    # а отчёт должен их показать — по ним видно, где разбор оборвался.
    spans, outside = split_into_reports(blocks, units, floor=0)
    if not spans:
        print(
            "Ни одного направления не объявлено. Либо запись не о разборе по "
            "кругу, либо справочник не о ней."
        )
        return 1
    reports = report_layout(spans, outside, units, floor=args.min_seconds)

    if args.noted:
        report_against_noted(reports, read_noted(args.noted), units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
