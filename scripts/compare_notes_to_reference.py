"""Сверка раздела «Отметили» с эталоном: по направлениям и по числам.

    python scripts/compare_notes_to_reference.py протокол.md эталон.md \
        --directory справочник.csv --transcript стенограмма.json

Полнота раздела меряется не так, как полнота поручений. Поручение ищут по
тексту: оно звучит одной фразой, и совпадение двух формулировок — вопрос
похожести. Пункт «Отметили» привязан к направлению, и совпадение здесь
объективное: доклад по этому направлению в протоколе либо есть, либо нет.

Считаются две величины, и путать их нельзя:

*   **направления** — по скольким эталонным докладам у нас есть пункт. Это
    полнота раздела, и она отвечает на вопрос «не потеряли ли доклад целиком».
*   **числа** — сколько эталонных цифр повторено в наших тезисах. Это точность
    по существу: в протоколе такого рода спорят словами, а цифру переносят в
    отчёт как есть, и выдуманная цифра дороже неловкой формулировки.

Второе считается вместе с потолком. Часть чисел подписанного протокола в
записи не звучит вовсе — секретарь взял их из документов, — и гнаться за ними
значит выдумывать. Потолок печатается рядом, иначе полнота льстит.

Эталоном годится и готовый список пунктов, и подписанный протокол целиком: в
нём находится раздел с заголовком «Отметили» и читается только он.

Файлы с записями совещаний в репозиторий не попадают: пути передаются
аргументами, ничего не читается по умолчанию.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.directory import Directory, read_directory  # noqa: E402
# Числа словами читаются тем же разбором, каким их читает проверка тезисов:
# считать «полтора» двумя способами значит мерить не то, что работает.
from minuteforge.notes import _as_number, _spoken_numbers  # noqa: E402

#: Пункт нумерованного списка: «12. Информацию такого-то…».
_ITEM = re.compile(r"^(\d{1,3})[.)]\s+(.*)$")

#: Число в тексте: 93,5 · 42 · 2026 · 15.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def read_section(path: Path, word: str = "отметили") -> list[tuple[str, str]]:
    """Нумерованные пункты раздела. Нет такого заголовка — читается весь файл."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    heading = next(
        (i for i, line in enumerate(lines)
         if line.startswith("#") and word in line.lower()),
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
        found = _ITEM.match(line)
        if found:
            numbers.append(found.group(1))
            items.append([found.group(2)])
        elif items:
            items[-1].append(line)
    return [(number, " ".join(parts)) for number, parts in zip(numbers, items)]


def read_ours(path: Path) -> dict[str, list[str]]:
    """Наш раздел «Отметили»: направление и тезисы под ним."""
    ours: dict[str, list[str]] = {}
    current = ""
    inside = False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            if "отметили" in line.lower():
                inside = True
                continue
            if inside:
                break
        if not inside:
            continue
        found = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", line)
        if found:
            current = found.group(1)
            ours[current] = []
        elif current and line.startswith("- "):
            ours[current].append(line[2:])
    return ours


def numbers_in(text: str) -> set[str]:
    """Числа текста — и цифрами, и словами."""
    return {_as_number(found) for found in _NUMBER.findall(text or "")} | _spoken_numbers(text or "")


def heard_numbers(path: Path | None) -> set[str] | None:
    """Числа, звучавшие в записи. Нет стенограммы — потолок не считается."""
    if not path:
        return None
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    segments = body.get("segments", []) if isinstance(body, dict) else body
    return numbers_in(" ".join(str(part.get("text", "")) for part in segments))


def report(
    ours: dict[str, list[str]],
    reference: Sequence[tuple[str, str]],
    units: Directory,
    heard: set[str] | None,
) -> None:
    wanted: dict[str, str] = {}
    general: list[str] = []
    for number, text in reference:
        named = units.find_all(text)
        # Пункт, в котором названо одно направление, — доклад своей очереди.
        # Названо несколько — это общий доклад, где перечисляют всех подряд,
        # и по кругу он не собирается никогда.
        if len(named) == 1:
            wanted.setdefault(named[0], text)
        else:
            general.append(number)

    print(
        f"Эталон: пунктов {len(reference)}, из них своей очереди {len(wanted)}. "
        f"Наш раздел: пунктов {len(ours)}, тезисов {sum(len(v) for v in ours.values())}"
    )
    if general:
        print(
            f"Не по кругу — пункты {', '.join(general)}: доклад об общем, и "
            "направления у него нет. По региону такой пункт не собрать."
        )
    print()

    print(f"{'направление':<32}{'тезисов':>9}{'чисел':>7}{'звучало':>9}{'нашлось':>9}")
    got = reachable = total = 0
    missing: list[str] = []
    for name, text in wanted.items():
        said = " ".join(ours.get(name, []))
        want = numbers_in(text)
        here = want & numbers_in(said)
        could = want if heard is None else want & heard
        got += len(here)
        reachable += len(could)
        total += len(want)
        if name not in ours:
            missing.append(name)
        print(
            f"{name[:31]:<32}{len(ours.get(name, [])):>9}{len(want):>7}"
            f"{len(could):>9}{len(here):>9}"
            f"{'   — пункта нет' if name not in ours else ''}"
        )

    print()
    print(f"Направлений: {len(wanted) - len(missing)} из {len(wanted)}")
    if missing:
        print(f"Не собрались: {', '.join(missing)}")
    if heard is None:
        print(f"Чисел эталона повторено: {got} из {total} — потолок не считался")
        print("Дайте --transcript, иначе непонятно, за чем вообще гнаться.")
    else:
        print(
            f"Чисел эталона повторено: {got} из {reachable} звучавших "
            f"(всего в эталоне {total}; {total - reachable} не звучало в записи "
            "вовсе — их секретарь взял из документов)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Сверка раздела «Отметили» с эталоном",
    )
    parser.add_argument("protocol", type=Path, help="наш протокол.md")
    parser.add_argument("reference", type=Path, help="эталон: список пунктов или протокол")
    parser.add_argument("--directory", type=Path, required=True, help="справочник направлений")
    parser.add_argument(
        "--transcript", type=Path, default=None,
        help="стенограмма json — по ней считается потолок по числам",
    )
    args = parser.parse_args(argv)

    units = read_directory(args.directory)
    if not units:
        print("Справочник пуст — сверять не по чему.")
        return 1
    ours = read_ours(args.protocol)
    if not ours:
        print("В протоколе нет раздела «Отметили». Он собирается настройкой take_notes.")
        return 1
    report(ours, read_section(args.reference), units, heard_numbers(args.transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
