"""Выжимка из протокола для секретаря: что показать и что спросить.

    python scripts/sheet_for_secretary.py протокол.md > лист.md

Показывать весь протокол бессмысленно: сто с лишним тезисов по тринадцати
направлениям никто читать не будет, а на «ну, надо дорабатывать» решения не
примешь. Нужна выжимка — и вопросы, на которые отвечают да или нет.

Вопросы здесь про материал, а не про слог, и это принципиально. Формулировку
инструмент человеку не навязывает: подписанный протокол переписан в
канцелярскую форму рукой секретаря, и подгонять разбор под неё значит
переобучаться под одного делопроизводителя. Спрашивать надо о другом — хватает
ли материала, нет ли пропущенного, не дороже ли вычитка, чем работа с нуля.

Направления берутся разной длины: самое длинное, самое короткое и два между
ними. Длинное показывает главную беду — двадцать строк на доклад там, где в
подписанном протоколе их четыре; короткое показывает, хватает ли материала,
когда доклад был на три минуты.

Лист получается с настоящими фамилиями и названиями организаций. В репозиторий
ему не место, и никуда, кроме как коллеге, он не идёт.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

QUESTIONS = """
## Что мы просим оценить

Это черновик, собранный из записи совещания без участия человека. Оценить
надо **материал, а не слог**: формулировки мы сознательно оставляем секретарю,
потому что канцелярская форма у каждого своя.

Вопросы по разделу «Отметили»:

1. **Хватило бы этого, чтобы написать пункт, не пересматривая запись?**
   По каждому из четырёх — да или нет. Если нет, чего именно не хватает.

2. **Есть ли строки, которых в протоколе быть не должно вообще?**
   Не «неудачно сказано», а «этого сюда не пишут».

3. **Чего не хватает?** Что вы бы внесли обязательно, а здесь этого нет.
   Пропущенное в готовом документе не видно никак, и это самый дорогой вопрос.

4. **Сколько строк на один доклад нужно?** Здесь их от четырёх до двадцати.
   Двадцать — это полезное сырьё или мусор, который дольше вычёркивать, чем
   написать пункт заново?

5. **Должность и ФИО докладчика** — вы дописываете их по памяти, по списку
   участников, или без них пункт недействителен?

Вопросы по разделу «Решили»:

6. **Сколько строк вы бы вычеркнули?** Примерно каждая третья у нас лишняя.
   Дорого ли это при вычитке или терпимо?

7. **Пункты без исполнителя** (раздел «Требуют уточнения») — их правда
   приходится выяснять, или адресат обычно понятен из хода совещания?

И один общий:

8. **Сколько времени ушло бы на доводку такого черновика до документа —
   и сколько уходит сейчас, когда его пишут с нуля?**
"""


def sections(text: str, title: str) -> list[str]:
    """Строки одного раздела разметки, до следующего заголовка того же уровня."""
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines)
         if line.startswith("## ") and title.lower() in line.lower()),
        None,
    )
    if start is None:
        return []
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return lines[start + 1:end]


def notes_by_unit(text: str) -> dict[str, list[str]]:
    """Пункты «Отметили»: заголовок пункта и строки под ним."""
    found: dict[str, list[str]] = {}
    current = ""
    for line in sections(text, "Отметили"):
        head = re.match(r"^\d+\.\s+(\*\*.+)$", line.strip())
        if head:
            current = head.group(1)
            found[current] = []
        elif current and line.strip().startswith("- "):
            found[current].append(line.strip())
    return found


def spread(notes: dict[str, list[str]], count: int = 4) -> list[str]:
    """Направления разной длины: самое длинное, самое короткое и середина."""
    order = sorted(notes, key=lambda name: len(notes[name]))
    if len(order) <= count:
        return order
    step = (len(order) - 1) / (count - 1)
    return [order[round(position * step)] for position in range(count)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Выжимка из протокола для секретаря")
    parser.add_argument("protocol", type=Path, help="протокол.md")
    parser.add_argument("--units", nargs="*", default=None, help="какие направления взять")
    parser.add_argument("--decisions", type=int, default=6, help="сколько пунктов «Решили»")
    parser.add_argument("--unclear", type=int, default=4, help="сколько пунктов без исполнителя")
    args = parser.parse_args(argv)

    text = args.protocol.read_text(encoding="utf-8-sig")
    notes = notes_by_unit(text)
    if not notes:
        print("В протоколе нет раздела «Отметили».", file=sys.stderr)
        return 1

    chosen = args.units or spread(notes)
    picked = [name for name in notes if any(unit in name for unit in chosen)]

    out = ["# Черновик протокола, собранный из записи", ""]
    out.append(
        "Ниже — часть протокола совещания, собранная программой по видеозаписи. "
        "Человек к ней не прикасался: ни одна строка не написана и не поправлена "
        "рукой."
    )
    out.append("")
    # Шапка целиком, до первого раздела: там дата, председатель и список
    # участников. Обрывать её нельзя — «**Участники:**» без самого списка
    # выглядит потерянным текстом.
    head = []
    for line in text.splitlines()[1:]:
        if line.startswith("## "):
            break
        head.append(line)
    out.extend(line for line in head if line.strip())
    out.append("")
    out.append(
        f"Показаны {len(picked)} пункта раздела «Отметили» из {len(notes)} — "
        "разной длины, от самого короткого доклада до самого длинного."
    )
    out.extend(["", "## Отметили", ""])
    for number, name in enumerate(picked, 1):
        out.append(f"{number}. {name}")
        out.extend(f"    {line}" for line in notes[name])
        out.append("")

    # «Решили» — раздел документа, «Поручения» — как он назывался в прогонах
    # до того, как встроенная разметка стала собирать документ, а не таблицу.
    # Старые прогоны показывают тоже: сравнивать их придётся ещё долго.
    decisions = [
        line for line in (sections(text, "Решили") or sections(text, "Поручения"))
        if line.strip()
    ]
    stop = next((i for i, line in enumerate(decisions) if line.startswith("###")), len(decisions))
    out.extend(["## Решили", ""])
    kept = 0
    for line in decisions[:stop]:
        row = line.startswith("|") and not set(line) <= set("|-: ")
        point = bool(re.match(r"^\d+\.\s", line))
        if point or (row and not line.startswith("| №")):
            kept += 1
            if kept > args.decisions:
                break
            if point:
                out.append("")
        out.append(line)
    out.append("")

    unclear = [line for line in decisions[stop:] if line.strip().startswith("- ")]
    if unclear:
        out.extend(["## Требуют уточнения", "",
                    "Поручения, у которых не назван ни исполнитель, ни регион:", ""])
        out.extend(unclear[:args.unclear])
        out.append("")

    out.append(QUESTIONS.strip())
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
