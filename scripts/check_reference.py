"""Сверка эталона со стенограммой: что в нём звучало, а что дописано.

    python scripts/check_reference.py эталон_поручения.csv стенограмма.txt

Эталон — не истина, а работа человека. Секретарь дописывает понятое из
контекста, сглаживает формулировки, а иногда вносит пункт, которого вслух не
было: он знает, о чём договорились до заседания, и знает, что в документе
этот пункт нужен.

Знать долю такого обязательно, и вот почему. Если пятая часть эталона в
записи не звучала, то полнота 100 % недостижима в принципе — гнаться за ней
значит учить инструмент выдумывать ровно то, за что мы его ругаем.

Считается доля значащих слов пункта, которые нашлись **в одном месте
записи**, — а не по всей стенограмме. Разница решающая: в трёхчасовом
совещании найдётся почти любое слово, и сверка с корпусом целиком показывает
100 % для чего угодно, включая выдуманное. Настоящее поручение звучит в
одном месте, и там же лежат почти все его слова.

Место — реплика с соседними: поручение и его предмет часто разнесены на две
фразы, как и в самом разборе.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.tasks import _significant  # noqa: E402

#: Ниже этой доли пункт считается дописанным: слов из него в записи нет.
HEARD = 0.6

#: Между «звучало» и «дописано» лежит полоса, где человек переформулировал
#: сильно, но по делу. Её показываем отдельно, а не приписываем ни к чему.
DOUBTFUL = 0.4


def read_tasks(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    return [
        {
            "time": (row.get("Время") or "").strip(),
            "what": (row.get("Поручение") or "").strip(),
            "unit": (row.get("Регион") or "").strip(),
        }
        for row in rows
    ]


#: Сколько реплик подряд считается одним местом записи. Поручение и его
#: предмет разнесены на две фразы чаще, чем стоят в одной.
NEARBY = 3


def read_windows(path: str | Path) -> list[set[str]]:
    """Стенограмма местами: каждое — реплика с соседними."""
    text = Path(path).read_text(encoding="utf-8-sig")
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", " ", text)
    text = re.sub(r"\b(SPEAKER[_ ]?\d+|UNKNOWN)\b", " ", text)
    lines = [line for line in text.splitlines() if line.strip()]
    return [
        _significant(" ".join(lines[start:start + NEARBY]))
        for start in range(max(len(lines) - NEARBY + 1, 1))
    ]


def heard_share(what: str, places: list[set[str]]) -> float:
    """Насколько полно пункт звучит в лучшем для него месте записи."""
    words = _significant(what)
    if not words:
        return 0.0
    return max((len(words & place) / len(words) for place in places), default=0.0)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("нужно: эталон_поручения.csv стенограмма.txt")
        return 2

    tasks = read_tasks(argv[1])
    places = read_windows(argv[2])

    scored = [(heard_share(task["what"], places), task) for task in tasks]
    heard = [item for item in scored if item[0] >= HEARD]
    doubtful = [item for item in scored if DOUBTFUL <= item[0] < HEARD]
    added = [item for item in scored if item[0] < DOUBTFUL]

    total = len(tasks) or 1
    print(f"эталон: {len(tasks)} пунктов, мест в стенограмме: {len(places)}")
    print(f"  звучало почти дословно: {len(heard)} ({len(heard)/total:.0%})")
    print(f"  сильно переформулировано: {len(doubtful)} ({len(doubtful)/total:.0%})")
    print(f"  в записи не найдено: {len(added)} ({len(added)/total:.0%})")

    for title, items in (("ПЕРЕФОРМУЛИРОВАНО", doubtful), ("НЕ НАЙДЕНО В ЗАПИСИ", added)):
        if not items:
            continue
        print(f"\n--- {title} ---")
        for share, task in sorted(items, key=lambda item: item[0]):
            place = f"{task['time']} " if task["time"] else ""
            print(f"  {share:.0%}  {place}{task['what'][:88]}")

    print(
        "\nПотолок полноты для разбора: "
        f"{(len(heard) + len(doubtful)) / total:.0%} — выше него можно подняться, "
        "только выдумывая."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
