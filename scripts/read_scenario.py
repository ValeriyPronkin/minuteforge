"""Что видно в сценарии заседания.

    python scripts/read_scenario.py сценарий.docx --directory справочник.csv

К заседанию готовят технический сценарий — кто когда берёт слово. В нём
лежит разом то, что мы собирали руками по трём файлам: выступающие с
должностями, привязка каждого к своему направлению, порядок разбора и имя
ведущего.

Скрипт показывает, что из документа вычиталось, — до того, как это уедет в
прогон. Сценарий пишут люди, и пишут по-разному: где-то тире, где-то нет,
где-то фамилия впереди должности. Посмотреть глазами дешевле, чем потом
разбираться, почему в шапке протокола не тот человек.

Файлы с материалами совещаний в репозиторий не попадают: путь передаётся
аргументом.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.directory import read_directory  # noqa: E402
from minuteforge.scenario import read_scenario  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Что видно в сценарии заседания")
    parser.add_argument("scenario", type=Path, help="сценарий: docx или текст")
    parser.add_argument(
        "--directory", type=Path, default=None,
        help="справочник направлений — без него названия не опознаются",
    )
    args = parser.parse_args(argv)

    units = read_directory(args.directory) if args.directory else None
    found = read_scenario(args.scenario, units)
    if not found:
        print(
            "Из сценария ничего не вычиталось. Либо это не сценарий, либо "
            "написан он иначе, чем те, на которых правило проверено."
        )
        return 1

    print(f"Ведёт: {found.chair or '— не разобрать'}")
    print()
    if found.units:
        print(f"Разбор по {len(found.units)} направлениям, в этом порядке:")
        for number, name in enumerate(found.units, 1):
            print(f"  {number:>2}. {name}")
    elif units is None:
        print("Направления не опознаны: не задан справочник (--directory).")
    print()

    print(f"Выступающих: {len(found.people)}")
    for person in found.people:
        where = person.org or "— общий доклад"
        print(f"  {person.name:<34} {where:<28} {person.position[:60]}")
    print()
    print(
        "Всё это подсказка, а не решение: сценарий может и не сбыться — "
        "кто-то не выйдет на связь, порядок переставят. Человек смотрит и "
        "правит."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
