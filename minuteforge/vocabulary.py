"""Свои слова: чем поручают у вас и что поручением у вас не считается.

Правило, отличающее поручение от доклада, держится на списке слов. Список
этот языковой, а не предметный — «прошу», «обеспечьте», «доложите» звучат
одинаково везде, — и потому лежит в коде. Но у каждой организации поверх него
есть своё.

Своё бывает двух видов. Одни слова у вас поручают, а в общем списке их нет:
«взять на контроль», «отработать», «поручается». Другие выглядят поручением,
но им не являются именно у вас: на стройке «объект сдать» — строка графика, а
не задание, у связистов «слышите» — не просьба.

Угадать это снаружи нельзя, а цена ошибки разная в обе стороны: пропущенное
поручение не попадёт в протокол, лишнее — уйдёт в рассылку. Поэтому список
приносит тот, кто ведёт протокол, обычным файлом рядом с настройками.

Файл дополняет встроенное, а не заменяет: без него всё работает как прежде.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

#: Как называют раздел, в котором перечислено то, чем поручают.
ORDERS = {"поручают", "поручение", "поручения", "требование", "требования"}

#: И раздел, в котором перечислено то, что поручением не считается.
NOT_ORDERS = {
    "не поручают", "не поручение", "не поручения", "исключения", "исключение",
}


@dataclass(frozen=True)
class Vocabulary:
    """Свои слова поверх встроенных."""

    #: Чем поручают у вас: добавляется к встроенному списку.
    orders: frozenset[str] = frozenset()
    #: Что поручением не считается: вычитается из него. Сильнее, чем
    #: добавление: слово отсюда не сделает фразу поручением, даже если оно
    #: похоже на повелительное наклонение.
    not_orders: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.orders or self.not_orders)


EMPTY = Vocabulary()


def read_vocabulary(path: str | Path | None) -> Vocabulary:
    """Читает свои слова из csv или txt.

    Две колонки: раздел и слова через запятую. Разделов два, называть их
    можно по-разному — «поручают» и «не поручают», «требования» и
    «исключения»::

        Раздел;Слова
        поручают;взять на контроль, отработать, поручается
        не поручают;слышите, сдать

    Файла нет — пустой словарь и строка в журнал. Останавливать из-за этого
    работу нельзя: встроенного списка хватает.
    """
    if not path:
        return EMPTY
    file = Path(path).expanduser()
    if not file.exists():
        logger.warning("Свой словарь слов не найден: {}", file)
        return EMPTY
    try:
        lines = [line for line in file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except OSError as exc:
        logger.warning("Свой словарь слов не прочитан: {}", exc)
        return EMPTY

    orders: set[str] = set()
    not_orders: set[str] = set()
    delimiter = "\t" if any("\t" in line for line in lines) else ";"
    for row in csv.reader(lines, delimiter=delimiter):
        cells = [cell.strip() for cell in row if cell.strip()]
        if len(cells) < 2:
            continue
        section = cells[0].lower()
        words = {
            word.strip().lower()
            for cell in cells[1:]
            for word in cell.split(",")
            if word.strip()
        }
        if section in ORDERS:
            orders |= words
        elif section in NOT_ORDERS:
            not_orders |= words

    found = Vocabulary(frozenset(orders), frozenset(not_orders))
    if found:
        logger.info(
            "Свои слова: поручают {}, не поручают {} — из {}",
            len(found.orders), len(found.not_orders), file,
        )
    else:
        logger.warning(
            "Свой словарь {} прочитан, но разделов «поручают» и «не поручают» "
            "в нём не нашлось", file,
        )
    return found


def words_of(text: Iterable[str]) -> frozenset[str]:
    """Слова из перечисления через запятую — по одному, в нижнем регистре."""
    return frozenset(
        word.strip().lower() for line in text for word in line.split(",") if word.strip()
    )
