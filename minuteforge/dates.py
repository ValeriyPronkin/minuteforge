"""Сроки: от сказанного вслух к дате в календаре.

На совещании сроки называют относительно: «через две недели», «до конца
года», «сегодня же», «до пятницы». Само по себе это не срок — по нему нельзя
ни поставить напоминание, ни спросить. Сроком это становится, когда известна
дата совещания: две недели считаются от неё, а не от дня, когда
делопроизводитель открыл протокол.

Поэтому дата совещания здесь — обязательное условие, а не украшение шапки.
Нет её — не будет и вычисленных дат: выдумывать точку отсчёта нельзя, иначе
поручение получит срок, о котором никто не договаривался.

Сказанное при этом не подменяется. В протоколе остаётся «через две недели», а
дата приписывается рядом: спорить будут о сказанном, а работать по дате.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

#: Месяцы в родительном падеже — так их и произносят: «до пятнадцатого
#: ноября», «на 30 ноября».
MONTHS: dict[str, int] = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}

#: Дни недели: «до пятницы», «к среде».
WEEKDAYS: dict[str, int] = {
    "понедельник": 0, "вторник": 1, "сред": 2, "четверг": 3,
    "пятниц": 4, "суббот": 5, "воскресен": 6,
}

#: Числительные словами. Дальше двенадцати сроки не называют: «через
#: пятнадцать недель» на совещании не звучит, звучит «до конца года».
NUMERALS: dict[str, int] = {
    "один": 1, "одну": 1, "одной": 1, "два": 2, "две": 2, "двух": 2,
    "три": 3, "трёх": 3, "трех": 3, "четыре": 4, "четырёх": 4, "четырех": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10,
    "десяти": 10,
}

#: Единицы, которыми меряют срок. Слово без числительного значит одну:
#: «через месяц» — это «через один месяц».
_UNITS = ("недел", "месяц", "день", "дня", "дней", "дне", "час", "год")

_DOTTED = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?\b")
_DAY_MONTH = re.compile(r"\b(\d{1,2})\s+([а-яё]{3,})", re.IGNORECASE)
_THROUGH = re.compile(
    r"(?:через|в течение|в течении)\s+(?:ближайш\w+\s+)?([\wё]+)(?:\s+([а-яё]+))?",
    re.IGNORECASE,
)


def parse_meeting_date(text: str) -> date | None:
    """Дата совещания из того, что вписал делопроизводитель.

    Поле свободное — «05.06.2025», «5 июня 2025», «05.06.2025, 11:00», —
    потому что его заполняет человек, а не программа. Не разобрали — значит
    нет: считать сроки не от чего.
    """
    text = (text or "").strip()
    if not text:
        return None
    dotted = _DOTTED.search(text)
    if dotted:
        day, month, year = dotted.groups()
        return _make(int(day), int(month), _full_year(year))
    spelled = _DAY_MONTH.search(text)
    if spelled:
        day, word = spelled.groups()
        month = _month_of(word)
        if month:
            year = re.search(r"\b(20\d{2})\b", text)
            return _make(int(day), month, int(year.group(1)) if year else date.today().year)
    return None


def resolve(due: str, meeting: date | None) -> date | None:
    """Считает дату из того, как срок прозвучал.

    Возвращает ``None``, если срок не переводится в дату: «еженедельно» и
    «постоянно» — это порядок работы, а не день, и приписывать им дату
    значило бы сочинять.
    """
    if meeting is None:
        return None
    lowered = " ".join((due or "").lower().split())
    if not lowered:
        return None

    if "послезавтра" in lowered:
        return meeting + timedelta(days=2)
    if "завтра" in lowered:
        return meeting + timedelta(days=1)
    if "сегодня" in lowered or "немедленн" in lowered or "срочно" in lowered:
        return meeting

    if "конца года" in lowered or "конец года" in lowered:
        return date(meeting.year, 12, 31)
    if "конца месяца" in lowered or "конец месяца" in lowered:
        return _end_of_month(meeting)
    if "конца квартала" in lowered or "конец квартала" in lowered:
        return _end_of_quarter(meeting)
    if "конца недели" in lowered or "конец недели" in lowered:
        # По-деловому конец недели — пятница, а не воскресенье: в субботу
        # спрашивать не с кого.
        return _next_weekday(meeting, 4)

    exact = _exact_date(lowered, meeting)
    if exact:
        return exact

    through = _THROUGH.search(lowered)
    if through:
        count, unit = through.groups()
        if count.startswith(_UNITS) or not unit:
            # «Через месяц», «через неделю» — числительного нет, значит одна.
            count, unit = "один", count if count.startswith(_UNITS) else (unit or "")
        size = NUMERALS.get(count) or (int(count) if count.isdigit() else None)
        if size:
            if unit.startswith("недел"):
                return meeting + timedelta(weeks=size)
            if unit.startswith(("день", "дня", "дней", "дне")):
                return meeting + timedelta(days=size)
            if unit.startswith("месяц"):
                return _add_months(meeting, size)
            if unit.startswith("час"):
                return meeting

    for word, number in WEEKDAYS.items():
        if word in lowered:
            return _next_weekday(meeting, number)
    return None


def as_text(day: date | None) -> str:
    """Дата так, как её пишут в протоколе."""
    return day.strftime("%d.%m.%Y") if day else ""


def _exact_date(lowered: str, meeting: date) -> date | None:
    """Названный день: «на 30 ноября», «до 15.10», «к 1 декабря»."""
    dotted = _DOTTED.search(lowered)
    if dotted:
        day, month, year = dotted.groups()
        return _make(int(day), int(month), _full_year(year) if year else meeting.year)
    spelled = _DAY_MONTH.search(lowered)
    if spelled:
        day, word = spelled.groups()
        month = _month_of(word)
        if month:
            named = _make(int(day), month, meeting.year)
            # Названный месяц уже прошёл — значит, речь о следующем годе:
            # «до 15 января» сказанное в декабре это январь следующего.
            if named and named < meeting:
                return _make(int(day), month, meeting.year + 1)
            return named
    return None


def _month_of(word: str) -> int | None:
    lowered = word.lower()
    # Май отдельно: его основа «ма» совпадает с началом слова «март».
    if lowered.startswith("мая") or lowered.startswith("май"):
        return 5
    for stem, number in MONTHS.items():
        if stem == "ма":
            continue
        if lowered.startswith(stem):
            return number
    return None


def _make(day: int, month: int, year: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _full_year(year: str | None) -> int:
    if not year:
        return date.today().year
    number = int(year)
    return number if number > 100 else 2000 + number


def _next_weekday(start: date, weekday: int) -> date:
    """Ближайший такой день после совещания.

    «До пятницы», сказанное в пятницу, — это следующая пятница: срок,
    истекающий в момент, когда его назначили, смысла не имеет.
    """
    ahead = (weekday - start.weekday()) % 7
    return start + timedelta(days=ahead or 7)


def _end_of_month(day: date) -> date:
    return date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])


def _end_of_quarter(day: date) -> date:
    month = ((day.month - 1) // 3 + 1) * 3
    return date(day.year, month, calendar.monthrange(day.year, month)[1])


def _add_months(day: date, count: int) -> date:
    month = day.month - 1 + count
    year = day.year + month // 12
    month = month % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))
