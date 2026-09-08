"""Сроки: от сказанного вслух к дате в календаре."""

from datetime import date

from minuteforge.dates import as_text, parse_meeting_date, resolve

MEETING = date(2026, 9, 8)  # вторник


def test_the_meeting_date_is_read_the_way_a_person_writes_it():
    """Поле свободное: его заполняет делопроизводитель, а не программа."""
    assert parse_meeting_date("08.09.2026, 11:00") == MEETING
    assert parse_meeting_date("8 сентября 2026") == MEETING
    assert parse_meeting_date("08.09.26") == MEETING
    assert parse_meeting_date("") is None
    assert parse_meeting_date("вторник") is None, "не разобрали — значит нет"


def test_relative_deadlines_count_from_the_meeting():
    assert resolve("через две недели", MEETING) == date(2026, 9, 22)
    assert resolve("в течение ближайших двух недель", MEETING) == date(2026, 9, 22)
    assert resolve("через 20 дней", MEETING) == date(2026, 9, 28)
    assert resolve("через месяц", MEETING) == date(2026, 10, 8)
    assert resolve("сегодня же", MEETING) == MEETING
    assert resolve("завтра", MEETING) == date(2026, 9, 9)


def test_the_end_of_something_is_a_date_too():
    assert resolve("до конца года", MEETING) == date(2026, 12, 31)
    assert resolve("до конца месяца", MEETING) == date(2026, 9, 30)
    assert resolve("до конца квартала", MEETING) == date(2026, 9, 30)
    # По-деловому конец недели — пятница: в субботу спрашивать не с кого.
    assert resolve("до конца недели", MEETING) == date(2026, 9, 11)


def test_a_named_day_keeps_its_year():
    assert resolve("на 30 ноября", MEETING) == date(2026, 11, 30)
    assert resolve("до 15.10", MEETING) == date(2026, 10, 15)
    # Названный месяц уже прошёл — значит, речь о следующем годе.
    assert resolve("к 1 марта", MEETING) == date(2027, 3, 1)


def test_a_weekday_is_the_next_one():
    """«До пятницы», сказанное в пятницу, — это следующая пятница: срок,
    истекающий в момент назначения, смысла не имеет."""
    assert resolve("до пятницы", MEETING) == date(2026, 9, 11)
    assert resolve("до вторника", MEETING) == date(2026, 9, 15)


def test_what_is_not_a_date_stays_without_one():
    """«Еженедельно» и «постоянно» — порядок работы, а не день."""
    assert resolve("еженедельно", MEETING) is None
    assert resolve("постоянно", MEETING) is None
    assert resolve("через две недели", None) is None, "не от чего считать"
    assert as_text(None) == ""
