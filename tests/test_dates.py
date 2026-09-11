"""Сроки: от сказанного вслух к дате в календаре."""

from datetime import date, datetime, timezone

from minuteforge.dates import (
    Recorded,
    as_text,
    date_from_name,
    parse_meeting_date,
    resolve,
)

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


def test_the_date_of_the_meeting_is_read_from_the_recording_name():
    """Имя файла пишет человек или программа записи, и дата в нём почти
    всегда есть. Это надёжнее свойств файла: имя переживает копирование."""
    assert date_from_name("Совещание 07.09.2026.mp4").day == date(2026, 9, 7)
    assert date_from_name("2026-09-07 10-30 штаб.mkv").as_text() == "07.09.2026, 10:30"
    assert date_from_name("штаб_2026.09.07_1030.mp4").as_text() == "07.09.2026, 10:30"
    assert date_from_name("20260907.mp4").as_text() == "07.09.2026"


def test_a_number_that_is_not_a_date_is_not_read_as_one():
    """Размер кадра и номер части — не дата совещания."""
    assert date_from_name("Recording_1920x1080.mp4") is None
    assert date_from_name("запись.mp4") is None
    assert date_from_name("") is None
    # Тридцать четвёртого числа не бывает: разбор идёт дальше, а не выдаёт
    # ближайшее похожее.
    assert date_from_name("20261234.mp4") is None


def test_zoom_names_its_recordings_by_greenwich():
    """«GMT20260907-070012» — это десять утра по Москве. Оставить как есть
    значило бы вписать в протокол время, которого на совещании не было."""
    found = date_from_name("GMT20260907-070012_Recording_1920x1080.mp4")
    # В какое именно время это превратится, решает часовой пояс машины —
    # той самой, где сидит делопроизводитель.
    expected = datetime(2026, 9, 7, 7, 0, tzinfo=timezone.utc).astimezone()
    assert found.as_text() == f"{expected:%d.%m.%Y}, {expected:%H:%M}"


def test_the_time_is_kept_apart_from_the_day():
    """Час, которого никто не называл, приписывать нельзя."""
    assert Recorded(date(2026, 9, 7)).as_text() == "07.09.2026"
    assert Recorded(date(2026, 9, 7), "11:00").as_text() == "07.09.2026, 11:00"
    # То, что подставлено в поле, должно разбираться обратно: иначе сроки
    # посчитаются не от той даты.
    assert parse_meeting_date(Recorded(date(2026, 9, 7), "11:00").as_text()) == date(2026, 9, 7)

def test_a_range_of_days_is_not_a_date():
    """«Сделать к 7-8» — это седьмое-восьмое число, а числовой разбор читает
    это как 7.08. На записи 27 августа такой срок вышел на три недели раньше
    самого совещания."""
    from datetime import date

    from minuteforge.dates import resolve

    assert resolve("к 7-8", date(2026, 8, 27)) is None


def test_a_numeric_date_behind_the_meeting_is_not_a_deadline():
    """Сроком назад не назначают — и для чисел это так же, как для слов.
    Раньше защита стояла только на «до 15 октября», а «до 15.10.2025»
    проходило насквозь."""
    from datetime import date

    from minuteforge.dates import resolve

    assert resolve("до 15.10.2025", date(2026, 8, 27)) is None
    assert resolve("до 15.10", date(2026, 8, 27)) == date(2026, 10, 15)
