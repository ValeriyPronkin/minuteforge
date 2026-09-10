"""Направления: чей вопрос разбирают, по закрытому списку из файла."""

from minuteforge.blocks import Block
from minuteforge.directory import (
    DEFAULT_LABEL,
    EMPTY,
    Directory,
    Entry,
    at,
    read_directory,
)


def directory(label: str = DEFAULT_LABEL) -> Directory:
    """Справочник для проверок: никакой предметной области, одни выдумки."""
    return Directory(
        entries=(
            Entry("Северный филиал", ("северн",)),
            Entry("Заречный филиал", ("заречн",)),
            Entry("Восточная площадка", ("восточн",)),
            Entry("Садовый участок", ("сад",)),
        ),
        label=label,
    )


def test_a_name_is_recognised_through_its_cases():
    """«Северного», «Северному», «Северном» — одно направление, а
    морфологии для этого не нужно."""
    units = directory()
    assert units.find("Следующий филиал Северный") == "Северный филиал"
    assert units.find("Коллеги Заречного филиала, просьба подтвердить") == "Заречный филиал"
    assert units.find("Далее Восточная площадка на примере цеха") == "Восточная площадка"


def test_a_mangled_name_is_not_a_unit():
    """Распознавание коверкает названия, и это главный довод за список.

    Неверный адресат хуже пустого: пустой заставляет уточнить перед
    рассылкой, неверный уходит в рассылку как есть.
    """
    assert directory().find("Следующий филиал Сивернай") == ""
    assert directory().find("Далее Заподная площадка") == ""


def test_a_name_is_matched_from_the_start_of_a_word():
    """Совпадение искали подстрокой где угодно внутри слова, и в графе
    появлялось направление, которого в разговоре не было."""
    assert directory().find("Это очень досадно") == ""
    assert directory().find("Пересеверный вариант нам не подходит") == ""


def test_a_short_name_needs_the_end_of_the_word_too():
    """У короткой основы хвост должен быть длиной с падежное окончание:
    граница слова есть и у «сада», и у «садитесь»."""
    units = directory()
    assert units.find("Сад на связи") == "Садовый участок"
    assert units.find("По саду замечаний нет") == "Садовый участок"
    assert units.find("Садитесь, пожалуйста") == ""


def test_the_unit_holds_until_the_next_one():
    """Объявляют один раз, а поручают потом четверть часа."""
    marks = directory().follow([
        Block("SPEAKER_02", "Следующий филиал Северный.", 100, 120),
        Block("SPEAKER_08", "Готовность 84%. Просьба подтвердить срок.", 200, 260),
        Block("SPEAKER_02", "Следующий филиал Заречный.", 400, 420),
    ])

    assert at(marks, 250) == "Северный филиал"
    assert at(marks, 450) == "Заречный филиал"
    assert at(marks, 50) == "", "до первого перехода адресата нет"


def test_the_name_is_given_in_the_next_phrase():
    """Объявляют и называют порознь: «Следующая площадка.» — и уже
    следующей фразой «Иван Иванович, Восточная». Пока смотрели только фразу
    перехода, начало совещания оставалось без направления."""
    marks = directory().follow([
        Block("SPEAKER_02", "Хорошо. Следующая площадка.", 100, 120),
        Block("SPEAKER_12", "Иван Иванович, Восточная. Коротко доложу.", 130, 200),
    ])

    assert at(marks, 150) == "Восточная площадка"


def test_an_unreadable_announcement_clears_the_unit():
    """Объявление, в котором названия не разобрать, — тоже переход.
    Оставить прежнее значило бы подписать чужие поручения тому, кого уже
    разобрали."""
    marks = directory().follow([
        Block("SPEAKER_02", "Следующий филиал Северный.", 100, 120),
        Block("SPEAKER_02", "Следующая площадка Заподная.", 300, 320),
        Block("SPEAKER_02", "Организуйте сейчас фотоотчёт.", 400, 420),
    ])

    assert at(marks, 200) == "Северный филиал"
    assert at(marks, 400) == ""


def test_a_quiet_move_counts_only_when_the_name_is_given():
    """«Далее Восточная площадка» — переход. «И так далее» — конец
    перечисления, и по нему разбор уходил туда, что упомянуто ниже
    мимоходом."""
    marks = directory().follow([
        Block("SPEAKER_12", "Далее Восточная площадка на примере цеха.", 100, 130),
        Block("SPEAKER_02", "Рассказывает стройнадзор и так далее.", 200, 230),
        Block("SPEAKER_02", "Как коллеги с Заречного заявляют, объект ведём.", 240, 280),
    ])

    assert at(marks, 250) == "Восточная площадка"


def test_moving_to_a_slide_does_not_clear_the_unit():
    """«Переходим к фотоматериалу» сказано посреди разбора, и стирать по
    нему нельзя: направление не менялось."""
    marks = directory().follow([
        Block("SPEAKER_12", "Далее Восточная площадка на примере цеха.", 100, 130),
        Block("SPEAKER_12", "Переходим к фотоматериалу. Видим навалы.", 200, 260),
    ])

    assert at(marks, 250) == "Восточная площадка"


def test_counting_your_own_questions_is_not_a_move():
    """«Следующий вопрос уже прозвучал» — ведущий считает свои вопросы к
    тому же самому направлению, а не уходит с него."""
    marks = directory().follow([
        Block("SPEAKER_02", "Следующий филиал Северный.", 100, 130),
        Block("SPEAKER_02", "Следующий вопрос уже прозвучал. Финансирование есть?", 200, 260),
    ])

    assert at(marks, 250) == "Северный филиал"


def test_moving_to_another_item_of_the_agenda_clears_the_unit():
    """А переход к следующему вопросу повестки — конец разбора. Иначе
    поручения общей части достаются тому, кого разбирали до неё."""
    marks = directory().follow([
        Block("SPEAKER_02", "Следующий филиал Северный.", 100, 130),
        Block("SPEAKER_02", "Переходим к следующему вопросу, уважаемые коллеги.", 300, 330),
        Block("SPEAKER_11", "Прошу обратить внимание на нормативные акты.", 400, 430),
    ])

    assert at(marks, 200) == "Северный филиал"
    assert at(marks, 420) == "", "общая часть — не Северный филиал"


def test_an_announcement_inside_a_long_reply_counts_from_its_place():
    """Реплика ведущего бывает в две минуты, и объявление в её конце,
    отнесённое к началу, отдавало следующему всё, что поручено в этой же
    реплике предыдущему."""
    marks = directory().follow([
        Block("SPEAKER_02", "Следующий филиал Северный.", 0, 10),
        Block(
            "SPEAKER_02",
            "Дату ввода объекта назовите. " + "Разбираемся дальше. " * 40
            + "Следующий филиал Заречный.",
            100, 200,
        ),
    ])

    assert at(marks, 110) == "Северный филиал", "начало реплики — ещё прежнее"
    assert at(marks, 199) == "Заречный филиал"


def test_an_address_is_told_apart_from_a_mention():
    """«Коллеги Заречного филиала, просьба подтвердить» — обращение, а
    «Заречный филиал поддерживает предложение» — упоминание в докладе."""
    units = directory()
    assert units.starts_with("Коллеги Заречного филиала, просьба подтвердить") == "Заречный филиал"
    assert units.starts_with("Администрация Восточной площадки, дайте оценку") == (
        "Восточная площадка"
    )
    assert units.starts_with("Отмечу, что Заречный филиал выполнил план") == ""


def test_without_a_directory_nothing_is_invented():
    """Нет файла — графа пустая. Услышанное название не берётся: его
    коверкает распознавание, а неверный адресат хуже пустого."""
    assert not EMPTY
    assert EMPTY.find("Далее Восточная площадка") == ""
    assert EMPTY.follow([Block("S", "Следующая площадка Восточная.", 0, 10)]) == []


def test_the_directory_is_read_from_a_file(tmp_path):
    """Файл правят в Excel: точка с запятой, заголовок, запятые внутри."""
    file = tmp_path / "справочник.csv"
    file.write_text(
        "Направление;Как звучит\n"
        "Северный филиал;северн, Дальний цех\n"
        "Заречный филиал;южн\n",
        encoding="utf-8-sig",
    )
    units = read_directory(file, label="Площадка")

    assert len(units) == 2
    assert units.label == "Площадка"
    assert units.find("Далее Северный филиал") == "Северный филиал"
    assert units.find("Проверим Дальний цех") == "Северный филиал", "синоним — то же самое"


def test_a_directory_without_a_second_column_still_works(tmp_path):
    """Как звучит — необязательно: без него узнаётся само название."""
    file = tmp_path / "справочник.csv"
    file.write_text("Северный филиал\nЗаречный филиал\n", encoding="utf-8")

    assert read_directory(file).find("Далее Северный филиал") == "Северный филиал"


def test_a_missing_directory_does_not_stop_the_work(tmp_path):
    """Протокол собирается и без графы: падать из-за справочника нельзя."""
    assert not read_directory(tmp_path / "нет-такого.csv")
    assert not read_directory(None)
    assert not read_directory("")


def test_the_third_column_says_how_to_address(tmp_path):
    """«Ставропольский край» → «Правительству Ставропольского края»: по-русски
    это другой падеж, а падежей инструмент не знает."""
    file = tmp_path / "справочник.csv"
    file.write_text(
        "Направление;Как звучит;Кому адресовать\n"
        "Первая площадка;первая площадк, северный цех;Руководству первой площадки\n"
        "Второй участок;второй участ\n",
        encoding="utf-8",
    )
    units = read_directory(file)

    assert units.addressees() == {"Первая площадка": "Руководству первой площадки"}
    assert units.find("Северный цех отчитался") == "Первая площадка", "созвучия целы"
