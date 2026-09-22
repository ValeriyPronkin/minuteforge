from minuteforge.assignees import (
    COLLECTIVE_NAMES,
    Roster,
    by_roster,
    circle,
    roster_of,
    same_person,
)
from minuteforge.directory import read_directory
from minuteforge.people import Person
from minuteforge.tasks import Task


# Имена здесь выдуманы целиком: настоящим участникам совещаний в репозитории
# не место, даже в проверках.
PEOPLE = (
    Person("Семенов Алексей Валерьевич", "первый замминистра", "Минчистоты"),
    Person("Огородникова Наталья Юрьевна", "директор департамента", "ППК"),
    Person("Белянин Андрей Георгиевич", "заместитель губернатора", "Луговская область"),
    Person("Белянина Ольга Евгеньевна", "министр ЖКХ", "Луговская область"),
)

DIRECTORY = (
    "Направление;Как звучит;Кому адресовать\n"
    "Луговская область;Луговск;Правительству Луговской области\n"
    "Приморный край;Приморск;Правительству Приморного края\n"
)


def directory_of(tmp_path):
    path = tmp_path / "справочник.csv"
    path.write_text(DIRECTORY, encoding="utf-8")
    return read_directory(path)


# ------------------------------------------------------------------- круги

def test_a_circle_is_recognised_however_it_is_declined():
    assert circle("регионам") == "все регионы"
    assert circle("субъекты Российской Федерации") == "все регионы"
    assert circle("Коллеги,") == "все участники"


def test_a_circle_is_recognised_with_its_heads():
    """«Все руководители региональных штабов» — те же «региональные штабы».

    В таблице контроля это была отдельная строка, и по ней спрашивали
    порознь с той, где стояло короткое название.
    """
    assert circle("все руководители региональных штабов") == "региональные штабы"
    assert circle("всем регионам") == "все регионы"


def test_a_turn_of_speech_is_not_a_circle():
    """«Участники форума» — оборот из доклада, а не круг исполнителей.

    От «участников совещания» отличается одним словом, и по созвучию
    поручение ушло бы всему залу.
    """
    assert circle("Участники форума") == ""
    assert circle("руководитель аппарата") == ""


# -------------------------------------------------------------------- опись

def test_a_person_is_brought_to_the_registry_spelling(tmp_path):
    """Одного человека называют то полным именем, то именем с отчеством."""
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    assert roster.canonical("Андрей Георгиевич") == "Белянин Андрей Георгиевич"
    assert roster.canonical("Семенов Алексей") == "Семенов Алексей Валерьевич"


def test_a_registry_line_read_as_a_name_is_cleaned(tmp_path):
    """«Докладывает такой-то» модель берёт из переклички целиком."""
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    assert (
        roster.canonical("Докладывает Огородникова Наталья")
        == "Огородникова Наталья Юрьевна"
    )


def test_a_name_written_exactly_as_in_the_registry_survives(tmp_path):
    """Проверять надо по описи, а не по тому, изменилась ли строка.

    Названный правильнее всех не меняется при сведении — и вычёркивался бы
    первым.
    """
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    assert (
        roster.canonical("Семенов Алексей Валерьевич")
        == "Семенов Алексей Валерьевич"
    )


def test_a_single_given_name_is_not_enough(tmp_path):
    """Тёзок на заседании несколько, и по одному имени выбирать нельзя."""
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    assert roster.canonical("Андрей") == ""


def test_a_stranger_is_not_an_assignee(tmp_path):
    """Кого на заседании не было, в графу не попадает."""
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    assert roster.canonical("Участники форума") == ""
    assert roster.canonical("Прохожий Иван Иванович") == ""


def test_a_region_becomes_the_one_who_answers_for_it(tmp_path):
    """Регион в графе исполнителя — указание на ответственного.

    Как его назвать, решает справочник: там третья колонка и заведена.
    """
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    assert roster.canonical("Луговская область") == "Правительству Луговской области"


def test_a_circle_survives_the_roster(tmp_path):
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    assert roster.canonical("регионам") == "все регионы"


def test_without_a_registry_the_name_stays_as_it_was(tmp_path):
    """Реестр участников готовят не к каждому заседанию.

    Без него опись не запирает поимённых исполнителей: разбирать без
    реестра инструмент умел и прежде.
    """
    roster = roster_of(None, directory_of(tmp_path))
    assert roster.canonical("Прохожий Иван Иванович") == "Прохожий Иван Иванович"
    assert roster.canonical("Луговская область") == "Правительству Луговской области"


def test_an_empty_roster_changes_nothing():
    roster = Roster()
    assert not roster
    assert roster.canonical("Кто угодно Иванович") == "Кто угодно Иванович"


# ------------------------------------------------------------ опись в разборе

def test_tasks_keep_their_count(tmp_path):
    """Неузнанный исполнитель стирается, но пункт остаётся пунктом."""
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    tasks = [
        Task(what="Подготовить план", who="Андрей Георгиевич"),
        Task(what="Вывезти навалы", who="Участники форума"),
        Task(what="Доложить", who=""),
    ]
    fixed = by_roster(tasks, roster)
    assert len(fixed) == 3
    assert [task.who for task in fixed] == ["Белянин Андрей Георгиевич", "", ""]


def test_options_offer_a_way_to_say_nobody(tmp_path):
    """Без «не назван» закрытый список заставляет выбрать кого-нибудь.

    Адресата называют хорошо если у трети поручений, и полная графа из
    выдуманных имён хуже честной полупустой.
    """
    options = roster_of(PEOPLE, directory_of(tmp_path)).options_for(
        "Андрей Георгиевич, по Луговской области наведите порядок."
    )
    assert options[0] == ""
    assert "Белянин Андрей Георгиевич" in options
    assert "Правительству Луговской области" in options
    # Круги стоят за любым заседанием: к ним не обращаются по фамилии.
    assert COLLECTIVE_NAMES <= set(options)


def test_the_chair_is_still_recognised_by_any_spelling():
    """Сводить написания :func:`same_person` умел и прежде — переезд её не
    трогает."""
    assert same_person("Гамзатбек Ханифович", "Хамзатбег Кариллович")
    assert not same_person("Андрей Петрович", "Антон Петрович")


# ------------------------------------------------------- опись в окне и в схеме

def test_the_window_offers_only_those_named_in_it(tmp_path):
    """Всю опись модели не показать: полторы сотни имён не влезут в окно.

    Сужать есть по чему — исполнителя называют рядом с поручением.
    """
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    options = roster.options_for(
        "Андрей Георгиевич, возьмите под контроль. По Луговской области "
        "прошу навести порядок."
    )
    assert "Белянин Андрей Георгиевич" in options
    assert "Правительству Луговской области" in options
    # Тот, о ком в окне речи не было, в список не идёт.
    assert "Семенов Алексей Валерьевич" not in options
    assert "Правительству Приморного края" not in options


def test_a_speaker_is_a_candidate_even_unnamed(tmp_path):
    """Взяться сделать самому — тоже поручение, а себя по имени не зовут."""
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    options = roster.options_for(
        "Направим письмо сегодня же.", speakers=["Семенов Алексей Валерьевич"],
    )
    assert "Семенов Алексей Валерьевич" in options


def test_a_single_given_name_does_not_pull_in_all_namesakes(tmp_path):
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    options = roster.options_for("Андрей, посмотрите.")
    assert "Белянин Андрей Георгиевич" not in options


def test_the_schema_locks_the_assignee_to_the_roster(tmp_path):
    """Свободная строка держится просьбой, ``enum`` — грамматикой."""
    from minuteforge.tasks import schema_with

    options = roster_of(PEOPLE, directory_of(tmp_path)).options_for(
        "Андрей Георгиевич, наведите порядок."
    )
    schema = schema_with(options)
    who = schema["properties"]["tasks"]["items"]["properties"]["who"]
    assert who["enum"] == options
    assert "" in who["enum"], "без «не назван» модель обязана выбрать кого-нибудь"
    # Остальные поля схемы остаются свободными: поручение и срок из описи не
    # выбирают.
    assert schema["properties"]["tasks"]["items"]["properties"]["what"] == {
        "type": "string"
    }


def test_the_window_prompt_shows_the_choices(tmp_path):
    from minuteforge.blocks import Block
    from minuteforge.chunking import Chunk
    from minuteforge.tasks import build_prompt

    chunk = Chunk([Block("Иванов", "Наведите порядок.")], index=1, total=1)
    _, user = build_prompt(chunk, assignees=["", "все регионы", "Семенов Алексей Валерьевич"])
    assert "- все регионы" in user
    assert "- Семенов Алексей Валерьевич" in user
    assert "пустым" in user
    # Пустая строка — ответ, а не вариант из списка: её не перечисляют.
    assert "- \n" not in user


def test_without_a_roster_the_prompt_is_as_it_was():
    from minuteforge.blocks import Block
    from minuteforge.chunking import Chunk
    from minuteforge.tasks import build_prompt

    chunk = Chunk([Block("Иванов", "Наведите порядок.")], index=1, total=1)
    _, user = build_prompt(chunk)
    assert user.startswith("Стенограмма:")


def test_the_roster_stage_runs_and_is_recorded(tmp_path):
    """Приведение к описи — отдельная стадия, и её видно в разборе.

    Стадия, прошедшая молча, неотличима от невыполненной: пункт пропал, а
    почему — не сказано ни в журнале, ни в логе.
    """
    from minuteforge.blocks import Block
    from minuteforge.chunking import Chunk
    from minuteforge.journal import Journal
    from minuteforge.llm import Reply
    from minuteforge.tasks import extract_tasks

    class Client:
        settings = None

        def complete(self, system, user, **kwargs):
            return Reply(text=(
                "Поручение: Подготовить отчёт по площадкам\n"
                "Кому: Андрей Георгиевич\n"
                "Срок: не указан"
            ))

    chunk = Chunk(
        [Block("Семенов Алексей Валерьевич",
               "Андрей Георгиевич, подготовьте отчёт по площадкам.")],
        index=1, total=1,
    )
    record = Journal()
    roster = roster_of(PEOPLE, directory_of(tmp_path))
    tasks = extract_tasks(
        [chunk], Client(), journal=record, roster=roster, corpus=chunk.text,
    )
    assert [task.who for task in tasks] == ["Белянин Андрей Георгиевич"]
    assert any("описи" in step.title for step in record.steps)


# --------------------------------------------------------- круги этого штаба

def test_the_circles_of_this_staff_are_recognised():
    """Волонтёрам и генподрядчику поручают вслух, и адресат у них есть."""
    assert circle("волонтёров") == "волонтёры"
    assert circle("волонтеры") == "волонтёры"
    assert circle("генподрядной организации") == "генеральный подрядчик"
    assert circle("Генеральному подрядчику") == "генеральный подрядчик"


def test_a_contractor_without_gen_is_not_a_circle():
    """«Подрядчик отстаёт от графика» звучит в каждом докладе.

    Там речь об одном конкретном подрядчике конкретного объекта, а не о
    круге исполнителей.
    """
    assert circle("подрядчик") == ""
    assert circle("подрядной организации") == ""


def test_only_a_room_wide_circle_gives_up_its_region():
    """Генподрядчик у каждого объекта свой: без региона с него не спросить.

    Направление снимают только у того поручения, что сказано всему залу, —
    иначе адресат сужается до одного субъекта.
    """
    from minuteforge.assignees import ROOM_WIDE

    assert "все регионы" in ROOM_WIDE
    assert "региональные штабы" in ROOM_WIDE
    assert "генеральный подрядчик" not in ROOM_WIDE
    assert "управляющие компании" not in ROOM_WIDE
    assert ROOM_WIDE <= COLLECTIVE_NAMES
