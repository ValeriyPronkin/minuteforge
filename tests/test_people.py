import io

from minuteforge.blocks import Block
from minuteforge.people import (
    Person,
    find,
    merge_suggestions,
    mentioned_people,
    names,
    read_people,
    suggest_speakers,
)


def uploaded(text: str, name: str = "people.csv"):
    stream = io.BytesIO(text.encode("utf-8"))
    stream.name = name
    return stream


# ------------------------------------------------------------ чтение списка

def test_table_with_positions_is_read():
    people = read_people(uploaded(
        "ФИО;Должность;Организация\n"
        "Семенов А.В.;первый замминистр;Минчистоты\n"
        "Белянина О.Е.;министр экологии;Минэкологии\n"
    ))
    assert names(people) == ["Семенов А.В.", "Белянина О.Е."]
    assert people[0].position == "первый замминистр"
    assert people[0].full == "Семенов А.В., первый замминистр, Минчистоты"


def test_plain_list_one_per_line():
    people = read_people(uploaded(
        "Семенов Алексей Валерьевич — первый замминистр\n"
        "Белянина Ольга Евгеньевна, министр экологии\n"
        "Иванов Иван Иванович\n",
        "участники.txt",
    ))
    assert len(people) == 3
    assert people[0].position == "первый замминистр"
    assert people[1].position == "министр экологии"
    assert people[2].position == ""


def test_office_noise_is_skipped():
    """В реестрах из делопроизводства хватает разделителей и номеров страниц."""
    people = read_people(uploaded("Иванов И.И.\n-------\n12\n|  |\nПетров П.П.\n", "с.txt"))
    assert names(people) == ["Иванов И.И.", "Петров П.П."]


def test_empty_list_is_not_an_error():
    assert read_people(uploaded("", "пусто.txt")) == []


def test_person_is_found_by_name_ignoring_case():
    people = [Person("Семенов А.В.", "замминистр")]
    assert find(people, " семенов а.в. ").position == "замминистр"
    assert find(people, "Кто-то") is None


# --------------------------------------------------- имена из самой записи

ANNOUNCEMENT = (
    "У нас на связи сегодня от Московской области Семенов Алексей Валерьевич, "
    "первый замминистр чистоты Московской области. Алексей Валерьевич, "
    "прокомментируйте, пожалуйста."
)


def test_name_and_position_are_taken_from_the_speech():
    """Список участников на таких совещаниях звучит вслух — это единственный
    источник имён, когда реестра нет."""
    people = mentioned_people(ANNOUNCEMENT)
    assert len(people) == 1
    assert people[0].name == "Семенов Алексей Валерьевич"
    assert "замминистр" in people[0].position


def test_mangled_patronymic_is_still_recognised():
    """Whisper коверкает отчества: «Евгеньевна» приезжает как «Евгенина».
    Правило про три слова с большой буквы подряд ловит и такое."""
    people = mentioned_people("Ольга Евгенина Белянина, министр экологии области.")
    assert people and people[0].name == "Ольга Евгенина Белянина"


def test_ordinary_phrases_are_not_mistaken_for_names():
    """«Московской области» с большой буквы — не человек: обычные
    словосочетания прерываются строчной буквой."""
    assert mentioned_people("Ставка Банка России снижается в Московской области.") == []


def test_one_person_named_twice_appears_once():
    text = ANNOUNCEMENT + " Ещё раз: Семенов Алексей Валерьевич доложит позже."
    assert len(mentioned_people(text)) == 1


# ------------------------------------------------------- догадка по порядку

def test_the_one_introduced_is_the_one_who_speaks_next():
    """Ведущий называет человека и передаёт слово — значит имя из хвоста
    чужой реплики принадлежит тому, кто заговорил следующим."""
    blocks = [
        Block("SPEAKER_02", ANNOUNCEMENT),
        Block("SPEAKER_01", "Уважаемые коллеги, добрый день. Доложу по технике."),
    ]
    guesses = suggest_speakers(blocks)
    assert guesses["SPEAKER_01"].name == "Семенов Алексей Валерьевич"
    assert "SPEAKER_02" not in guesses, "о себе в третьем лице не говорят"


def test_no_guess_when_two_people_are_named_at_once():
    """Названы двое — кто из них заговорил, неизвестно. Молчать честнее,
    чем подписать наугад."""
    blocks = [
        Block("SPEAKER_00", "Слово Иванову Ивану Ивановичу и Петрову Петру Петровичу."),
        Block("SPEAKER_01", "Добрый день."),
    ]
    assert suggest_speakers(blocks) == {}


# ----------------------------------------------------------- слияние двух

def test_roster_spelling_wins_over_the_recording():
    """В записи «Евгенина» вместо «Евгеньевна» — в списке написано верно."""
    roster = [Person("Белянина Ольга Евгеньевна", "министр экологии")]
    heard = [Person("Ольга Евгенина Белянина", "министра экологии и ресурсов")]

    merged = merge_suggestions(roster, heard)
    assert names(merged) == ["Белянина Ольга Евгеньевна"]
    assert merged[0].position == "министр экологии"


def test_person_absent_from_the_roster_is_added():
    """Список бывает неполным: кто-то подключился, кого-то забыли внести."""
    roster = [Person("Иванов Иван Иванович")]
    heard = [Person("Семенов Алексей Валерьевич", "замминистр")]
    assert len(merge_suggestions(roster, heard)) == 2


def test_merging_without_a_roster_keeps_what_was_heard():
    heard = [Person("Семенов Алексей Валерьевич")]
    assert merge_suggestions([], heard) == heard
