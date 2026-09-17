"""Сценарий заседания: кто говорит, по кому разбор и кто ведёт."""

from minuteforge.directory import Directory, Entry
from minuteforge.scenario import read_scenario


def directory() -> Directory:
    """Справочник для проверок: никакой предметной области, одни выдумки."""
    return Directory(entries=(
        Entry("Северный филиал", ("северн",)),
        Entry("Заречный филиал", ("заречн",)),
        Entry("Восточная площадка", ("восточн",)),
    ))


SCRIPT = """\
Технический сценарий
Вступительное слово Первого заместителя Директора И.И. ПЕТРОВА.
I. Первый вопрос повестки
Доклад Сидоров Павел Олегович – Начальник управления развития.
П.О. СИДОРОВ. Презентация. Северный филиал.
Комментарий И.И. ПЕТРОВ.
Слово Главный инженер Северного филиала — Кузнецов Игорь Петрович.
П.О. СИДОРОВ. Презентация. Заречный филиал.
ВИДЕО Заречный филиал
Слово Начальник цеха Заречного филиала Орлова Мария Сергеевна.
Комментарий И.И. ПЕТРОВ.
II. Второй вопрос повестки
Доклад Волков Никита Андреевич – Директор по развитию.
Завершающее слово И.И. ПЕТРОВА.
"""


def scenario(tmp_path, text: str = SCRIPT):
    path = tmp_path / "сценарий.txt"
    path.write_text(text, encoding="utf-8")
    return read_scenario(path, directory())


def test_the_round_is_read_in_the_order_it_is_planned(tmp_path):
    """Сценарий — это справочник, собранный под одно заседание: направления
    в нём стоят в том порядке, в каком их будут слушать."""
    assert scenario(tmp_path).units == ["Северный филиал", "Заречный филиал"]


def test_a_speaker_belongs_to_the_unit_he_speaks_after(tmp_path):
    """Привязка берётся из порядка: за презентацией по филиалу слово даётся
    его человеку."""
    people = {p.name: p for p in scenario(tmp_path).people}
    assert people["Кузнецов Игорь Петрович"].org == "Северный филиал"
    assert people["Кузнецов Игорь Петрович"].position == "Главный инженер Северного филиала"
    assert people["Орлова Мария Сергеевна"].org == "Заречный филиал"


def test_a_position_without_a_dash_is_still_read(tmp_path):
    """Тире ставят не всегда: «Слово Начальник цеха Орлова Мария Сергеевна»."""
    people = {p.name: p for p in scenario(tmp_path).people}
    assert "Начальник цеха" in people["Орлова Мария Сергеевна"].position


def test_a_new_topic_clears_the_unit(tmp_path):
    """Разбор начинается заново, и федеральный докладчик, объявленный сразу
    за заголовком, не должен достаться последнему направлению прошлого
    вопроса."""
    people = {p.name: p for p in scenario(tmp_path).people}
    assert people["Волков Никита Андреевич"].org == ""
    assert people["Сидоров Павел Олегович"].org == ""


def test_the_one_who_comments_on_everything_is_the_chair(tmp_path):
    """В сценарии комментарий ведущего стоит после каждого доклада, а
    докладчик назван один раз."""
    assert scenario(tmp_path).chair == "И.И. Петров"


def test_a_missing_scenario_does_not_stop_the_work(tmp_path):
    """Всё, что он даёт, задаётся и руками."""
    assert not read_scenario(tmp_path / "нет.docx", directory())
    assert not read_scenario(None, directory())


def test_without_a_directory_no_units_are_invented(tmp_path):
    """Сценарий пишет названия как придётся, и брать их как есть значит
    завести направление из опечатки."""
    assert read_scenario_units(tmp_path) == []


def read_scenario_units(tmp_path):
    path = tmp_path / "сценарий.txt"
    path.write_text(SCRIPT, encoding="utf-8")
    return read_scenario(path).units
