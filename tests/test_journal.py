"""Журнал разбора: где потерялось и откуда взялось лишнее."""

import pytest
from loguru import logger

from minuteforge import journal as J
from minuteforge.chunking import Chunk
from minuteforge.blocks import Block
from minuteforge.journal import Journal, setup_file_log
from minuteforge.tasks import Task


def window(index=1, text="Иванов, подготовьте справку."):
    return Chunk(blocks=[Block("SPEAKER_02", text, 100.0, 130.0)], index=index, total=2)


def test_the_funnel_shows_where_the_list_thinned():
    """Видно было только начало и конец, и «модель не нашла» выглядело так
    же, как «нашла, а мы отсеяли»."""
    record = Journal()
    было = [Task(what="Подготовить справку"), Task(what="Подтвердить")]
    стало = [Task(what="Подготовить справку")]
    record.step("Отсев обломков", было, стало)

    text = record.as_markdown()
    assert "| Отсев обломков | 2 | 1 | 1 |" in text
    assert "Подтвердить" in text, "снятый пункт должен быть виден поимённо"


def test_what_the_merge_produced_is_shown_too():
    """Склейка не отсеивает, а заменяет несколько записей одной, и в
    воронке это выглядит как потеря."""
    record = Journal()
    record.step(
        "Сведение одной фразой",
        [Task(what="Передать разговор"), Task(what="Отправить фотографии")],
        [Task(what="Передать разговор; отправить фотографии")],
    )

    text = record.as_markdown()
    assert "**стало** — Передать разговор; отправить фотографии" in text


def test_silent_windows_are_the_place_to_look():
    """Правило показало модели фразу, в которой поручение слышно, а пункта
    нет. Прочитать окно — единственный способ понять, кто промахнулся."""
    record = Journal()
    record.window(window(1), "Поручение: Подготовить справку", 1)
    record.window(window(2, "Видим показатели мощности."), "НЕТ ПОРУЧЕНИЙ", 0)

    assert len(record.silent_windows) == 1
    text = record.as_markdown()
    assert "Вернула поручения из **1**, промолчала в **1**" in text
    assert "Видим показатели мощности." in text
    assert "НЕТ ПОРУЧЕНИЙ" in text


def test_a_window_remembers_when_it_was_said():
    """Без времени окно не найти в двухчасовой записи."""
    record = Journal()
    record.window(window(), "", 0)
    assert "00:01:40" in record.as_markdown()


def test_an_empty_journal_is_still_readable():
    assert "Разбор протокола" in Journal().as_markdown()


@pytest.fixture
def fresh_log():
    """Журнал заводится один раз на процесс, и тестам нужен чистый."""
    J._HANDLER = None
    yield
    if J._HANDLER is not None:
        logger.remove(J._HANDLER)
    J._HANDLER = None


def test_the_log_goes_to_a_file(tmp_path, fresh_log):
    """Настройка log_dir была, а записи в файл не было: loguru писал только
    в консоль, а у приложения, запущенного ярлыком, её нет."""
    setup_file_log(tmp_path / "logs")
    logger.info("проверка записи")

    written = list((tmp_path / "logs").glob("*.log"))
    assert written, "файл журнала должен появиться"
    assert "проверка записи" in written[0].read_text(encoding="utf-8")


def test_the_log_is_set_up_once(tmp_path, fresh_log):
    """Интерфейс перечитывает свой сценарий на каждое нажатие. Без защиты
    обработчики копились бы десятками, а строки задваивались."""
    setup_file_log(tmp_path / "logs")
    first = J._HANDLER
    setup_file_log(tmp_path / "logs")

    assert J._HANDLER == first


def test_an_unwritable_folder_does_not_stop_the_work(tmp_path, fresh_log, capsys):
    """Журнал полезен, но не обязателен: падать из-за него нельзя."""
    busy = tmp_path / "занято"
    busy.write_text("это файл, а не папка", encoding="utf-8")

    assert setup_file_log(busy) is None
    assert "Журнал не ведётся" in capsys.readouterr().err


def test_the_dropped_theses_are_written_down_with_the_reason():
    """Отсев в разделе «Отметили» идёт правилами, и по документу его не
    видно: направления просто нет. Записка — единственное место, где
    отличить «доклада не было» от «все тезисы сочтены выдумкой»."""
    record = Journal()
    record.thesis("Северный филиал", 30.0, "Готовность цеха 93,5 процента.")
    record.thesis("Северный филиал", 30.0, "Даты выполнения", "короче обрывка")
    record.thesis("Заречный филиал", 330.0, "Отставание полтора года.",
                  "в куске найдено: 40% слов из 5")

    assert len(record.dropped_theses) == 2
    text = record.as_markdown()
    assert "## Отмеченное" in text
    assert "Северный филиал — 1 из 2" in text
    assert "Заречный филиал — 0 из 1" in text
    assert "в куске найдено: 40% слов из 5" in text


def test_without_theses_there_is_no_section_about_them():
    """Пустой заголовок в записке — такой же шум, как в протоколе."""
    assert "## Отмеченное" not in Journal().as_markdown()


def test_the_time_of_every_stage_is_written_down():
    """«Стало дольше» иначе обсуждается на ощупь: стадий девять, и по числу
    пунктов не угадать, какая из них съела время."""
    import time

    from minuteforge.tasks import Task

    record = Journal()
    record.window(window(), '{}', 1)
    time.sleep(0.05)
    tasks = [Task(what="Подготовить справку"), Task(what="Проверить площадки")]
    record.step("Отсев не поручений", tasks, tasks[:1])

    assert record.spent["Чтение окон"] > 0
    assert record.spent["Отсев не поручений"] >= 0.05
    text = record.as_markdown()
    assert "## Время" in text
    assert "| Стадия | Было | Стало | Ушло | Время |" in text


def test_without_any_work_there_is_no_time_section():
    assert "## Время" not in Journal().as_markdown()
