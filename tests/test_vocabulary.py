"""Свои слова: чем поручают у вас и что поручением у вас не считается."""

import pytest

from minuteforge.tasks import forget_words, is_directive, use_words
from minuteforge.vocabulary import EMPTY, read_vocabulary


@pytest.fixture(autouse=True)
def builtin_words():
    """Словарь заводится на прогон, и проверки не должны цеплять друг друга."""
    forget_words()
    yield
    forget_words()


def test_your_own_words_of_request_are_added():
    """«Отработать», «взять на контроль» — так поручают у одних и не
    поручают у других. Угадать это снаружи нельзя."""
    assert not is_directive("Замечание отработать до конца недели.")

    use_words(orders=["отработать"])
    assert is_directive("Замечание отработать до конца недели.")


def test_your_own_exceptions_win_over_the_built_in_rule():
    """«Не поручают» сильнее: слово оттуда не сделает фразу поручением,
    даже если оно похоже на повелительное наклонение."""
    assert is_directive("Объект сдайте до конца недели.")

    use_words(not_orders=["сдайте"])
    assert not is_directive("Объект сдайте до конца недели.")


def test_an_exception_removes_a_built_in_word():
    """У связистов «подтвердите» — про звук, а не про работу."""
    assert is_directive("Подтвердите готовность объекта.")

    use_words(not_orders=["подтвердите"])
    assert not is_directive("Подтвердите готовность объекта.")


def test_the_vocabulary_is_read_from_a_file(tmp_path):
    file = tmp_path / "слова.csv"
    file.write_text(
        "Раздел;Слова\n"
        "поручают;поручается, отработать, взять на контроль\n"
        "не поручают;слышите, сдать\n",
        encoding="utf-8-sig",
    )
    own = read_vocabulary(file)

    assert "отработать" in own.orders
    assert "взять на контроль" in own.orders
    assert own.not_orders == {"слышите", "сдать"}


def test_other_section_names_are_understood(tmp_path):
    """Разделы называют по-разному, и заставлять человека угадывать слово
    из документации незачем."""
    file = tmp_path / "слова.csv"
    file.write_text("требования;отработать\nисключения;сдать\n", encoding="utf-8")
    own = read_vocabulary(file)

    assert own.orders == {"отработать"}
    assert own.not_orders == {"сдать"}


def test_without_a_file_the_built_in_list_is_used(tmp_path):
    """Файла нет — работает как прежде: встроенного списка хватает."""
    assert not read_vocabulary(None)
    assert not read_vocabulary(tmp_path / "нет-такого.csv")
    assert read_vocabulary(None) is EMPTY
