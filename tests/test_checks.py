"""Проверка стенограммы на выдумки распознавания."""

from minuteforge.blocks import Block
from minuteforge.checks import suspicious


def test_looping_model_is_caught():
    """Зацикливание — главная примета выдумки: пять раз подряд одно и то же."""
    line = "Шамбулат Кириллович, Эктехнопарк Богочанский. " * 5
    found = suspicious([Block("SPEAKER_29", f"Добрый день. {line}", 8582.0, 8600.0)])

    assert len(found) == 1
    assert found[0].count == 5
    assert "02:23:02" in found[0].as_line()


def test_subtitle_credits_need_no_repeat():
    """«Субтитры сделал такой-то» — обрывок титров с обучающих видео.

    На совещании этого не произносил никто, поэтому хватает одного раза.
    """
    found = suspicious([Block("SPEAKER_00", "Продолжение следует... Субтитры создавал DimaTorzok", 37.0, 60.0)])

    assert [item.reason for item in found] == ["Титры вместо речи"] * 2


def test_roll_call_is_not_a_hallucination():
    """Перекличка на сорок регионов повторяет «видно, слышно» весь час.

    Это совещание, а не брак: зациклившаяся модель повторяет фразу тут же,
    подряд, а не с интервалом в десять минут.
    """
    line = "Коллеги, добрый день, как видно, слышно?"
    blocks = [Block("SPEAKER_14", line, second, second + 5) for second in (300.0, 900.0, 2000.0)]

    assert suspicious(blocks) == []


def test_short_courtesies_are_left_alone():
    """«Спасибо» звучит на совещании десятки раз и ничего не значит."""
    blocks = [Block("SPEAKER_14", "Спасибо.", second, second + 1) for second in (10.0, 20.0, 30.0, 40.0)]

    assert suspicious(blocks) == []


def test_filler_sentence_across_the_meeting_is_reported_once():
    """Большая модель вставляет одну и ту же гладкую фразу в разные доклады.

    Опаснее зацикливания: выглядит как сказанное и молча уезжает в протокол.
    """
    filler = "Вопрос в том, что все выступления сегодня говорят о мощнейшем безобразии."
    found = suspicious([
        Block("SPEAKER_11", f"Начало доклада. {filler} Середина. {filler} Конец. {filler}", 6372.0, 7000.0),
        Block("SPEAKER_08", f"{filler} Напоминаю, он из семи объектов.", 9000.0, 9100.0),
    ])

    assert len(found) == 1
    assert found[0].count == 4
    assert found[0].moments == [6372.0, 9000.0]
