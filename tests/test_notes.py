"""Отметили: о чём доложили по каждому направлению."""

import json

from minuteforge.blocks import Block
from minuteforge.directory import Directory, Entry
from minuteforge.llm import Reply
from minuteforge.notes import (
    Note,
    Report,
    _read,
    _why_dropped,
    _without_repeats,
    split_into_reports,
    take_notes,
)
from minuteforge.protocol import build_protocol


def directory() -> Directory:
    """Справочник для проверок: никакой предметной области, одни выдумки."""
    return Directory(entries=(
        Entry("Северный филиал", ("северн",)),
        Entry("Заречный филиал", ("заречн",)),
        Entry("Восточная площадка", ("восточн",)),
    ))


def meeting() -> list[Block]:
    """Совещание по кругу: повестка, два доклада и один окрик мимоходом."""
    return [
        Block("ведущий", "Коллеги, начинаем. На связи все.", 0, 30),
        Block("ведущий", "Переходим к первому вопросу. Северный филиал.", 30, 60),
        Block("докладчик", "Готовность цеха 93,5 процента. Ввод в декабре.", 60, 300),
        Block("ведущий", "Принято. Следующий филиал Заречный.", 300, 330),
        Block("другой", "Закуплено двадцать единиц техники. Простоев нет.", 330, 600),
        Block("ведущий", "Спасибо. Переходим к третьему вопросу повестки.", 600, 630),
        Block("ведущий", "Восточная площадка упоминалась в докладе.", 630, 660),
    ]


def test_reports_are_cut_by_the_round_and_kept_in_order():
    """Пункт — доклад целиком, и порядок по ходу совещания, а не по алфавиту."""
    reports, outside = split_into_reports(meeting(), directory(), floor=60)
    assert [r.unit for r in reports] == ["Северный филиал", "Заречный филиал"]
    # Доклад начинается там, где передали слово, а не с начала реплики, в
    # которой это сделано: «Переходим к первому вопросу» относится к общей
    # части, «Северный филиал» — уже к докладу.
    assert 30 < reports[0].start < 60
    assert reports[0].blocks[0].text == "Северный филиал."
    # Повестка до первого перехода и общая часть после него — вне разбора.
    assert [b.text for b in outside.blocks][0] == "Коллеги, начинаем. На связи все."


def test_a_passing_mention_is_not_a_report():
    """«Восточная площадка упоминалась в докладе» — не доклад по ней.

    Порог длины и нужен затем, чтобы упоминание не стало пунктом протокола.
    """
    reports, outside = split_into_reports(meeting(), directory(), floor=60)
    assert "Восточная площадка" not in [r.unit for r in reports]


def test_without_a_directory_nothing_is_laid_out():
    """Нет справочника — раскладывать доклады не на что, и выдумывать
    направления инструмент не берётся."""
    reports, outside = split_into_reports(meeting(), Directory())
    assert reports == []
    assert len(outside.blocks) == len(meeting())


class FakeClient:
    """Отвечает заготовленным на каждый запрос."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(self, system, user, **kwargs):
        self.prompts.append(user)
        answer = self.answers.pop(0) if self.answers else '{"theses": []}'
        return Reply(text=answer)


def said(*theses) -> str:
    return json.dumps({"theses": list(theses)}, ensure_ascii=False)


def test_notes_are_taken_for_every_report():
    """По каждому докладу — пункт с тезисами и временем."""
    reports, _ = split_into_reports(meeting(), directory(), floor=60)
    client = FakeClient(
        said("Готовность цеха 93,5 процента."),
        said("Закуплено двадцать единиц техники."),
    )
    notes = take_notes(reports, client)
    assert [n.unit for n in notes] == ["Северный филиал", "Заречный филиал"]
    assert notes[0].theses == ["Готовность цеха 93,5 процента."]
    assert notes[0].at == reports[0].start


def test_a_report_without_theses_does_not_become_an_item():
    """Пустой пункт хуже отсутствующего: он выглядит потерянным текстом."""
    reports, _ = split_into_reports(meeting(), directory(), floor=60)
    client = FakeClient(said(), said("Закуплено двадцать единиц техники."))
    assert [n.unit for n in take_notes(reports, client)] == ["Заречный филиал"]


def test_an_order_is_not_a_note():
    """Требование — это «Решили», а не «Отметили». Пункт, попавший не в свой
    раздел, хуже пропущенного: его исполнят дважды или не исполнят вовсе."""
    said_here = "Подготовьте план работ и представьте его до пятницы."
    assert "Решили" in _why_dropped(said_here, f"докладчик: {said_here}")


def test_a_demand_named_by_a_noun_is_still_a_demand():
    """«Необходимость провести мониторинг» — тот же приказ, только
    существительным, и правило про повелительное наклонение его не берёт."""
    said_here = "Необходимость провести мониторинг курортного региона."
    assert "Решили" in _why_dropped(said_here, f"докладчик: {said_here}")


def test_a_heading_from_a_slide_is_not_a_thesis():
    """Докладывают по слайду, и модель пересказывает заголовки граф."""
    source = "докладчик: Даты выполнения. Показатели мощности. Проценты."
    assert _why_dropped("Даты выполнения", source)
    assert _why_dropped("Показатели мощности", source)


def test_a_short_thesis_is_still_a_thesis():
    """Отсекается заведомый обрывок, а не всё короткое: «Фотоотчёт отправлен
    сегодня» короче иного заголовка графы."""
    source = "докладчик: Фотоотчет отправлен сегодня, как договаривались."
    assert not _why_dropped("Фотоотчет отправлен сегодня.", source)


def test_several_sentences_become_several_theses():
    """Модели сказано «одно предложение», а она складывает в один тезис три.
    Порознь их и читать легче, и проверять."""
    assert _read(said(
        "Готовность цеха 93,5 процента. Ввод в декабре."
    )) == ["Готовность цеха 93,5 процента.", "Ввод в декабре."]


def test_the_reason_for_dropping_is_written_down():
    """Отсев здесь идёт правилами, и по документу его не видно: направления
    просто нет. Причина уходит в записку, и по ней настраивают пороги."""
    source = "докладчик: Готовность цеха 93,5 процента."
    why = _why_dropped("Отставание от карты составляет полтора года.", source)
    assert "в куске найдено" in why


def test_an_invented_thesis_is_dropped():
    """Модель дописывает подробности, которых в куске не было."""
    source = "докладчик: Готовность цеха 93,5 процента. Ввод в декабре."
    assert not _why_dropped("Готовность цеха 93,5 процента.", source)
    assert _why_dropped("Отставание от дорожной карты составляет полтора года.", source)


def test_an_invented_number_is_dropped_however_it_is_worded():
    """Доля общих слов выдумку ловит плохо: пересказ законно меняет слова, а
    подставленное число слов почти не меняет. А выдуманное число — худшее,
    что может случиться с протоколом: словами спорят, цифру переносят в
    отчёт как есть."""
    source = "докладчик: Модернизация завершена, ввод до 31.12.2026 года."
    assert _why_dropped("Готовность составляет 93,5 %.", source).startswith(
        "в куске не звучало"
    )
    assert not _why_dropped("Модернизация завершена, ввод до 31.12.2026.", source)


def test_a_number_said_in_words_still_counts_as_heard():
    """В речи числа называют словами чаще, чем цифрами, а модель послушно
    переводит их в цифры. Без этого верный пересказ выглядел бы выдумкой."""
    source = (
        "докладчик: задержка реализации дорожных карт на полтора года, "
        "в городе полторы тысячи контейнерных площадок"
    )
    assert not _why_dropped(
        "Задержка реализации дорожных карт составляет 1,5 года.", source
    )
    assert not _why_dropped(
        "В городе полторы тысячи контейнерных площадок, то есть 1500.", source
    )
    assert _why_dropped(
        "Задержка реализации дорожных карт составляет 93,5 года.", source
    ).startswith("в куске не звучало")


def test_a_chunk_answered_in_english_is_asked_again():
    """`mistral` срывается на английский, и кусок теряется целиком: на записи
    03.09 так пропали Якутия и Калмыкия — по два десятка верных тезисов.
    Дешевле переспросить, чем потерять доклад."""
    reports, _ = split_into_reports(meeting(), directory(), floor=60)
    client = FakeClient(
        said("Workshop readiness is 93,5 percent."),
        said("Готовность цеха 93,5 процента."),
        said("Закуплено двадцать единиц техники."),
    )
    notes = take_notes(reports[:1], client)
    assert notes[0].theses == ["Готовность цеха 93,5 процента."]
    assert len(client.prompts) == 2


def test_an_answer_in_another_language_is_dropped():
    """Мелкая модель срывается на английский, и это видно только проверкой."""
    source = "докладчик: Готовность цеха 93,5 процента."
    assert _why_dropped("The workshop is 93,5 percent ready.", source) == "не по-русски"


def test_the_speaker_label_is_not_part_of_the_thesis():
    """Модель переписывает строку куска целиком, вместе с подписью. Чей
    доклад — стоит в заголовке пункта, в тезисе это лишнее."""
    assert _read(said("SPEAKER_12: В городе 400 площадок.")) == [
        "В городе 400 площадок."
    ]
    assert _read(said("Иванов И.И.: готовность 93 процента.")) == [
        "готовность 93 процента."
    ]


def test_a_colon_inside_a_thesis_is_not_a_label():
    """«Готовность: 93 %» — тезис, а не подпись говорящего."""
    assert _read(said("Готовность: 93 процента.")) == ["Готовность: 93 процента."]


def test_an_answer_wrapped_in_a_fence_is_still_read():
    """Модель оборачивает ответ в ```json — с этим ничего не поделать."""
    assert _read('```json\n{"theses": ["Ввод в декабре."]}\n```') == ["Ввод в декабре."]


def test_repeats_between_parts_are_merged():
    """Соседние части доклада пересказывают одно и то же."""
    assert _without_repeats([
        "Готовность цеха 93,5 процента.",
        "Готовность цеха составляет 93,5 процента.",
        "Ввод в декабре.",
    ]) == ["Готовность цеха 93,5 процента.", "Ввод в декабре."]


def test_the_section_is_rendered_before_the_orders():
    """В документе «Отметили» стоит раньше «Поручений» — так и в подписанном."""
    protocol = build_protocol(
        [],
        notes=[Note(unit="Северный филиал", theses=["Ввод в декабре."], at=30)],
    )
    text = protocol.as_markdown()
    assert "## Отметили" in text
    assert text.index("## Отметили") < text.index("## Решили")
    assert "Северный филиал" in text
    assert "- Ввод в декабре." in text


def test_without_notes_there_is_no_empty_section():
    """Пустой заголовок выглядит потерянным содержимым."""
    assert "## Отметили" not in build_protocol([]).as_markdown()
