import csv
from io import StringIO

from minuteforge.blocks import Block, Transcript
from minuteforge.protocol import Protocol, build_protocol
from minuteforge.tasks import Task


TASKS = [
    Task("Подготовить отчёт", "Иванов", "до среды", chunk=1),
    Task("Направить письмо", "Петров", "", chunk=2),
    Task("Собрать справку по инциденту", "", "к пятнице", chunk=2),
]


def transcript():
    return Transcript(
        [
            Block("Иванов", "Начнём.", 0, 30),
            Block("Петров", "Готов доложить.", 30, 120),
        ]
    )


def test_tasks_split_into_actionable_and_unclear():
    """Поручение без исполнителя — не брак распознавания, а частая правда
    совещаний: «надо подготовить справку», а кому — не сказали."""
    protocol = Protocol(tasks=TASKS)
    assert [t.what for t in protocol.actionable] == ["Подготовить отчёт", "Направить письмо"]
    assert [t.what for t in protocol.needs_clarification] == ["Собрать справку по инциденту"]


def test_markdown_has_a_table_and_a_clarification_section():
    text = Protocol(tasks=TASKS, date="05.06.2025").as_markdown()
    assert "**Дата:** 05.06.2025" in text
    assert "| 1 | — | Подготовить отчёт | Иванов | до среды |" in text
    assert "### Требуют уточнения" in text
    assert "- **—** Собрать справку по инциденту, срок: к пятнице" in text


def test_missing_deadline_is_a_dash_not_an_empty_cell():
    text = Protocol(tasks=[Task("Направить письмо", "Петров")]).as_markdown()
    assert "| 1 | — | Направить письмо | Петров | — |" in text


def test_empty_fields_are_skipped_not_left_blank():
    """Пустой заголовок «Место:» в документе выглядит как потерянные данные."""
    text = Protocol(tasks=TASKS).as_markdown()
    assert "**Место:**" not in text
    assert "**Дата:**" not in text


def test_meeting_without_tasks_says_so():
    text = Protocol(tasks=[]).as_markdown()
    assert "Поручений не зафиксировано." in text


def test_meeting_where_nobody_was_assigned():
    """Все поручения без исполнителя — таблица пуста, но пункты видны."""
    text = Protocol(tasks=[Task("Подумать над сроками")]).as_markdown()
    assert "Поручений с назначенным исполнителем нет." in text
    assert "- **—** Подумать над сроками" in text


def test_transcript_is_attached_only_on_request():
    protocol = Protocol(tasks=TASKS, transcript=transcript())
    assert "## Стенограмма" not in protocol.as_markdown()

    full = protocol.as_markdown(with_transcript=True)
    assert "## Стенограмма" in full
    assert "[00:00:30] Петров: Готов доложить." in full


def test_duration_comes_from_the_recording():
    text = Protocol(tasks=TASKS, transcript=transcript()).as_markdown()
    assert "**Длительность записи:** 2.0 мин" in text


def test_attendees_default_to_the_speakers_of_the_transcript():
    protocol = build_protocol(TASKS, transcript())
    assert protocol.attendees == ["Иванов", "Петров"]


def test_explicit_attendees_win():
    protocol = build_protocol(TASKS, transcript(), attendees=["Иванов", "Петров", "Сидоров"])
    assert "Сидоров" in protocol.attendees


def test_no_transcript_means_no_invented_attendees():
    """Приписать совещанию участника, которого не было, хуже, чем оставить
    строку незаполненной."""
    assert build_protocol(TASKS).attendees == []


def test_date_is_never_invented():
    """Дата обработки записи — не дата совещания."""
    assert build_protocol(TASKS, transcript()).date == ""


def test_csv_carries_every_task_including_unassigned():
    rows = list(csv.reader(StringIO(Protocol(tasks=TASKS).tasks_csv()), delimiter=";"))
    assert rows[0] == [
        "№", "Время", "Поручение", "Исполнитель", "Срок", "Кто сказал", "Цитата"
    ]
    assert len(rows) == 1 + len(TASKS)
    assert rows[3] == ["3", "—", "Собрать справку по инциденту", "", "к пятнице", "", ""]


def test_csv_uses_semicolon_for_excel():
    """Excel в русской локали читает csv через точку с запятой; с запятой
    весь протокол приезжает одной колонкой."""
    assert ";" in Protocol(tasks=TASKS).tasks_csv().splitlines()[0]


def test_unknown_is_not_listed_among_attendees():
    """Неопознанная реплика остаётся в стенограмме, но участником совещания
    человек по имени UNKNOWN не был."""
    with_unknown = Transcript(
        [Block("Иванов", "Раз.", 0, 5), Block("UNKNOWN", "Не слышно.", 5, 8)]
    )
    protocol = build_protocol(TASKS, with_unknown)
    assert protocol.attendees == ["Иванов"]
    assert "UNKNOWN" in protocol.as_markdown(with_transcript=True)


def test_line_after_the_attendee_list_is_not_glued_to_it():
    """Без пустой строки разметка приклеивает следующую строку к последнему
    участнику, и длительность записи оказывается его должностью."""
    from minuteforge.people import Person

    protocol = Protocol(
        tasks=TASKS,
        attendees=["Иванов И.И.", "SPEAKER_01"],
        people=[Person("Иванов И.И.", "начальник отдела")],
        transcript=transcript(),
    )
    lines = protocol.as_markdown().splitlines()
    last_attendee = max(i for i, line in enumerate(lines) if line.startswith("* "))
    assert lines[last_attendee + 1].strip() == "", "после списка нужна пустая строка"
    assert "Длительность записи" in "\n".join(lines[last_attendee:])


def test_unnamed_labels_are_counted_not_listed():
    """Метка SPEAKER_13 — не участник, а обозначение голоса.

    На двухчасовом совещании диаризация нарезает их десятками, и список из
    двадцати девяти безымянных голосов превращает протокол в бумагу,
    которую нельзя показать. Но и молчать о них нельзя: список неполон.
    """
    from minuteforge.people import Person

    protocol = Protocol(
        attendees=["SPEAKER_02", "Семенов А.В.", "SPEAKER_01", "UNKNOWN"],
        people=[Person("Семенов А.В.", "первый замминистр")],
    )
    text = protocol.as_markdown()

    assert "* Семенов А.В., первый замминистр" in text
    assert "SPEAKER_02" not in text
    assert "**Не опознано голосов:** 3" in text


def test_protocol_without_a_single_named_participant():
    """Никого не опознали — списка участников нет вовсе, а не пустой
    заголовок с двоеточием."""
    protocol = Protocol(attendees=["SPEAKER_00", "SPEAKER_01"])
    text = protocol.as_markdown()
    assert "**Участники:**" not in text
    assert "**Не опознано голосов:** 2" in text


def test_every_task_carries_its_place_in_the_recording():
    """Без времени человек ищет место в двухчасовой записи сам, и сверка
    восьмидесяти пунктов съедает больше, чем сэкономил разбор."""
    tasks = [
        Task("Подготовить план", "Ким С.А.", "до пятницы", chunk=2),
    ]
    tasks[0].at = 3930.0
    tasks[0].quote = "Сергей, подготовьте план работ и согласуйте до пятницы."
    tasks[0].said_by = "Орлов В.П."

    text = Protocol(tasks=tasks).as_markdown()
    assert "01:05:30" in text

    rows = list(csv.reader(StringIO(Protocol(tasks=tasks).tasks_csv()), delimiter=";"))
    assert rows[0] == ["№", "Время", "Поручение", "Исполнитель", "Срок", "Кто сказал", "Цитата"]
    assert rows[1][1] == "01:05:30"
    assert "подготовьте план" in rows[1][6]


def test_task_without_a_place_shows_a_dash_not_a_crash():
    text = Protocol(tasks=[Task("Подготовить план", "Ким С.А.")]).as_markdown()
    assert "| — |" in text
