import pytest

from protocall.blocks import Block
from protocall.chunking import Chunk
from protocall.llm import LLMError, Reply
from protocall.tasks import (
    NOTHING_FOUND,
    Task,
    build_prompt,
    dedupe,
    extract_tasks,
    parse_tasks,
)


class FakeClient:
    """Модель, отвечающая заранее заготовленным текстом."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, system, user, **kwargs):
        self.prompts.append((system, user))
        item = self.answers.pop(0) if self.answers else NOTHING_FOUND
        if isinstance(item, Exception):
            raise item
        return Reply(text=item)


ANSWER = """Поручение: Подготовить отчёт по продажам
Кому: Иванов
Срок: до среды

Поручение: Проверить подписание документов
Кому: Петров
Срок: не указан"""


def test_two_blocks_become_two_tasks():
    tasks = parse_tasks(ANSWER)
    assert [t.what for t in tasks] == [
        "Подготовить отчёт по продажам",
        "Проверить подписание документов",
    ]
    assert tasks[0].who == "Иванов" and tasks[0].due == "до среды"


def test_not_specified_becomes_empty_not_a_phrase():
    """«не указан» в поле — это заполненное поле.

    Оставь его как есть, и таблицу поручений уже не отфильтровать по тем,
    где исполнителя действительно нет.
    """
    tasks = parse_tasks(ANSWER)
    assert tasks[1].due == ""


def test_nothing_found_gives_no_tasks():
    """Совещание наполовину состоит из обсуждения: пустой ответ — норма."""
    assert parse_tasks(NOTHING_FOUND) == []
    assert parse_tasks("нет поручений") == []
    assert parse_tasks("") == []


def test_model_chatter_around_the_answer_is_tolerated():
    """Формат ответа — просьба к модели, а не гарантия."""
    answer = """Конечно! Вот найденные поручения:

1. Поручение: Направить письмо в министерство
   Кому: Сидоров
   Срок: 15 июня

Надеюсь, это помогло!"""
    tasks = parse_tasks(answer)
    assert len(tasks) == 1
    assert tasks[0].what == "Направить письмо в министерство"
    assert tasks[0].who == "Сидоров"


def test_missing_blank_line_between_blocks_still_splits():
    answer = """Поручение: Первое
Кому: Иванов
Поручение: Второе
Кому: Петров"""
    tasks = parse_tasks(answer)
    assert [t.what for t in tasks] == ["Первое", "Второе"]
    assert [t.who for t in tasks] == ["Иванов", "Петров"]


def test_field_synonyms_are_understood():
    answer = "Задача: Свести реестр\nОтветственный: Иванов\nДата: пятница"
    task = parse_tasks(answer)[0]
    assert (task.what, task.who, task.due) == ("Свести реестр", "Иванов", "пятница")


def test_block_without_the_task_itself_is_dropped():
    """Исполнитель без задачи бессмыслен."""
    assert parse_tasks("Кому: Иванов\nСрок: завтра") == []


def test_task_remembers_where_it_came_from():
    """По номеру фрагмента возвращаются к месту разговора и проверяют,
    не выдумала ли модель."""
    task = parse_tasks("Поручение: Сделать", chunk=3)[0]
    assert task.chunk == 3


def test_prompt_tells_the_model_it_sees_a_fragment():
    """Иначе модель пишет итоговый протокол по обрывку и придумывает
    недостающее."""
    chunk = Chunk([Block("Иванов", "Подготовьте отчёт.")], index=2, total=5)
    system, user = build_prompt(chunk)
    assert "2 из 5" in user
    assert "Иванов: Подготовьте отчёт." in user
    assert NOTHING_FOUND in system


def test_single_chunk_prompt_has_no_fragment_header():
    system, user = build_prompt(Chunk([Block("Иванов", "Раз.")], index=1, total=1))
    assert "Фрагмент" not in user


def test_duplicates_from_overlap_are_removed():
    """Куски идут с нахлёстом, поэтому пограничная реплика попадает в модель
    дважды. Без этого шага каждое второе поручение двоилось бы."""
    tasks = [
        Task("Подготовить отчёт", "Иванов", "до среды", chunk=1),
        Task("подготовить отчёт.", "", "", chunk=2),
    ]
    assert len(dedupe(tasks)) == 1


def test_dedupe_keeps_the_more_complete_version():
    tasks = [
        Task("Подготовить отчёт", "", "", chunk=1),
        Task("Подготовить отчёт", "Иванов", "до среды", chunk=2),
    ]
    kept = dedupe(tasks)[0]
    assert (kept.who, kept.due) == ("Иванов", "до среды")
    assert kept.chunk == 1, "ссылка на первое появление"


def test_different_tasks_are_not_merged():
    tasks = [Task("Первое"), Task("Второе")]
    assert len(dedupe(tasks)) == 2


def test_extract_walks_every_chunk():
    chunks = [Chunk([Block("Иванов", f"Реплика {i}")], index=i, total=3) for i in (1, 2, 3)]
    client = FakeClient(
        "Поручение: Первое\nКому: Иванов",
        NOTHING_FOUND,
        "Поручение: Второе\nКому: Петров",
    )
    tasks = extract_tasks(chunks, client)
    assert len(client.prompts) == 3
    assert [t.what for t in tasks] == ["Первое", "Второе"]


def test_failed_chunk_does_not_lose_the_whole_run():
    """Терять час работы из-за одного сбоя хуже, чем недосчитаться поручений
    из одного фрагмента."""
    chunks = [Chunk([Block("И", "раз")], index=i, total=2) for i in (1, 2)]
    client = FakeClient(LLMError("сервер устал"), "Поручение: Уцелевшее")
    tasks = extract_tasks(chunks, client)
    assert [t.what for t in tasks] == ["Уцелевшее"]


def test_strict_mode_reraises():
    chunks = [Chunk([Block("И", "раз")], index=1, total=1)]
    with pytest.raises(LLMError):
        extract_tasks(chunks, FakeClient(LLMError("сбой")), skip_failed=False)


def test_task_renders_as_a_line():
    assert Task("Свести реестр", "Иванов", "к пятнице").as_line() == (
        "Свести реестр; исполнитель: Иванов; срок: к пятнице"
    )
    assert Task("Свести реестр").as_line() == "Свести реестр"


def test_unknown_speaker_is_not_offered_to_the_model_as_a_person():
    """UNKNOWN — признак неопознанной реплики, а не фамилия.

    В списке участников модель принимает его за человека и начинает
    приписывать ему поручения.
    """
    chunk = Chunk(
        [Block("Иванов", "Раз."), Block("UNKNOWN", "Плохо слышно.")], index=1, total=2
    )
    _, user = build_prompt(chunk)
    assert "Участники фрагмента: Иванов." in user
    assert "UNKNOWN" not in user.split("Стенограмма:")[0]
    assert "UNKNOWN: Плохо слышно." in user, "сам текст теряться не должен"
