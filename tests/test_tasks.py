import pytest

from minuteforge.blocks import Block
from minuteforge.chunking import Chunk
from minuteforge.llm import LLMError, Reply
from minuteforge.tasks import (
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
    # В тексте фрагментов есть слова поручений: выписанное из воздуха
    # отсеивается проверкой на выдумку, и это правильно.
    chunks = [
        Chunk([Block("Иванов", "Подготовьте отчётность по первому вопросу.")], index=1, total=3),
        Chunk([Block("Иванов", "Обсуждение без поручений.")], index=2, total=3),
        Chunk([Block("Петров", "Направьте документы по второму вопросу.")], index=3, total=3),
    ]
    client = FakeClient(
        "Поручение: Подготовить отчётность\nКому: Иванов",
        NOTHING_FOUND,
        "Поручение: Направить документы\nКому: Петров",
    )
    tasks = extract_tasks(chunks, client)
    assert len(client.prompts) == 3
    assert [t.what for t in tasks] == ["Подготовить отчётность", "Направить документы"]


def test_failed_chunk_does_not_lose_the_whole_run():
    """Терять час работы из-за одного сбоя хуже, чем недосчитаться поручений
    из одного фрагмента."""
    chunks = [
        Chunk([Block("И", "Первый вопрос повестки, обсуждение.")], index=1, total=2),
        Chunk([Block("И", "Подготовьте справку по инцидентам за квартал.")], index=2, total=2),
    ]
    client = FakeClient(LLMError("сервер устал"), "Поручение: Подготовить справку по инцидентам")
    tasks = extract_tasks(chunks, client)
    assert [t.what for t in tasks] == ["Подготовить справку по инцидентам"]


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


def test_prompt_shows_the_model_an_example():
    """Семимиллиардная модель держит формат заметно устойчивее, когда видит
    образец. Из дешёвых способов вытянуть малую модель это самый заметный."""
    system, _ = build_prompt(Chunk([Block("Иванов", "Раз.")], index=1, total=1))
    assert "Пример." in system
    assert "Поручение: Подготовить план работ" in system
    # В примере нарочно есть поручение без исполнителя: модель должна видеть,
    # что «не указано» — допустимый ответ, а не повод выдумать фамилию.
    assert "Кому: не указано" in system


def test_the_example_itself_parses_by_our_own_rules():
    """Образец в промпте должен разбираться тем же кодом, что и ответ модели.
    Иначе мы просим формат, который сами не понимаем."""
    system, _ = build_prompt(Chunk([Block("Иванов", "Раз.")], index=1, total=1))
    answer = system.split("Ответ:", 1)[1]
    tasks = parse_tasks(answer)
    assert [t.what for t in tasks] == [
        "Подготовить план работ по выгрузке и согласовать",
        "Собрать справку по инцидентам",
    ]
    assert tasks[0].who == "Сергей Ким" and tasks[1].who == ""


def test_instruction_echoed_by_the_model_does_not_erase_the_answer():
    """Мелкая модель охотно пересказывает задание перед ответом, и
    фраза-маркер попадает в текст вместе с настоящими поручениями. Если
    проверять маркер первым, найденное выбрасывается и человек видит ноль.
    """
    answer = (
        "Хорошо! Я секретарь совещания. Если поручений нет, я отвечу "
        f"{NOTHING_FOUND}. Вот что нашёл:\n\n"
        "Поручение: Развернуть стенд\nКому: Тимур Асанов\nСрок: к среде"
    )
    tasks = parse_tasks(answer)
    assert len(tasks) == 1
    assert tasks[0].who == "Тимур Асанов"


def test_honest_nothing_found_still_means_nothing():
    assert parse_tasks(f"  {NOTHING_FOUND}  ") == []


def test_unparsed_answer_is_kept_for_diagnosis():
    """Ответ прозой, по-английски или не по формату выглядит так же, как
    пустой результат. Отличить одно от другого можно только по самому
    ответу."""
    chunks = [Chunk([Block("И", "раз")], index=1, total=1)]
    client = FakeClient("Модель порассуждала о погоде и ничего не выписала.")
    answers: list[str] = []

    tasks = extract_tasks(chunks, client, answers=answers)
    assert tasks == []
    assert answers == ["Модель порассуждала о погоде и ничего не выписала."]


# --------------------------------------------------------- строгий JSON

def test_json_answer_is_understood():
    """Мелкую модель уговорами формату не научить — её принуждает сервер."""
    answer = (
        '{"tasks": [{"what": "Подготовить план", "who": "Сергей Ким", "due": "до пятницы"},'
        ' {"what": "Собрать справку", "who": "", "due": ""}]}'
    )
    tasks = parse_tasks(answer)
    assert [t.what for t in tasks] == ["Подготовить план", "Собрать справку"]
    assert tasks[0].due == "до пятницы"
    assert tasks[1].who == ""


def test_empty_json_list_means_nothing_found():
    assert parse_tasks('{"tasks": []}') == []


def test_json_wrapped_in_chatter_is_still_read():
    """Даже под принуждением модель иногда добавляет слово от себя."""
    tasks = parse_tasks('Вот результат: {"tasks": [{"what": "Направить письмо"}]} Готово!')
    assert [t.what for t in tasks] == ["Направить письмо"]


def test_json_prompt_forbids_retelling_the_transcript():
    """Ровно то, чем мелкая модель и грешит: вместо ответа переписывает
    стенограмму целиком."""
    system, _ = build_prompt(Chunk([Block("И", "раз")], index=1, total=1), json_mode=True)
    assert "не переписывай" in system.lower()
    assert '{"tasks": []}' in system


def test_json_example_in_the_prompt_parses_by_our_own_rules():
    system, _ = build_prompt(Chunk([Block("И", "раз")], index=1, total=1), json_mode=True)
    example = system.split("Ответ:", 1)[1]
    tasks = parse_tasks(example)

    assert len(tasks) == 5, "в примере показаны все ходовые формы поручения"
    assert any(t.who == "" for t in tasks), "показано поручение без исполнителя"
    assert any("системно" in t.what for t in tasks), "показана форма «просьба»"
    assert any("выписки" in t.what for t in tasks), "показана форма «в протокол»"


def test_line_format_still_works_when_json_is_not_used():
    """Сервер может не уметь строгий JSON — тогда остаётся разбор по строкам."""
    assert parse_tasks("Поручение: Сделать\nКому: Иванов")[0].who == "Иванов"


# ------------------------------------------------ проверка на выдумку

FRAGMENT = (
    "Орлов В.П.: Просьба проанализировать невыбранные лимиты и доложить на "
    "следующем совещании. Ким С.А.: Направим письмо в министерство сегодня."
)


def test_invented_task_is_dropped():
    """Модель, потерявшая нить, сочиняет правдоподобные поручения из ничего.
    Отличить их можно ровно одним способом: посмотреть, есть ли эти слова в
    стенограмме.
    """
    from minuteforge.tasks import grounded

    real = Task("Проанализировать невыбранные лимиты", "Орлов В.П.", "")
    invented = Task("Подготовить презентацию для инвестиционного комитета", "Assistant", "")

    assert grounded(real, FRAGMENT)
    assert not grounded(invented, FRAGMENT)


def test_paraphrase_survives():
    """Модель имеет право переформулировать: требовать дословности значило
    бы выбрасывать хорошие поручения."""
    from minuteforge.tasks import grounded

    task = Task("Направить письмо в министерство", "", "сегодня")
    assert grounded(task, FRAGMENT)


def test_english_answer_is_dropped():
    """На русской стенограмме мелкая модель то и дело сваливается в
    английский, и просьбы в промпте этого не отменяют."""
    from minuteforge.tasks import is_russian

    assert not is_russian("Analyze the presentation given by the speaker")
    assert is_russian("Проанализировать невыбранные лимиты")


def test_sentence_in_the_assignee_field_is_cleared():
    """Модель кладёт туда целые предложения: «The team should analyze the
    current status…». В графе «Исполнитель» такому не место."""
    from minuteforge.tasks import clean_assignee

    long_one = "The team should analyze the current status of the projects and report"
    assert clean_assignee(long_one, FRAGMENT) == ""
    assert clean_assignee("Ким С.А.", FRAGMENT) == "Ким С.А."


def test_assignee_who_was_never_mentioned_is_cleared():
    """Выдуманный адресат хуже пустого поля: поручение уйдёт не тому."""
    from minuteforge.tasks import clean_assignee

    assert clean_assignee("Ассистент", FRAGMENT) == ""


def test_assignee_named_elsewhere_in_the_meeting_survives():
    """Человека представляют один раз, в начале, а поручение он получает
    часом позже — сверяться нужно со всей стенограммой, а не с фрагментом."""
    from minuteforge.tasks import clean_assignee

    corpus = "У нас на связи Белянина Ольга Евгеньевна, министр экологии. " + FRAGMENT
    assert clean_assignee("Белянина", corpus) == "Белянина"


def test_english_answer_triggers_one_retry():
    """Просьба в промпте модель не удерживает, но требование в самом конце
    запроса — обычно да: последнее указание весит больше всего."""
    chunk = Chunk(
        [Block("Орлов В.П.", "Просьба проанализировать невыбранные лимиты.")],
        index=1, total=1,
    )
    client = FakeClient(
        '{"tasks": [{"what": "Analyze the unused limits", "who": "", "due": ""}]}',
        '{"tasks": [{"what": "Проанализировать невыбранные лимиты", "who": "", "due": ""}]}',
    )
    tasks = extract_tasks([chunk], client)

    assert len(client.prompts) == 2, "должен быть ровно один повтор"
    assert "русском языке" in client.prompts[1][1]
    assert [t.what for t in tasks] == ["Проанализировать невыбранные лимиты"]


def test_russian_answer_is_not_retried():
    chunk = Chunk([Block("О", "Просьба проанализировать лимиты.")], index=1, total=1)
    client = FakeClient('{"tasks": [{"what": "Проанализировать лимиты", "who": "", "due": ""}]}')
    extract_tasks([chunk], client)
    assert len(client.prompts) == 1


def test_state_in_the_due_field_is_cleared():
    """«Завершена» и «Добрый день» в графе срока хуже пустого места: строка
    выглядит заполненной, а спросить по ней нечего."""
    from minuteforge.tasks import clean_due

    assert clean_due("Завершена") == ""
    assert clean_due("Добрый день") == ""
    assert clean_due("15 сентября") == "15 сентября"
    assert clean_due("до конца следующей недели") == "до конца следующей недели"


def test_greeting_is_not_an_assignee():
    """Поручение «уважаемым коллегам» разослать некому, а в графе контроля
    оно выглядит назначенным."""
    from minuteforge.tasks import clean_assignee

    corpus = "Уважаемые коллеги, мы начинаем. Ким С.А., подготовьте справку."
    assert clean_assignee("Уважаемые коллеги", corpus) == ""
    assert clean_assignee("Мы", corpus) == ""
    assert clean_assignee("Ким С.А.", corpus) == "Ким С.А."


# ------------------------------------------- свои указания модели

def test_extra_instructions_are_appended_not_replacing():
    """Правила формата менять нельзя: сломается разбор, а человек не поймёт
    почему. Своё дописывается в конец, отдельным разделом."""
    system, _ = build_prompt(
        Chunk([Block("И", "раз")], index=1, total=1),
        json_mode=True,
        extra="Поручения у нас называются заданиями.",
    )
    assert "Дополнительно:\nПоручения у нас называются заданиями." in system
    assert '{"tasks": []}' in system, "правила формата остались на месте"
    assert system.index("Дополнительно:") > system.index("Ответь строго")


def test_empty_extra_changes_nothing():
    plain, _ = build_prompt(Chunk([Block("И", "раз")], index=1, total=1), json_mode=True)
    spaced, _ = build_prompt(
        Chunk([Block("И", "раз")], index=1, total=1), json_mode=True, extra="   "
    )
    assert plain == spaced


def test_own_prompt_replaces_the_built_in_one():
    """Для тех, кто понимает, что делает: правила формата придётся написать
    самому."""
    system, _ = build_prompt(
        Chunk([Block("И", "раз")], index=1, total=1), base="Своё указание целиком."
    )
    assert system == "Своё указание целиком."


def test_unreadable_prompt_file_falls_back_to_the_built_in():
    """Опечатка в пути не должна ронять расчёт — берём встроенное."""
    from minuteforge.tasks import _own_prompt

    class Settings:
        prompt_file = "/такого/файла/нет.txt"

    assert _own_prompt(Settings()) == ""


def test_truncated_answer_keeps_what_fitted():
    """Модель упирается в предел длины и обрывается на полуслове. Поручения,
    успевшие уместиться, целы — терять их из-за одной незакрытой скобки
    значит выбрасывать минуту работы модели.
    """
    cut = (
        '{"tasks": [ {"what": "Проанализировать невыбранные лимиты", "who": "", "due": ""}, '
        '{"what": "Направить письмо в министерство", "who": "Ким С.А.", "due": "сегодня"}, '
        '{"what": "Подготовить справку по инци'
    )
    tasks = parse_tasks(cut)
    assert [t.what for t in tasks] == [
        "Проанализировать невыбранные лимиты",
        "Направить письмо в министерство",
    ]


def test_hopelessly_broken_answer_is_not_guessed():
    assert parse_tasks('{"tasks": [ {"wha') == []


# ------------------------------------------- сведение повторов моделью

def merged_client(answer: str):
    class Client:
        settings = None

        def __init__(self):
            self.prompts = []

        def complete(self, system, user, **kwargs):
            self.prompts.append((system, user))
            return Reply(text=answer)

    return Client()


def test_paraphrases_of_one_task_are_merged():
    """Фрагменты пересекаются, и одно поручение приезжает дважды разными
    словами. Понять, что это одно и то же, может только модель."""
    from minuteforge.tasks import merge_similar

    tasks = [
        Task("Подтвердить статус дорожной карты", "", "", chunk=2),
        Task("Проинформировать о текущем статусе реализации дорожной карты", "Голованова", "", chunk=1),
        Task("Направить письмо в министерство", "Ким С.А.", "сегодня", chunk=3),
    ]
    merged = merge_similar(tasks, merged_client('{"groups": [[1, 2], [3]]}'))

    assert len(merged) == 2
    assert merged[0].what == "Проинформировать о текущем статусе реализации дорожной карты"
    assert merged[0].who == "Голованова", "исполнитель берётся оттуда, где он есть"


def test_forgotten_task_survives():
    """Молчание модели не должно стоить поручения: номер, не попавший ни в
    одну группу, остаётся сам по себе."""
    from minuteforge.tasks import merge_similar

    tasks = [Task("Первое"), Task("Второе"), Task("Третье")]
    merged = merge_similar(tasks, merged_client('{"groups": [[1]]}'))
    assert len(merged) == 3


def test_model_cannot_rewrite_or_invent_tasks():
    """Модель возвращает номера, а сливает записи код — придумать своё или
    переписать чужое она не может."""
    from minuteforge.tasks import merge_similar

    tasks = [
        Task("Направить письмо"),
        Task("Подготовить справку"),
        Task("Проконтролировать сроки"),
        Task("Доложить на следующем совещании"),
    ]
    merged = merge_similar(
        tasks,
        merged_client(
            '{"groups": [[1, 2], [3], [4]], '
            '"tasks": [{"what": "Выдуманное поручение"}]}'
        ),
    )
    assert len(merged) == 3
    assert not any("Выдуманное" in t.what for t in merged), "текст берётся только наш"


def test_nonsense_grouping_is_ignored():
    """Номера вне списка и повторы в разных группах не должны ни удваивать
    поручения, ни терять их."""
    from minuteforge.tasks import merge_similar

    tasks = [Task("Первое"), Task("Второе")]
    merged = merge_similar(tasks, merged_client('{"groups": [[1, 99], [1, 2]]}'))
    assert len(merged) == 2


def test_failed_merge_leaves_everything_as_it_was():
    from minuteforge.tasks import merge_similar

    class Broken:
        settings = None

        def complete(self, *a, **kw):
            raise LLMError("сервер устал")

    tasks = [Task("Первое"), Task("Второе")]
    assert merge_similar(tasks, Broken()) == tasks


def test_wholesale_merging_is_rejected():
    """Мелкая модель, не поняв задачи, валит всё в две-три кучи: список из
    восемнадцати поручений превращается в пять. Это не повторы, это потеря
    тринадцати, и в готовом протоколе её не заметишь — он выглядит просто
    коротким.
    """
    from minuteforge.tasks import merge_similar

    tasks = [Task(f"Поручение {i}") for i in range(1, 19)]
    everything_in_two = '{"groups": [[1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18]]}'

    merged = merge_similar(tasks, merged_client(everything_in_two))
    assert len(merged) == 18, "сведение должно быть отвергнуто целиком"


def test_reasonable_merging_goes_through():
    from minuteforge.tasks import merge_similar

    tasks = [Task(f"Поручение {i}") for i in range(1, 11)]
    merged = merge_similar(tasks, merged_client('{"groups": [[1,2],[3],[4],[5],[6],[7],[8],[9],[10]]}'))
    assert len(merged) == 9


def test_a_speech_in_the_due_field_is_cleared():
    """Срок — это «до пятницы» или «15 сентября», а не абзац из доклада: в
    таблице контроля он занимает три строки, ничего не сообщая."""
    from minuteforge.tasks import clean_due

    speech = (
        "Сейчас, секунду. Комментариев уже неоднократно по Костромской области "
        "мы слушали, вы просили подключиться, мы подключались"
    )
    assert clean_due(speech) == ""
    assert clean_due("до конца следующей недели") == "до конца следующей недели"


def test_paraphrase_of_one_task_is_collapsed_by_rules():
    """Фрагментов много, границ много, и одно поручение приезжает в разных
    словах. Правила берут то, что видно по словам, — на большее они не
    претендуют."""
    tasks = [
        Task("Сформировать окончательную редакцию предложения по оптимизации дорожной карты"),
        Task("Сформировать итоговую редакцию предложения по оптимизации дорожной карты", "Артем"),
    ]
    kept = dedupe(tasks)
    assert len(kept) == 1
    assert kept[0].who == "Артем", "исполнитель берётся оттуда, где он есть"


def test_short_tasks_are_not_merged_on_a_chance_match():
    """«Работать с регионами» и «работать с региональными операторами»
    совпадут полностью по словам, а это разные поручения."""
    tasks = [
        Task("Плотно работать с регионами"),
        Task("Плотно работать с региональными операторами"),
    ]
    assert len(dedupe(tasks)) == 2


def test_similar_but_different_objects_stay_apart():
    tasks = [
        Task("Приступить к строительству объекта КПО Нижневартовска в этом году"),
        Task("Приступить к строительству объекта утилизации в Тюменской области"),
    ]
    assert len(dedupe(tasks)) == 2


def test_task_is_tied_to_the_reply_it_came_from():
    """По цитате видно сразу, поручение это или изложение доклада, и
    сверяться с записью приходится только в спорных случаях."""
    from minuteforge.tasks import attach_source

    chunk = Chunk([
        Block("Орлов В.П.", "Коротко о статусе программы за полтора года.", 3600, 3700),
        Block("Орлов В.П.", "Просьба проанализировать невыбранные лимиты и доложить.", 3930, 3990),
    ], index=1, total=1)

    task = attach_source([Task("Проанализировать невыбранные лимиты")], chunk)[0]

    assert task.at == 3930
    assert "невыбранные лимиты" in task.quote
    assert task.said_by == "Орлов В.П."


def test_unmatched_task_is_left_without_a_place_rather_than_a_wrong_one():
    from minuteforge.tasks import attach_source

    chunk = Chunk([Block("И", "Совсем о другом.", 10, 20)], index=1, total=1)
    task = attach_source([Task("Подготовить справку по инцидентам")], chunk)[0]
    assert task.at is None or task.quote == "Совсем о другом."
