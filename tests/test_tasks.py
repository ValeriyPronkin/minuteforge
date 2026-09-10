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


def test_an_address_to_the_room_names_the_executor():
    """На штабе, где собраны все субъекты, «прошу регионы обратить внимание»
    — поручение всем регионам, а не поручение в никуда: его рассылают и по
    нему спрашивают. А вот «мы» — местоимение без того, к кому относится.

    Написание приводится к одному, иначе «регионам», «регионы» и «субъектам»
    разъедутся по таблице как три разных исполнителя.
    """
    from minuteforge.tasks import clean_assignee

    corpus = "Уважаемые коллеги, мы начинаем. Ким С.А., подготовьте справку."
    assert clean_assignee("Уважаемые коллеги", corpus) == "все участники"
    assert clean_assignee("регионам", corpus) == "все регионы"
    assert clean_assignee("Субъектам", corpus) == "все регионы"
    assert clean_assignee("Мы", corpus) == ""
    assert clean_assignee("Ким С.А.", corpus) == "Ким С.А."


def test_a_collective_executor_needs_no_name_nearby():
    """Проверка «названа ли фамилия рядом» к «всем регионам» неприменима:
    их не называют, к ним обращаются."""
    from minuteforge.blocks import Block
    from minuteforge.chunking import Chunk
    from minuteforge.tasks import Task, attach_source

    blocks = [Block("SPEAKER_02", "Я также прошу регионы обратить на это особое внимание.", 0, 20)]
    chunk = Chunk(blocks, index=1, total=1)

    attached = attach_source([Task(what="Обратить особое внимание", who="все регионы")], chunk)
    assert attached[0].who == "все регионы"


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


def test_the_checker_gets_its_own_instructions():
    """Проверяющей нужны обратные указания: не что считать поручением, а что
    им не считается, хотя звучит похоже."""
    from minuteforge.llm import Reply
    from minuteforge.tasks import Task, verify

    seen = []

    class Judge:
        def complete(self, system, user, **kwargs):
            seen.append(system)
            return Reply(text='{"order": true}')

    verify(
        [Task(what="Сдать объект", quote="Объект сдать в марте.",
              context="Объект сдать в марте.")],
        Judge(),
        extra="«Объект сдать» — строка графика, а не поручение.",
    )

    assert "Дополнительно:\n«Объект сдать» — строка графика" in seen[0]
    assert '{"order": true}' in seen[0], "правила формата остались на месте"


def test_the_checker_without_instructions_is_asked_as_before():
    from minuteforge.llm import Reply
    from minuteforge.tasks import Task, verify

    seen = []

    class Judge:
        def complete(self, system, user, **kwargs):
            seen.append(system)
            return Reply(text='{"order": true}')

    task = Task(what="Подготовить справку", quote="Подготовьте.", context="Подготовьте.")
    verify([task], Judge())
    verify([task], Judge(), extra="   ")

    assert seen[0] == seen[1]


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
        Task("Проинформировать о текущем статусе реализации дорожной карты", "Ларина", "", chunk=1),
        Task("Направить письмо в министерство", "Ким С.А.", "сегодня", chunk=3),
    ]
    merged = merge_similar(tasks, merged_client('{"groups": [[1, 2], [3]]}'))

    assert len(merged) == 2
    assert merged[0].what == "Проинформировать о текущем статусе реализации дорожной карты"
    assert merged[0].who == "Ларина", "исполнитель берётся оттуда, где он есть"


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


def test_quote_is_the_sentence_not_the_start_of_the_speech():
    """После разметки по голосам реплика — это выступление на десять минут.
    Восемь поручений из одной речи получали одну отметку времени, и
    перематывать по ней было некуда."""
    from minuteforge.tasks import attach_source

    speech = (
        "Спасибо, уважаемые коллеги. "
        "Задачи перед отраслью объёмные. "
        "Мы произведём очень плотный аудит всех процессов. "
        "Также нужно плотно работать с региональными операторами."
    )
    chunk = Chunk([Block("SPEAKER_01", speech, 3600, 3900)], index=1, total=1)

    audit, regions = attach_source(
        [Task("Произвести очень плотный аудит"),
         Task("Плотно работать с региональными операторами")],
        chunk,
    )

    assert "аудит всех процессов" in audit.quote
    assert "региональными операторами" in regions.quote
    assert audit.at != regions.at
    assert audit.at > 3600


def test_long_sentence_does_not_win_by_sheer_length():
    """Совпадение считается от слов поручения: иначе выигрывает самое
    длинное предложение, в котором просто больше слов."""
    from minuteforge.tasks import attach_source

    text = (
        "Выслать протокол. "
        "Затем мы обсудим протокол, регламент, сроки, участников, площадку, "
        "повестку, решения, выслать материалы и многое другое по списку."
    )
    chunk = Chunk([Block("И", text, 0, 100)], index=1, total=1)
    task = attach_source([Task("Выслать протокол")], chunk)[0]
    assert task.quote == "Выслать протокол."


def test_assignee_named_far_away_is_not_kept():
    """Раньше фамилия принималась, если встречалась где угодно в стенограмме.
    На двухчасовом совещании чаще всех звучит имя ведущего, и он оказывался
    адресатом там, где сам же и говорил. Неверный адресат хуже пустого."""
    from minuteforge.tasks import attach_source

    chunk = Chunk([
        Block("Ларина А.Н.", "Передача возможна после согласования.", 100, 160),
    ], index=1, total=1)

    task = attach_source(
        [Task("Прокомментировать передачу", "Огородникова Наталья Юрьевна")], chunk
    )[0]
    assert task.who == ""


def test_assignee_named_in_the_next_reply_is_kept():
    """Адресата часто называют рядом: «Сергей Анатольевич, подготовьте» — и
    следом он сам отвечает, а поручение цепляется к его словам."""
    from minuteforge.tasks import attach_source

    chunk = Chunk([
        Block("Орлов В.П.", "Сергей Анатольевич, подготовьте план работ.", 0, 30),
        Block("Ким С.А.", "Хорошо, план работ подготовлю к пятнице.", 30, 60),
    ], index=1, total=1)

    task = attach_source([Task("Подготовить план работ", "Сергей Анатольевич")], chunk)[0]
    assert task.who == "Сергей Анатольевич"


def test_speaker_taking_it_on_himself_stays_the_assignee():
    """«Сейчас уточним информацию, я вам доложу» — поручение, и исполнитель
    известен: тот, кто это сказал."""
    from minuteforge.tasks import attach_source

    chunk = Chunk([
        Block("Семенов А.В.", "По оборудованию сейчас уточним информацию и доложу.", 0, 40),
    ], index=1, total=1)

    task = attach_source([Task("Уточнить информацию", "Семенов А.В.")], chunk)[0]
    assert task.who == "Семенов А.В."


def test_declension_does_not_break_the_match():
    from minuteforge.tasks import _named_in

    assert _named_in("просьба к Огородниковой подготовить", "Огородникова Наталья")
    assert not _named_in("просьба подготовить справку", "Огородникова Наталья")


def test_a_statement_is_not_an_order():
    """Главный источник мусора в протоколе — не выдумки, а настоящие фразы,
    переписанные в повелительном наклонении."""
    from minuteforge.tasks import is_directive

    assert not is_directive("Видим показатели мощности.")
    assert not is_directive("Вот эта техника, она стоит третий год.")
    assert not is_directive("В МАКСе создана группа, где мы еженедельно отчитываемся.")
    assert not is_directive("Кто отвечает за вывоз крупногабаритного мусора?")

    assert is_directive("У меня просьба усилить мониторинг всего региона.")
    assert is_directive("Организуйте сейчас фотоотчет.")
    assert is_directive("Необходимо уже сейчас обеспечить выполнение объёма.")
    assert is_directive("Штаб должен начать работать по-другому.")


def test_a_greeting_is_not_an_imperative():
    """«Здравствуйте» кончается на «-йте» так же, как «зафиксируйте».

    Пока хвостовое правило этого не различало, вся перекличка штаба —
    полторы сотни реплик «здравствуйте, как видно, слышно» — считалась
    поручениями.
    """
    from minuteforge.tasks import is_directive

    assert not is_directive("Здравствуйте, Ленинградская область. Как нас видно, слышно?")
    assert not is_directive("Извините, повторите вопрос.")
    assert not is_directive("Подскажите, как слышно?")
    assert is_directive("Здравствуйте. Зафиксируйте это в протоколе."), (
        "приветствие рядом с поручением поручения не отменяет"
    )


def test_a_reproach_in_the_past_is_not_an_order():
    """«Должен был принять и рассчитаться» — упрёк за несделанное."""
    from minuteforge.tasks import is_directive

    assert not is_directive("Он должен был принять, проверить и рассчитаться.")


def test_orders_without_an_order_in_the_source_are_dropped():
    from minuteforge.tasks import Task, keep_directives

    kept = keep_directives([
        Task(what="Проанализировать показатели мощности", quote="Видим показатели мощности."),
        Task(what="Усилить мониторинг", quote="У меня просьба усилить мониторинг."),
        Task(what="Без источника проверить нечем", quote=""),
    ])

    assert [task.what for task in kept] == [
        "Усилить мониторинг", "Без источника проверить нечем",
    ]


def test_one_reply_gives_one_order():
    """Модель разбирает стенограмму с нахлёстом и одну фразу выписывает
    дважды разными словами. Для делопроизводителя это один пункт."""
    from minuteforge.tasks import Task, one_per_place

    quote = "Все фотографии отправьте губернатору с подробным отчётом."
    kept = one_per_place([
        Task(what="Отправить фотографии губернатору", quote=quote),
        Task(what="Отправить фотографии и подробный отчёт губернатору", who="Штаб", quote=quote),
    ])

    assert len(kept) == 1
    assert kept[0].who == "Штаб", "остаётся то, где назван исполнитель"


def test_different_replies_keep_their_own_orders():
    from minuteforge.tasks import Task, one_per_place

    kept = one_per_place([
        Task(what="Отправить фотографии", quote="Фотографии отправьте губернатору."),
        Task(what="Отправить фотографии", quote="И в министерство фотографии направьте."),
    ])

    assert len(kept) == 2


def test_reasoning_about_work_is_not_an_order():
    """«Если надо собрать всех участников — значит, это надо делать» сказано
    о работе вообще: ни кому, ни к какому сроку."""
    from minuteforge.tasks import is_directive

    assert not is_directive("Если надо собрать всех участников и держать на контроле, значит, это надо делать.")
    assert not is_directive("Ну если надо усилить контроль за проектировщиками, эту работу надо сделать.")
    assert is_directive("Если будут вопросы, прошу доложить отдельно."), "внутри условия бывает и поручение"


def test_the_agenda_is_not_an_order():
    """Распорядок совещания — не работа: «предлагается рассмотреть Пермский
    край» это следующий пункт повестки, а не поручение кому-то."""
    from minuteforge.tasks import is_directive

    assert not is_directive("Как раз предлагается рассмотреть Пермский край и Красноярский край.")
    assert not is_directive("Предлагаем коллег не заслушивать, они всё оперативно отработают.")
    assert is_directive("Предлагаю направить письмо в министерство с уточнённой цифрой.")
    assert is_directive("Предлагаю перейти к следующему вопросу и прошу подготовить справку.")


def test_a_one_word_scrap_yields_to_the_full_order():
    """«Передать» рядом с полным пунктом из той же фразы — обломок разбора."""
    from minuteforge.tasks import Task, one_per_place

    quote = "Поэтому передайте разговор, все фотографии отправьте губернатору."
    kept = one_per_place([
        Task(what="Передать", quote=quote),
        Task(what="Отправить фотографии губернатору", quote=quote),
    ])

    assert [task.what for task in kept] == ["Отправить фотографии губернатору"]


def test_three_orders_in_one_sentence_stay_three():
    """В одной фразе поручений бывает три, и короткое ничем не хуже длинного."""
    from minuteforge.tasks import Task, one_per_place

    quote = "Необходимо обновлять контейнерный парк, завершать создание площадок и обеспечивать вывоз КГО."
    kept = one_per_place([
        Task(what="Обновлять контейнерный парк", quote=quote),
        Task(what="Завершить создание площадок", quote=quote),
        Task(what="Обеспечивать вывоз КГО", quote=quote),
    ])

    assert len(kept) == 3


def test_the_quote_comes_with_its_neighbours():
    """Одного предложения для проверки мало.

    «Просьба подтвердить» само по себе не говорит, что подтвердить, — а с
    соседней фразой говорит всё. Человек, вычитывающий протокол, читает
    именно кусок, а не обрезанную фразу.
    """
    from minuteforge.blocks import Block
    from minuteforge.chunking import Chunk
    from minuteforge.tasks import Task, attach_source

    chunk = Chunk(blocks=[Block(
        "SPEAKER_08",
        "Цех компостирования готов на 95,7%. Просьба подтвердить. "
        "Следующий регион у нас Чувашия.",
        0, 60,
    )])
    task = attach_source([Task(what="Подтвердить готовность цеха")], chunk)[0]

    assert task.quote == "Просьба подтвердить."
    assert "95,7%" in task.context
    assert "Чувашия" in task.context


def test_a_filler_is_not_an_order():
    """«Да, смотрите…» и «давайте сэкономим время» — присказка.

    По ним в протоколе стояли пункты «Показать прессу» и «Экономить
    время»: слово обращено к вниманию, а не к работе.
    """
    from minuteforge.tasks import is_directive

    assert not is_directive("Это вот шестая. Сейчас, да, смотрите, а я пока покажу.")
    assert not is_directive("Принято. Давайте сэкономим время.")
    assert is_directive("Смотрите на этот слайд и подготовьте справку."), (
        "рядом с настоящим поручением присказка ему не мешает"
    )


def test_the_roll_call_asking_to_confirm_the_line_is_not_an_order():
    """«Красноярский край видно, слышно, подтвердите?» — это про связь.

    В протоколе это стояло пунктом «Подтвердить» с исполнителем-участником.
    Вне переклички то же слово поручает, и там оно остаётся.
    """
    from minuteforge.tasks import worth_showing

    assert not worth_showing("Добрый день, Красноярский край видно, слышно, поттвердите?")
    assert not worth_showing("Коллеги, просьба проверить микрофон.")
    assert worth_showing("Коллеги Ростовской области, просьба подтвердите срок ввода.")
    assert worth_showing("Слышно. Иванов, подготовьте справку.")


def test_the_addressee_is_taken_from_the_start_of_the_phrase():
    """Поручение открывается адресатом, и модель его почти не выписывает.

    Из семидесяти двух поручений исполнитель стоял у шестнадцати, хотя в
    самой реплике адресат назван.
    """
    from minuteforge.tasks import addressee

    assert addressee("Наталья Ковач, просьба обозначить срок ввода.") == "Наталья Ковач"
    assert addressee("Ильдар Наилевич, подтвердите наличие воды.") == "Ильдар Наилевич"
    assert addressee("Так, Ольга Аркадьевна, пригласите их на совещание.") == "Ольга Аркадьевна"
    # Обращение к направлению узнаётся по справочнику, а не по окончанию
    # слова: угадывать названия значит выдумывать адресата.
    from minuteforge.directory import Directory, Entry

    units = Directory(entries=(Entry("Северный филиал", ("северн",)),))
    assert addressee("Коллеги Северного филиала, просьба подтвердите срок.", units) == (
        "Северный филиал"
    )
    assert addressee("Коллеги Северного филиала, просьба подтвердите срок.") == (
        "все участники"
    ), "без справочника обращение к организации адресатом не считается"
    assert addressee("Коллеги, просьба подтвердить объекты.") == "все участники"


def test_a_name_in_the_middle_is_not_an_addressee():
    """Имя в середине фразы — чаще о ком говорят, чем кому поручают.

    Неверный адресат хуже пустого: пустой заставляет уточнить перед
    рассылкой, неверный уходит в рассылку как есть.
    """
    from minuteforge.tasks import addressee

    assert addressee("Принято, продолжаем дальше.") == ""
    assert addressee("Организуйте сейчас фотоотчет, вот прямо сейчас.") == ""
    assert addressee("По докладам, которые Виктор Аркадьевич предоставляет, видно.") == ""


def test_the_model_keeps_its_own_assignee():
    """Своё модель не переписывает: она видит всю фразу, правило — начало."""
    from minuteforge.tasks import Task, with_addressee

    tasks = [
        Task(what="Подтвердить срок", quote="Наталья Ковач, просьба подтвердить срок."),
        Task(what="Подтвердить срок", who="Ростовская область",
             quote="Наталья Ковач, просьба подтвердить срок."),
    ]
    filled = with_addressee(tasks)

    assert filled[0].who == "Наталья Ковач"
    assert filled[1].who == "Ростовская область"


def test_the_check_drops_a_report_retold_as_an_order():
    """Правилами доклад от поручения уже не отличить — спрашиваем модель."""
    from minuteforge.llm import Reply
    from minuteforge.tasks import Task, verify

    class Judge:
        def complete(self, system, user, **kwargs):
            return Reply(text='{"order": false}' if "отходы" in user else '{"order": true}')

    kept = verify([
        Task(what="Подготовить справку", quote="Подготовьте справку.", context="Подготовьте справку."),
        Task(what="Накапливать отходы раздельно", quote="Отходы должны накапливаться раздельно.",
             context="Отходы должны накапливаться раздельно."),
        Task(what="Проверить площадки", quote="Проверьте площадки.", context="Проверьте площадки."),
    ], Judge())

    assert [t.what for t in kept] == ["Подготовить справку", "Проверить площадки"]


def test_a_check_that_drops_everything_is_not_believed():
    """Мелкая модель, не поняв вопроса, отвечает «нет» подряд.

    Пустой протокол выглядит так, будто на совещании ничего не поручали, —
    и заметить подмену нельзя.
    """
    from minuteforge.llm import Reply
    from minuteforge.tasks import Task, verify

    class AlwaysNo:
        def complete(self, system, user, **kwargs):
            return Reply(text='{"order": false}')

    tasks = [Task(what=f"Поручение {i}", quote="Подготовьте.", context="Подготовьте.")
             for i in range(5)]

    assert verify(tasks, AlwaysNo()) == tasks


def test_a_silent_server_does_not_cost_an_order():
    from minuteforge.llm import LLMError
    from minuteforge.tasks import Task, verify

    class Silent:
        def complete(self, system, user, **kwargs):
            raise LLMError("сервер не ответил")

    tasks = [Task(what="Подготовить справку", quote="Подготовьте.", context="Подготовьте.")]
    assert verify(tasks, Silent()) == tasks


def test_a_one_word_order_is_not_an_order():
    """«Подтвердить» — обломок фразы «Просьба подтверждить», а не задание.

    Глагол без предмета нельзя ни разослать, ни проверить: что
    подтвердить, кому и к какому сроку — в таком пункте не сказано.
    """
    from minuteforge.tasks import Task, keep_meaningful

    kept = keep_meaningful([
        Task(what="Подтвердить"),
        Task(what="Подтвердить финансирование"),
        Task(what="Организовать фотоотчет"),
        Task(what="Доложить"),
    ])

    assert [t.what for t in kept] == ["Подтвердить финансирование", "Организовать фотоотчет"]


def test_the_chair_is_not_given_orders():
    """Докладчик открывает речь обращением к ведущему.

    «Хамзатбек Ханифович, ранее отмечали… Предлагаю уточнить у региона, кто
    отвечает за площадки» — поручение здесь региону, а в протоколе стоял
    исполнителем ведущий.
    """
    from minuteforge.blocks import Block
    from minuteforge.tasks import Task, most_addressed, with_addressee

    blocks = [
        Block("SPEAKER_12", "Гамзатбек Ханифович, разрешите доложить.", 0, 10),
        Block("SPEAKER_08", "Гамзатбек Ханифович, объект готов на 84%.", 10, 20),
        Block("SPEAKER_11", "Гамзатбек Ханифович, если позволите.", 20, 30),
    ]
    chair = most_addressed(blocks)
    assert chair == "Гамзатбек Ханифович"

    task = Task(
        what="Уточнить, кто отвечает за площадки",
        quote="Гамзатбек Ханифович, предлагаю уточнить у региона.",
    )
    assert with_addressee([task], chair=chair)[0].who == ""
    assert with_addressee([task])[0].who == "Гамзатбек Ханифович", (
        "без ведущего правило работает как прежде"
    )


def test_a_single_greeting_does_not_make_a_chair():
    """К ведущему обращаются весь штаб, а случайное обращение звучит раз."""
    from minuteforge.blocks import Block
    from minuteforge.tasks import most_addressed

    assert most_addressed([Block("S1", "Иван Петрович, доложите.", 0, 5)]) == ""


def test_one_chair_heard_four_ways_is_one_person():
    """Распознавание коверкает имя, и ведущий приезжает четырьмя людьми.

    Сравнивать строки целиком нельзя: «Андрей Петрович» и «Антон
    Николаевич» похожи сильнее, чем «Гамзатбек Ханифович» и «Хамзатбег
    Кариллович», — а первые двое разные люди. Различает их имя, не отчество.
    """
    from minuteforge.tasks import same_person

    assert same_person("Гамзатбек Ханифович", "Хамзатбег Кариллович")
    assert same_person("Гамзатбек Ханифович", "Хамзатбек Ханифович")
    assert same_person("Ильдар Наилевич", "Ильдар Наилович")
    assert not same_person("Андрей Петрович", "Антон Петрович")
    assert not same_person("Андрей Петрович", "Анатолий Петрович")


def test_the_deadline_is_taken_from_the_phrase():
    """Модель почти не заполняет срок: он стоял у одного поручения из сорока.

    А вслух он звучит: «через две недели», «сегодня же», «на 30 ноября».
    Правилу это по силам, как и адресату.
    """
    from minuteforge.tasks import spoken_due

    assert spoken_due("Через две недели вернёмся на штаб.") == "Через две недели"
    assert spoken_due("Отошлите фотографии прям сегодня же.") == "сегодня же"
    assert spoken_due("Просьба подтвердите срок ввода на 30 ноября.") == "на 30 ноября"
    assert spoken_due("Обращайтесь в течение ближайших двух недель.") == (
        "в течение ближайших двух недель"
    )
    assert spoken_due("Организуйте сейчас фотоотчёт.") == ""


def test_a_date_looking_back_is_not_a_deadline():
    """«По информации на 2 сентября готовность 84%» — ссылка на прошлое.

    Сроком назад не назначают, а в таблице контроля такая дата выглядит
    просроченным поручением.
    """
    from minuteforge.tasks import spoken_due

    assert spoken_due("По информации на 2 сентября строительная готовность 84%.") == ""
    assert spoken_due(
        "По информации на 2 сентября готовность 84%. Просьба подтвердить срок на 30 ноября."
    ) == "на 30 ноября"


def test_the_model_keeps_its_own_deadline():
    from minuteforge.tasks import Task, with_due

    tasks = [
        Task(what="Подготовить справку", quote="Подготовьте справку. До среды."),
        Task(what="Подготовить справку", due="до пятницы",
             quote="Подготовьте справку. До среды."),
    ]
    filled = with_due(tasks)

    assert filled[0].due == "До среды"
    assert filled[1].due == "до пятницы"


def test_one_phrase_gives_one_item():
    """Три действия одной фразой — одно поручение, а не три строки.

    Адресат у них один и срок один; тремя строками в таблице контроля это
    только запутает.
    """
    from minuteforge.tasks import Task, one_per_phrase

    quote = (
        "Передайте разговор, все фотографии отправьте губернатору с подробным "
        "отчётом, кратно усиленным мониторингом муниципальных образований."
    )
    kept = one_per_phrase([
        Task(what="Передать разговор", quote=quote, at=10.0),
        Task(what="Отправить фотографии губернатору", quote=quote, who="Ставропольский край"),
        Task(what="Усилить мониторинг муниципальных образований", quote=quote, due="сегодня"),
    ])

    assert len(kept) == 1
    assert kept[0].what == (
        "Передать разговор; отправить фотографии губернатору; "
        "усилить мониторинг муниципальных образований"
    )
    assert kept[0].who == "Ставропольский край", "адресат берётся оттуда, где он назван"
    assert kept[0].due == "сегодня"
    assert kept[0].at == 10.0


def test_orders_from_different_phrases_stay_apart():
    from minuteforge.tasks import Task, one_per_phrase

    kept = one_per_phrase([
        Task(what="Подготовить справку", quote="Подготовьте справку."),
        Task(what="Организовать фотоотчёт", quote="Организуйте фотоотчёт."),
    ])

    assert len(kept) == 2


def test_a_paragraph_without_full_stops_is_not_merged():
    """«Одна фраза» решается по точкам, а расставляет их распознавание.

    Там, где две минуты речи приехали одним предложением, сведение дало бы
    пункт, по которому нельзя спросить. Лучше оставить раздельно.
    """
    from minuteforge.tasks import Task, one_per_phrase

    quote = "И вот " + "дальше говорим про объекты и площадки " * 10
    group = [
        Task(what="Обновить контейнерный парк во всех муниципальных образованиях округа",
             quote=quote),
        Task(what="Завершить создание контейнерных площадок по утверждённому графику",
             quote=quote),
        Task(what="Обеспечить своевременный вывоз крупногабаритных отходов из дворов",
             quote=quote),
        Task(what="Представить фотоматериалы по каждой площадке отдельным отчётом",
             quote=quote),
        Task(what="Согласовать с региональным оператором маршруты вывоза на зимний период",
             quote=quote),
    ]

    assert len(one_per_phrase(group)) == 5


def test_the_same_action_said_twice_is_not_doubled():
    from minuteforge.tasks import Task, one_per_phrase

    quote = "Организуйте фотоотчёт прямо сейчас."
    kept = one_per_phrase([
        Task(what="Организовать фотоотчёт прямо сейчас", quote=quote),
        Task(what="Организовать фотоотчёт", quote=quote),
    ])

    assert kept[0].what == "Организовать фотоотчёт прямо сейчас"


def test_a_reflexive_imperative_is_a_directive_too():
    """«-йтесь» — то же повелительное, что и «-йте», только у возвратных.

    «Постарайтесь в ноябре все завершить» поручением не считалось: правило
    знало окончание «-йте» и не знало «-йтесь».
    """
    from minuteforge.tasks import is_directive

    assert is_directive("Постарайтесь, наверное, в ноябре уже все завершить.")
    assert is_directive("Занимайтесь этим вопросом сами.")
    assert not is_directive("Вы находитесь на объекте раз в месяц.")


def test_the_plural_of_a_word_of_request_counts_as_well():
    """В списке были «предлагаю» и «предлагается», не было «предлагаем» —
    и «Предлагаем вынести на следующий штаб» проходило мимо."""
    from minuteforge.tasks import is_directive

    assert is_directive("Предлагаем их вынести на следующий штаб.")
    assert is_directive("Рекомендуем посмотреть на этот опыт.")


def test_a_road_across_is_not_a_deadline():
    """«Через дорогу от сквера» разбиралось как срок, и пункт получал
    графу «Срок: через дорогу». После «через» нужна единица времени."""
    from minuteforge.tasks import spoken_due

    assert spoken_due("Через дорогу от сквера возле контейнеров навалы.") == ""
    assert spoken_due("Через две недели вернёмся.") == "Через две недели"
    assert spoken_due("Обещали через месяц.") == "через месяц"
    assert spoken_due("Через 20 дней вернёмся в порядке мониторинга.") == "Через 20 дней"


def test_today_is_a_deadline_only_in_its_own_phrase():
    """«Сегодня» — единственное слово из списка сроков, которое в русском
    чаще наречие: «стандарт, который сегодня будет в стране». Из соседней
    фразы оно давало пункту сроком день совещания, которого никто не
    назначал."""
    from minuteforge.tasks import with_due

    own = Task(
        what="Отправить фотографии губернатору",
        quote="Все эти фотографии отошлите к губернатору прям сегодня же.",
        context="Мы знаем губернатора. Все эти фотографии отошлите к губернатору "
        "прям сегодня же. Дальше.",
    )
    neighbour = Task(
        what="Пригласить на свой штаб и разобраться",
        quote="Тогда пригласите на свой штаб и разберитесь.",
        context="Тогда пригласите на свой штаб и разберитесь. Уже три месяца "
        "осталось, чтобы сформировать стандарт, который сегодня будет в стране.",
    )

    assert with_due([own])[0].due == "сегодня же"
    assert with_due([neighbour])[0].due == "", "«сегодня» из соседней фразы — наречие"


def test_a_named_day_from_a_neighbouring_phrase_still_counts():
    """Настоящий срок из соседней фразы берётся как и раньше: «Подготовьте
    справку. До среды» — обычный способ его назначить."""
    from minuteforge.tasks import with_due

    task = Task(
        what="Завершить работы по объекту",
        quote="Постарайтесь в ноябре уже все завершить.",
        context="Постарайтесь в ноябре уже все завершить. Более детально через "
        "две недели.",
    )
    assert with_due([task])[0].due == "через две недели"


def test_a_phrase_with_a_named_day_is_shown_to_the_model():
    """Поручение сплошь и рядом стоит без глагола: «и через неделю на
    доклад замгубернатора». Правилом такое не берётся, а срок в нём назван
    прямо — значит, показать модели стоит."""
    from minuteforge.tasks import is_directive, worth_showing

    phrase = "И через неделю на доклад замгубернатора с полным отчётом."
    assert not is_directive(phrase), "повелительного наклонения тут нет"
    assert worth_showing(phrase)
    # А «сегодня» на это права не даёт: иначе в отбор попадёт каждая
    # вторая фраза доклада.
    assert not worth_showing("По представленным сегодня фотоматериалам видим рост.")


def test_a_named_day_saves_a_task_from_the_sieve_but_a_cadence_does_not():
    """Отсев не поручений теперь пропускает фразу с названным днём — иначе
    единственный пункт со сроком терялся. Порядок работы в это не входит:
    «мониторинг производится ежедневно» — описание, а не задание."""
    from minuteforge.tasks import keep_directives

    named = Task(
        what="Доложить заместителю губернатора",
        quote="И через неделю на доклад замгубернатора с полным отчётом.",
    )
    cadence = Task(
        what="Производить мониторинг",
        quote="Мониторинг производится ежедневно посредством выезда обращений.",
    )

    assert keep_directives([named]) == [named]
    assert keep_directives([cadence]) == []


def test_a_softly_worded_requirement_is_still_a_directive():
    """«Обращаю внимание региона на планы по строительству площадок» и
    «желательно, чтобы штаб проводил лично губернатор» — так поручают на
    совещании чаще, чем «обеспечьте»."""
    from minuteforge.tasks import is_directive

    assert is_directive("Отдельно обращаю внимание региона на планы по площадкам.")
    assert is_directive("Особо обращаем внимание на финансирование, которое заявлено.")
    assert is_directive("И желательно, чтобы штаб проводил лично руководитель.")


def test_a_task_ordered_by_the_neighbouring_phrase_survives():
    """Цитата выбирается по совпадению слов, и ею становится фраза доклада —
    слов в ней больше. А велено соседней. Отсев смотрел на цитату и снимал
    настоящее поручение."""
    from minuteforge.tasks import is_directive, keep_directives, ordered_nearby

    task = Task(
        what="Завершить работы по наружным сетям и благоустройству",
        quote="Я так понимаю, вы отстаете по наружным сетям и по благоустройству.",
        context="Мы видим красивый объект. Я так понимаю, вы отстаете по наружным "
        "сетям и по благоустройству. Завершайте быстрее.",
    )
    assert not is_directive(task.quote), "сама цитата поручением не звучит"
    assert ordered_nearby(task)
    assert keep_directives([task]) == [task]


def test_someone_elses_order_nearby_does_not_save_a_report():
    """Проверка узкая: рядом должно стоять то же действие, что в пункте.
    Иначе любой пункт спасался бы соседним поручением о другом — окна
    собираются вокруг таких фраз, и рядом они почти всегда."""
    from minuteforge.tasks import keep_directives, ordered_nearby

    report = Task(
        what="Проанализировать показатели мощности",
        quote="Видим показатели мощности.",
        context="Видим показатели мощности. Завершайте быстрее.",
    )
    assert not ordered_nearby(report)
    assert keep_directives([report]) == []


def test_a_task_with_a_named_day_is_not_put_to_the_check():
    """Названный день ставят поручению, а не докладу. На живой записи
    проверка сняла «принять информацию через неделю на контроль» — ровно
    тот пункт, ради которого графа сроков и заведена."""
    from minuteforge.tasks import verify

    class Refuses:
        settings = None

        def complete(self, system, user, **kwargs):
            return Reply(text='{"order": false}')

    dated = Task(what="Представить углублённый доклад", due="через неделю",
                 quote="Информацию принимаем через неделю на контроль.",
                 context="Информацию принимаем через неделю на контроль.")
    plain = Task(what="Проанализировать показатели", due="",
                 quote="Видим показатели мощности.",
                 context="Видим показатели мощности. Дальше цифры.")

    # Пунктов со сроком двое: иначе сработает защита от модели, которая
    # отвечает «нет» подряд, и проверка отменится целиком.
    kept = verify([dated, dated, plain, plain], Refuses())
    assert dated in kept, "пункт со сроком проверка не трогает"
    assert plain not in kept


def test_the_check_can_be_done_by_another_model():
    """Выписывать и проверять — разные задачи. Щедрая модель находит
    больше, строгая реже ошибается; когда это две разные модели, каждая
    делает своё."""
    from minuteforge.chunking import Chunk
    from minuteforge.config import Settings

    class Writer:
        settings = Settings(llm_json_mode=False, verify_tasks=True, merge_similar=False)

        def __init__(self):
            self.unloaded = False
            self.asked = []

        def complete(self, system, user, **kwargs):
            self.asked.append(user)
            if user == "Готов?":
                return Reply(text="Да")
            return Reply(text="Поручение: Подготовить план работ\nКому: \nСрок: ")

        def unload(self):
            self.unloaded = True
            return True

    class Checker:
        settings = Writer.settings

        def __init__(self):
            self.asked = []

        def complete(self, system, user, **kwargs):
            self.asked.append(user)
            return Reply(text="да")

    writer, checker = Writer(), Checker()
    chunk = Chunk(
        blocks=[Block("SPEAKER_02", "Иванов, подготовьте план работ.", 0.0, 10.0)],
        index=1, total=1,
    )
    found = extract_tasks([chunk], writer, verifier=checker)

    assert [t.what for t in found] == ["Подготовить план работ"]
    assert checker.asked, "проверять должна вторая модель"
    assert not any("Стенограмма:" in q for q in checker.asked), (
        "выписывать её не просили"
    )
    assert writer.unloaded, "первую модель надо выгрузить, иначе двум не хватит карты"


def test_one_model_does_both_when_no_second_is_set():
    """Без отдельной модели проверки ничего не меняется — и выгружать
    нечего: та же модель работает дальше."""
    from minuteforge.chunking import Chunk
    from minuteforge.config import Settings

    class Both:
        settings = Settings(llm_json_mode=False, verify_tasks=True, merge_similar=False)

        def __init__(self):
            self.unloaded = False

        def complete(self, system, user, **kwargs):
            if user == "Готов?":
                return Reply(text="Да")
            if "Выписанный пункт" in user:
                return Reply(text="да")
            return Reply(text="Поручение: Подготовить план работ\nКому: \nСрок: ")

        def unload(self):
            self.unloaded = True
            return True

    client = Both()
    chunk = Chunk(
        blocks=[Block("SPEAKER_02", "Иванов, подготовьте план работ.", 0.0, 10.0)],
        index=1, total=1,
    )
    found = extract_tasks([chunk], client)

    assert [t.what for t in found] == ["Подготовить план работ"]
    assert not client.unloaded
