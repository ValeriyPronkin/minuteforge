"""Извлечение поручений из стенограммы.

Модель просят не пересказать совещание, а найти в нём поручения: кому, что и
к какому сроку. Ответ спрашивается строками, а не JSON: небольшая местная
модель ломает структуру на первом же длинном тексте, а строку вида
``Поручение: … Кому: … Срок: …`` она держит устойчиво. Разбор здесь
соответственно снисходительный — модель почти всегда добавит от себя
вступление, нумерацию или заголовок, и это не повод потерять поручение.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Sequence

from loguru import logger

from .blocks import UNKNOWN
from .chunking import Chunk
from .llm import LLMClient, LLMError
from .progress import Progress, Step, report

#: Что модель отвечает, когда поручений в куске нет. Совещание наполовину
#: состоит из обсуждения, и «ничего не найдено» — нормальный ответ, а не сбой.
NOTHING_FOUND = "НЕТ ПОРУЧЕНИЙ"

EXTRACT_SYSTEM = f"""Ты — секретарь совещания. Твоя задача — найти в стенограмме поручения:
что поручено, кому и к какому сроку.

Правила:
1. Выписывай только то, что прозвучало. Не додумывай исполнителей и сроки.
2. Одно поручение — один блок из трёх строк:
Поручение: <что сделать>
Кому: <кому поручено; если не названо — «не указано»>
Срок: <когда; если не назван — «не указан»>
3. Блоки разделяй пустой строкой.
4. Если поручений в тексте нет, ответь ровно: {NOTHING_FOUND}
5. Не пиши ничего, кроме блоков поручений.

Поручение — это указание совершить действие: «подготовьте», «направьте»,
«обеспечьте», «доложите». Мнение, вопрос или обсуждение поручением не являются.

Пример.

Стенограмма:
Орлов В.П.: Сергей, подготовьте план работ по выгрузке и согласуйте его со мной до пятницы.
Сергей Ким: Принято.
Орлов В.П.: Ещё нужна справка по инцидентам, без неё приёмку не пройдём.
Сергей Ким: А кто её собирает?
Орлов В.П.: Решим отдельно.

Ответ:
Поручение: Подготовить план работ по выгрузке и согласовать
Кому: Сергей Ким
Срок: до пятницы

Поручение: Собрать справку по инцидентам
Кому: не указано
Срок: не указан"""

EXTRACT_SYSTEM_JSON = """Ты — секретарь совещания. Найди в стенограмме поручения.

Поручение — это указание совершить действие. На деловых совещаниях оно
звучит по-разному, и все эти формы считаются поручением:

«подготовьте», «направьте», «обеспечьте», «доложите», «представьте»;
«просьба сделать», «прошу зафиксировать», «большая просьба и даже требование»;
«проконтролируйте», «уточнитесь», «свяжитесь», «проработайте», «ускорьте»;
«выносим в протокол», «в протокольное решение», «отфиксируем»;
а также обещание самого выступающего: «направим сегодня письмо», «доложу отдельно».

Мнение, вопрос и обсуждение поручениями не являются. Констатация — тоже:
«одобрен лимит» это не поручение подготовить лимит.

Не пересказывай и не переписывай стенограмму. Не описывай, о чём шла речь.
Выписывай только поручения. Не додумывай исполнителей и сроки: чего не
прозвучало, оставь пустым.

Отвечай по-русски. Стенограмма русская, и поручения должны быть выписаны
теми же словами, какими прозвучали.

Ответь строго в таком виде, без единого слова вокруг:

{"tasks": [{"what": "что сделать", "who": "кому", "due": "к какому сроку"}]}

Если поручений нет, ответь: {"tasks": []}

Пример.

Стенограмма:
Орлов В.П.: Сергей, подготовьте план работ и согласуйте со мной до пятницы.
Сергей Ким: Принято.
Орлов В.П.: Ещё нужна справка по инцидентам.
Сергей Ким: А кто её собирает?
Орлов В.П.: Решим отдельно.
Орлов В.П.: И просьба всем работать системно, объёмы не снижать.
Ким С.А.: Направим сегодня письмо в министерство с уточнённой цифрой.
Орлов В.П.: В протокол выносим пункт: службе ускорить выдачу выписки.

Ответ:
{"tasks": [{"what": "Подготовить план работ и согласовать", "who": "Сергей Ким", "due": "до пятницы"}, {"what": "Собрать справку по инцидентам", "who": "", "due": ""}, {"what": "Работать системно, не снижать объёмы", "who": "", "due": ""}, {"what": "Направить письмо в министерство с уточнённой цифрой", "who": "Ким С.А.", "due": "сегодня"}, {"what": "Ускорить выдачу выписки", "who": "служба", "due": ""}]}"""

#: Схема ответа. Сервер держит модель в её рамках, отсекая на каждом шаге
#: всё, что в неё не укладывается: придумать свою структуру или свалиться в
#: пересказ становится физически невозможно.
TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "who": {"type": "string"},
                    "due": {"type": "string"},
                },
                "required": ["what", "who", "due"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

_FIELD_ALIASES = {
    "what": ("поручение", "задача", "что сделать", "task"),
    "who": ("кому", "исполнитель", "ответственный", "assignee"),
    "due": ("срок", "к какому сроку", "дата", "due"),
}

_NOT_SET = {"не указано", "не указан", "не указана", "нет", "-", "—", "не назван", "none"}

_LINE = re.compile(r"^\s*[-*\d.)\s]*([А-Яа-яЁёA-Za-z ]+?)\s*:\s*(.+?)\s*$")


@dataclass
class Task:
    """Одно поручение."""

    what: str
    who: str = ""
    due: str = ""
    #: Из какого куска стенограммы взято.
    chunk: int = 0
    #: Секунда записи, где это прозвучало. Главное поле для проверки: без
    #: него человеку приходится искать место в двухчасовом видео, и проверка
    #: восьмидесяти пунктов съедает больше времени, чем сэкономил разбор.
    at: float | None = None
    #: Реплика, из которой поручение выписано, как она есть в стенограмме.
    #: По ней видно сразу, поручение это или изложение доклада.
    quote: str = ""
    #: Кто это сказал.
    said_by: str = ""

    @property
    def key(self) -> str:
        """Ключ для сравнения поручений между собой."""
        return _normalize(self.what)

    def as_line(self) -> str:
        parts = [self.what]
        if self.who:
            parts.append(f"исполнитель: {self.who}")
        if self.due:
            parts.append(f"срок: {self.due}")
        return "; ".join(parts)


def build_prompt(
    chunk: Chunk,
    *,
    json_mode: bool = False,
    extra: str = "",
    base: str = "",
) -> tuple[str, str]:
    """Собирает пару «системный промпт, текст запроса» для куска.

    Модели сообщается, что она видит часть совещания, а не всё: иначе она
    склонна писать итоговый протокол по обрывку и придумывать недостающее.
    """
    # UNKNOWN — не человек, а признак того, что диаризация не опознала
    # реплику. В списке участников он выглядит фамилией, и модель начинает
    # приписывать ему поручения.
    named = [s for s in chunk.speakers if s != UNKNOWN]
    header = ""
    if chunk.total > 1:
        header = f"Фрагмент {chunk.index} из {chunk.total}. "
        if named:
            header += f"Участники фрагмента: {', '.join(named)}."
        header += "\n\n"
    system = base or (EXTRACT_SYSTEM_JSON if json_mode else EXTRACT_SYSTEM)
    if extra.strip():
        # В конец: последнее указание модель держит лучше всего. И отдельным
        # разделом, чтобы человек, дописавший своё, видел, где оно кончается
        # и начинается наше.
        system = f"{system}\n\nДополнительно:\n{extra.strip()}"
    return system, f"{header}Стенограмма:\n{chunk.text}"


def parse_tasks(answer: str, *, chunk: int = 0) -> list[Task]:
    """Разбирает ответ модели в список поручений.

    Разбор терпит вступления, нумерацию, маркеры списка и лишние заголовки:
    формат ответа модели — просьба, а не гарантия. Единственное, чего он не
    прощает, — отсутствия самого текста поручения: блок без «Поручение»
    пропускается, потому что исполнитель без задачи бессмыслен.
    """
    text = (answer or "").strip()
    if not text:
        return []

    from_json = _parse_json(text, chunk)
    if from_json is not None:
        return from_json

    tasks: list[Task] = []
    current: dict[str, str] = {}

    for raw in text.splitlines():
        line = raw.strip().strip("`")
        if not line:
            if current.get("what"):
                tasks.append(_build(current, chunk))
            current = {}
            continue

        match = _LINE.match(line)
        if not match:
            continue
        field = _field_of(match.group(1))
        if field is None:
            continue
        # Второе «Поручение» без пустой строки между блоками: модель забыла
        # разделитель, но новое поручение началось.
        if field == "what" and current.get("what"):
            tasks.append(_build(current, chunk))
            current = {}
        current[field] = match.group(2).strip()

    if current.get("what"):
        tasks.append(_build(current, chunk))

    if tasks:
        return tasks

    # Разобрать нечего. Отличаем честное «поручений нет» от ответа, который
    # мы просто не поняли: мелкая модель охотно пересказывает задание перед
    # ответом, и фраза-маркер попадает в текст вместе с настоящими
    # поручениями. Поэтому маркер проверяется последним, а не первым.
    if NOTHING_FOUND.lower() in text.lower():
        return []

    if text.lstrip().startswith("{"):
        # JSON есть, а нашего в нём нет: модель придумала свою структуру.
        # Так ведёт себя мелкая модель, когда сервер требует «какой-нибудь
        # JSON», но не держит схему.
        logger.warning(
            "Модель вернула JSON не по схеме — поручений в нём нет. "
            "Проверьте, что сервер понимает строгую схему ответа. Ответ: {}",
            text[:200].replace("\n", " "),
        )
    else:
        logger.warning(
            "Ответ модели не разобран, поручений не извлечено. Ответ: {}",
            text[:300].replace("\n", " "),
        )
    return []


def is_russian(text: str) -> bool:
    """Русский ли это текст.

    Мелкая модель на русской стенограмме то и дело сваливается в английский,
    и никакие просьбы в промпте этого не отменяют. Проверка механическая:
    считаем долю кириллицы среди букв.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    cyrillic = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return cyrillic / len(letters) >= 0.5


#: Кого нельзя записать в исполнители. Это не люди и не подразделения, а
#: обращения: поручение «коллегам» разослать некому, и в графе контроля от
#: него один вред — оно выглядит назначенным, а спросить не с кого.
_NOT_AN_ASSIGNEE = {
    "мы", "вы", "они", "все", "коллеги", "уважаемые коллеги", "участники",
    "участники совещания", "присутствующие", "всем", "нам", "ассистент",
    "assistant", "команда", "секретарь совещания",
}

#: По этим приметам строка похожа на срок. Модель кладёт в это поле что
#: угодно — «Завершена», «Отправлено», «Добрый день», — и такой «срок» в
#: таблице контроля хуже пустого: по нему нельзя ни спросить, ни отсортировать.
_LOOKS_LIKE_DUE = (
    "до ", "к ", "через ", "сегодня", "завтра", "недел", "месяц", "квартал",
    "год", "числ", "янв", "фев", "март", "апрел", "ма", "июн", "июл", "авг",
    "сент", "октяб", "нояб", "декаб", "понедельник", "вторник", "сред",
    "четверг", "пятниц", "суббот", "воскресен", "срок", "постоянно",
    "ежемесячн", "еженедельн", "немедленн", "срочно",
)


def clean_due(due: str) -> str:
    """Оставляет срок, только если это похоже на срок.

    Модель охотно заполняет это поле состоянием («Завершена», «В процессе»)
    или обрывком реплики («Добрый день»). В таблице контроля такое хуже
    пустого: строка выглядит заполненной, а спросить по ней нечего.
    """
    due = (due or "").strip()
    if not due or not is_russian(due):
        return ""
    if len(due.split()) > 8:
        # Срок — это «до пятницы» или «15 сентября», а не абзац. Модель
        # кладёт сюда кусок доклада, и в таблице контроля он занимает три
        # строки, ничего при этом не сообщая.
        return ""
    lowered = due.lower()
    if any(ch.isdigit() for ch in lowered):
        return due
    return due if any(mark in lowered for mark in _LOOKS_LIKE_DUE) else ""


def _significant(text: str) -> set[str]:
    """Слова, по которым можно узнать, о том ли речь.

    Короткие отбрасываются: «в», «на», «для» есть в любом тексте и ничего не
    доказывают. Берётся начало слова, чтобы падежи не мешали: «поручение» и
    «поручения» — одно и то же.
    """
    return {word[:6] for word in re.findall(r"\w{5,}", (text or "").lower())}


def grounded(task: Task, source: str) -> bool:
    """Опирается ли поручение на текст, из которого взято.

    Это и есть проверка на выдумку. Модель, потерявшая нить, сочиняет
    правдоподобные поручения из ничего — «Analyze the presentation,
    исполнитель Assistant, срок 2022-08-15» — и отличить их от настоящих
    можно ровно одним способом: посмотреть, есть ли эти слова в стенограмме.

    Порог невысокий: модель имеет право переформулировать, и требовать
    дословности значило бы выбрасывать хорошие поручения.
    """
    words = _significant(task.what)
    if not words:
        return False
    found = words & _significant(source)
    return len(found) / len(words) >= 0.3


def clean_assignee(who: str, source: str) -> str:
    """Оставляет исполнителя, только если он в стенограмме и похож на имя.

    Модель кладёт в это поле целые предложения — «The team should analyze
    the current status…» — и выдумывает адресатов, которых на совещании не
    было. Ни то, ни другое в графу «Исполнитель» не годится.
    """
    who = (who or "").strip()
    if not who:
        return ""
    if who.lower().strip(".,") in _NOT_AN_ASSIGNEE:
        return ""
    if len(who.split()) > 6:
        return ""  # это предложение, а не исполнитель
    if not is_russian(who):
        return ""
    known = _significant(source)
    words = _significant(who)
    if words and not (words & known):
        return ""  # такого на совещании не называли
    return who


def extract_tasks(
    chunks: Sequence[Chunk],
    client: LLMClient,
    *,
    skip_failed: bool = True,
    progress: Progress | None = None,
    answers: list[str] | None = None,
    corpus: str | None = None,
) -> list[Task]:
    """Проходит по кускам стенограммы и собирает поручения.

    :param skip_failed: что делать, если модель не ответила на кусок. По
        умолчанию — пропустить и идти дальше: терять весь час работы из-за
        одного сбоя хуже, чем недосчитаться поручений из одного фрагмента.
        Пропуск пишется в журнал, молча это не проходит.
    :param progress: куда сообщать о ходе. Местная модель отвечает по
        десятку секунд на фрагмент, и на длинном совещании это минуты
        тишины, за которые человек успевает решить, что всё повисло.
    :param corpus: вся стенограмма целиком — по ней проверяются исполнители.
        Человека называют один раз, в начале совещания, а поручение он
        получает часом позже, в другом фрагменте; сверяться только с
        фрагментом значило бы вычёркивать верные имена.
    :param answers: куда сложить ответы модели как есть. Когда поручений не
        нашлось, это единственный способ понять почему: модель могла
        ответить прозой, по-английски или пересказать задание вместо
        ответа — и всё это выглядит одинаково, как пустой результат.
    """
    collected: list[Task] = []
    total = len(chunks) or 1

    for position, chunk in enumerate(chunks, 1):
        title = f"Фрагмент {chunk.index} из {chunk.total}"
        json_mode = getattr(client, "settings", None) and client.settings.llm_json_mode
        report(progress, Step(name="chunk", title=title, share=(position - 1) / total))

        started = time.perf_counter()
        settings = getattr(client, "settings", None)
        system, user = build_prompt(
            chunk,
            json_mode=bool(json_mode),
            extra=getattr(settings, "prompt_extra", "") or "",
            base=_own_prompt(settings),
        )
        try:
            reply = client.complete(
                system, user,
                json_mode=bool(json_mode),
                schema=TASKS_SCHEMA if json_mode else None,
            )
        except LLMError as exc:
            if not skip_failed:
                raise
            logger.warning("Фрагмент {}/{} пропущен: {}", chunk.index, chunk.total, exc)
            report(progress, Step(
                name="chunk",
                title=f"{title} — пропущен: {exc}",
                done=True,
                elapsed_s=round(time.perf_counter() - started, 1),
                share=position / total,
            ))
            continue

        if answers is not None:
            answers.append(reply.text)
        parsed = parse_tasks(reply.text, chunk=chunk.index)
        found = attach_source(
            _keep_sound(parsed, chunk.text, corpus or chunk.text), chunk
        )

        if reply.truncated:
            logger.warning(
                "Фрагмент {}: ответ обрезан по длине, спасено поручений {}. "
                "Помогает больший llm_max_tokens или фрагменты помельче.",
                chunk.index, len(parsed),
            )

        # Модель ответила по-английски, и от фрагмента ничего не осталось.
        # Просьба в промпте её не удержала, но повтор с требованием в самом
        # конце запроса — обычно да: последнее указание весит больше всего.
        if parsed and not found and not any(is_russian(t.what) for t in parsed):
            logger.info("Фрагмент {}: ответ по-английски, повторяю запрос", chunk.index)
            try:
                again = client.complete(
                    system,
                    f"{user}\n\nВНИМАНИЕ: ответ должен быть на русском языке.",
                    json_mode=bool(json_mode),
                    schema=TASKS_SCHEMA if json_mode else None,
                )
            except LLMError as exc:
                logger.warning("Повтор не удался: {}", exc)
            else:
                if answers is not None:
                    answers.append(again.text)
                found = attach_source(
                    _keep_sound(
                        parse_tasks(again.text, chunk=chunk.index),
                        chunk.text,
                        corpus or chunk.text,
                    ),
                    chunk,
                )
        logger.info(
            "Фрагмент {}/{}: поручений {}, ответ за {} с",
            chunk.index, chunk.total, len(found), reply.elapsed_s,
        )
        collected.extend(found)
        report(progress, Step(
            name="chunk",
            title=f"{title} — поручений {len(found)}",
            done=True,
            elapsed_s=round(time.perf_counter() - started, 1),
            share=position / total,
        ))

    merged = dedupe(collected)
    if settings is not None and getattr(settings, "merge_similar", False) and merged:
        merged = merge_similar(merged, client, json_mode=bool(json_mode))
    return merged


def _own_prompt(settings) -> str:
    """Своё системное указание из файла, если оно задано."""
    path = getattr(settings, "prompt_file", None)
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        logger.warning("Не прочитать {}, беру встроенное указание: {}", path, exc)
        return ""


def attach_source(tasks: list[Task], chunk: Chunk) -> list[Task]:
    """Привязывает поручение к реплике, из которой оно выписано.

    Ищется реплика с наибольшим совпадением по значимым словам. Способ
    грубый, но проверять его дёшево: человек видит цитату рядом с
    поручением и сразу понимает, то ли это место.

    Без привязки проверка выглядит так: восемьдесят пунктов и двухчасовая
    запись, ищите сами. С привязкой — взгляд на цитату и, если сомнительно,
    переход к отметке времени.
    """
    attached = []
    for task in tasks:
        words = _significant(task.what)
        best, score = None, 0.0
        for block in chunk.blocks:
            if not words:
                break
            overlap = len(words & _significant(block.text)) / len(words)
            if overlap > score:
                best, score = block, overlap
        attached.append(Task(
            what=task.what, who=task.who, due=task.due, chunk=task.chunk,
            at=best.start if best is not None else None,
            quote=_short_quote(best.text) if best is not None else "",
            said_by=best.speaker if best is not None else "",
        ))
    return attached


def _short_quote(text: str, limit: int = 220) -> str:
    """Цитата достаточной длины, чтобы понять, о чём речь, но не абзац."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _keep_sound(tasks: list[Task], source: str, corpus: str) -> list[Task]:
    """Отсеивает выдуманное и подчищает исполнителей.

    Поручение сверяется с тем фрагментом, откуда взято, а исполнитель — со
    всей стенограммой: человека представляют один раз, а поручения он
    получает потом.
    """
    kept: list[Task] = []
    for task in tasks:
        if not is_russian(task.what):
            logger.info("Отброшено не по-русски: {}", task.what[:80])
            continue
        if not grounded(task, source):
            logger.warning("Отброшено как выдуманное, в стенограмме нет: {}", task.what[:80])
            continue
        kept.append(Task(
            what=task.what,
            who=clean_assignee(task.who, corpus),
            due=clean_due(task.due),
            chunk=task.chunk,
        ))
    return kept


GROUPS_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}
    },
    "required": ["groups"],
    "additionalProperties": False,
}

MERGE_SYSTEM = """Ты — секретарь совещания. Тебе дан список поручений, выписанных по
частям из одной стенограммы. Из-за деления на части одно и то же поручение могло
попасть в список дважды, разными словами.

Твоя задача — только сгруппировать повторы. Верни номера поручений, которые
означают одно и то же:

{"groups": [[1, 4], [2], [3, 5, 6]]}

Правила:
1. Каждый номер встречается ровно один раз.
2. Поручение без пары — группа из одного номера.
3. Не пиши текст поручений, только номера. Ничего не добавляй и не выбрасывай.
4. Разные поручения одному человеку — это разные поручения, не объединяй их."""


def merge_similar(
    tasks: list[Task],
    client: LLMClient,
    *,
    json_mode: bool = True,
) -> list[Task]:
    """Сводит поручения, сказанные разными словами.

    Правила снимают только буквальные повторы, а фрагменты пересекаются, и
    одно поручение приезжает дважды: «подтвердить статус дорожной карты» и
    «проинформировать о текущем статусе дорожной карты». Понять, что это
    одно и то же, может только модель.

    Но решает она не всё: модель возвращает **номера**, а сливает записи код.
    Так она физически не может ни переписать поручение по-своему, ни
    выдумать новое, ни потерять то, о котором забыла, — не упомянутый номер
    остаётся сам по себе.
    """
    if len(tasks) < 2:
        return tasks

    listing = "\n".join(f"{i}. {t.as_line()}" for i, t in enumerate(tasks, 1))
    try:
        reply = client.complete(
            MERGE_SYSTEM, listing,
            json_mode=json_mode,
            schema=GROUPS_SCHEMA if json_mode else None,
        )
    except LLMError as exc:
        logger.warning("Свести повторы не удалось, оставляю как есть: {}", exc)
        return tasks

    groups = _read_groups(reply.text, len(tasks))
    if not _grouping_is_sane(groups, len(tasks)):
        logger.warning(
            "Сведение повторов отвергнуто: {} поручений свелись бы к {}. "
            "Столько повторов в одном совещании не бывает — похоже, модель "
            "сгруппировала неродственное. Оставляю список как есть.",
            len(tasks), len(groups),
        )
        return tasks

    merged = [_merge_group(tasks, indexes) for indexes in groups]
    if len(merged) < len(tasks):
        logger.info("Повторы сведены: {} -> {}", len(tasks), len(merged))
    return merged


#: Насколько список вправе сжаться. Повторы берутся с границ фрагментов, и
#: их столько же, сколько границ: на семи фрагментах это единицы, а не
#: две трети списка.
MERGE_FLOOR = 0.6

#: Больше трёх записей об одном поручении не бывает даже при нахлёсте.
MERGE_MAX_GROUP = 3


def _grouping_is_sane(groups: list[list[int]], count: int) -> bool:
    """Похоже ли сведение на правду.

    Мелкая модель, не поняв задачи, валит всё в две-три кучи — и список из
    восемнадцати поручений превращается в пять. Это не повторы, это потеря
    тринадцати поручений, и заметить её в готовом протоколе нельзя: он
    выглядит просто коротким.
    """
    if len(groups) < count * MERGE_FLOOR:
        return False
    return all(len(group) <= MERGE_MAX_GROUP for group in groups)


def _read_groups(answer: str, count: int) -> list[list[int]]:
    """Разбирает группы, добирая всё, о чём модель умолчала.

    Номер, не попавший ни в одну группу, становится своей группой: молчание
    модели не должно стоить поручения.
    """
    try:
        body = json.loads(answer[answer.find("{") : answer.rfind("}") + 1])
        raw = body.get("groups") or []
    except (ValueError, AttributeError):
        raw = []

    groups: list[list[int]] = []
    seen: set[int] = set()
    for group in raw:
        if not isinstance(group, list):
            continue
        clean = [int(n) - 1 for n in group if isinstance(n, int) and 1 <= n <= count]
        clean = [n for n in clean if n not in seen]
        if clean:
            groups.append(clean)
            seen.update(clean)

    groups.extend([[n] for n in range(count) if n not in seen])
    return sorted(groups, key=lambda g: g[0])


def _merge_group(tasks: list[Task], indexes: list[int]) -> Task:
    """Из нескольких записей об одном поручении — одна, самая полная."""
    group = [tasks[i] for i in indexes]
    # Берём самую подробную формулировку: короткая обычно теряет условие.
    best = max(group, key=lambda t: len(t.what))
    return Task(
        what=best.what,
        who=next((t.who for t in group if t.who), ""),
        due=next((t.due for t in group if t.due), ""),
        chunk=min(t.chunk for t in group),
        at=next((t.at for t in group if t.at is not None), None),
        quote=next((t.quote for t in group if t.quote), ""),
        said_by=next((t.said_by for t in group if t.said_by), ""),
    )


#: Насколько два поручения должны совпасть словами, чтобы считаться одним.
#: Порог высокий намеренно. Поручения из одного совещания похожи между собой
#: по построению: «приступить к строительству объекта КПО Нижневартовска» и
#: «приступить к строительству объекта утилизации в Тюменской области»
#: совпадают на три четверти, а это разные стройки в разных регионах.
#: Различают их одно-два слова, и порог должен быть выше, чем цена этих слов.
SAME_TASK = 0.85

#: Короче этого поручения не сравниваются: на трёх словах любое совпадение
#: случайно.
MIN_WORDS_TO_COMPARE = 4


def _looks_same(one: Task, other: Task) -> bool:
    """Об одном ли эти два поручения.

    Сравниваются значимые слова. Проверка тупая и потому предсказуемая: она
    ловит пересказ теми же словами, а решать про разные слова — работа
    модели, и она с ней, как выяснилось, не справляется.
    """
    left, right = _significant(one.what), _significant(other.what)
    if min(len(left), len(right)) < MIN_WORDS_TO_COMPARE:
        # На трёх словах совпадение ничего не значит: «работать с регионами»
        # и «работать с региональными операторами» совпадут полностью, а это
        # разные поручения.
        return False
    common = left & right
    return len(common) / min(len(left), len(right)) >= SAME_TASK


def dedupe(tasks: Iterable[Task]) -> list[Task]:
    """Убирает повторы.

    Повторы неизбежны: куски стенограммы идут с нахлёстом, поэтому реплика на
    границе попадает в модель дважды. Без этого шага каждое второе поручение
    в протоколе двоилось бы.

    Из двух одинаковых остаётся то, где больше сведений: если в одном
    фрагменте модель увидела задачу без срока, а в другом — со сроком, срок
    сохраняется.
    """
    kept: list[Task] = []
    for task in tasks:
        twin = next(
            (i for i, other in enumerate(kept)
             if other.key == task.key or _looks_same(other, task)),
            None,
        )
        if twin is None:
            kept.append(task)
            continue
        # Из двух записей об одном поручении берётся подробная, а
        # исполнитель и срок — оттуда, где они есть.
        old_task = kept[twin]
        kept[twin] = Task(
            what=max(old_task.what, task.what, key=len),
            who=old_task.who or task.who,
            due=old_task.due or task.due,
            chunk=min(old_task.chunk, task.chunk) or old_task.chunk,
            at=old_task.at if old_task.at is not None else task.at,
            quote=old_task.quote or task.quote,
            said_by=old_task.said_by or task.said_by,
        )
    return kept


def _parse_json(text: str, chunk: int) -> list[Task] | None:
    """Разбирает ответ в JSON, если это он.

    ``None`` означает «это не JSON» — тогда работает разбор по строкам.
    Пустой список — «JSON разобран, поручений в нём нет»; это разные вещи,
    и путать их нельзя.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        body = json.loads(text[start : end + 1])
    except ValueError:
        body = _salvage(text[start:])
        if body is None:
            return None
    if not isinstance(body, dict):
        return None

    items = body.get("tasks")
    if not isinstance(items, list):
        return None

    tasks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        what = str(item.get("what") or "").strip()
        if what:
            tasks.append(Task(
                what=what,
                who=_clean(str(item.get("who") or "")),
                due=_clean(str(item.get("due") or "")),
                chunk=chunk,
            ))
    return tasks


def _salvage(text: str) -> dict | None:
    """Собирает поручения из оборванного ответа.

    Модель упирается в предел длины и обрывается на полуслове: JSON не
    закрыт, и целиком он не разбирается. Но поручения, успевшие уместиться,
    целы — терять их вместе с последним, недописанным, обидно: это минута
    работы модели, выброшенная из-за одной скобки.
    """
    items = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            item = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("what"):
            items.append(item)
    return {"tasks": items} if items else None


def _build(fields: dict[str, str], chunk: int) -> Task:
    return Task(
        what=fields.get("what", "").strip(),
        who=_clean(fields.get("who", "")),
        due=_clean(fields.get("due", "")),
        chunk=chunk,
    )


def _clean(value: str) -> str:
    """Пустое значение и «не указано» — одно и то же.

    Строку «не указано» нельзя оставлять как есть: в отчёте она выглядит
    заполненным полем, и таблицу поручений уже не отфильтровать по тем, где
    исполнителя действительно нет.
    """
    value = (value or "").strip().strip(".").strip()
    return "" if value.lower() in _NOT_SET else value


def _field_of(label: str) -> str | None:
    lowered = label.strip().lower()
    for field, aliases in _FIELD_ALIASES.items():
        if lowered in aliases:
            return field
    return None


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
