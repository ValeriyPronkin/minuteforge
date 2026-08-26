"""Извлечение поручений из стенограммы.

Модель просят не пересказать совещание, а найти в нём поручения: кому, что и
к какому сроку. Ответ спрашивается строками, а не JSON: небольшая местная
модель ломает структуру на первом же длинном тексте, а строку вида
``Поручение: … Кому: … Срок: …`` она держит устойчиво. Разбор здесь
соответственно снисходительный — модель почти всегда добавит от себя
вступление, нумерацию или заголовок, и это не повод потерять поручение.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from loguru import logger

from .blocks import UNKNOWN
from .chunking import Chunk
from .llm import LLMClient, LLMError

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
    #: Из какого куска стенограммы взято. По номеру можно вернуться к месту
    #: разговора и проверить, не выдумала ли модель.
    chunk: int = 0

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


def build_prompt(chunk: Chunk) -> tuple[str, str]:
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
    return EXTRACT_SYSTEM, f"{header}Стенограмма:\n{chunk.text}"


def parse_tasks(answer: str, *, chunk: int = 0) -> list[Task]:
    """Разбирает ответ модели в список поручений.

    Разбор терпит вступления, нумерацию, маркеры списка и лишние заголовки:
    формат ответа модели — просьба, а не гарантия. Единственное, чего он не
    прощает, — отсутствия самого текста поручения: блок без «Поручение»
    пропускается, потому что исполнитель без задачи бессмыслен.
    """
    text = (answer or "").strip()
    if not text or NOTHING_FOUND.lower() in text.lower():
        return []

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
    return tasks


def extract_tasks(
    chunks: Sequence[Chunk],
    client: LLMClient,
    *,
    skip_failed: bool = True,
) -> list[Task]:
    """Проходит по кускам стенограммы и собирает поручения.

    :param skip_failed: что делать, если модель не ответила на кусок. По
        умолчанию — пропустить и идти дальше: терять весь час работы из-за
        одного сбоя хуже, чем недосчитаться поручений из одного фрагмента.
        Пропуск пишется в журнал, молча это не проходит.
    """
    collected: list[Task] = []
    for chunk in chunks:
        system, user = build_prompt(chunk)
        try:
            reply = client.complete(system, user)
        except LLMError as exc:
            if not skip_failed:
                raise
            logger.warning("Фрагмент {}/{} пропущен: {}", chunk.index, chunk.total, exc)
            continue

        found = parse_tasks(reply.text, chunk=chunk.index)
        logger.info(
            "Фрагмент {}/{}: поручений {}, ответ за {} с",
            chunk.index, chunk.total, len(found), reply.elapsed_s,
        )
        collected.extend(found)

    return dedupe(collected)


def dedupe(tasks: Iterable[Task]) -> list[Task]:
    """Убирает повторы.

    Повторы неизбежны: куски стенограммы идут с нахлёстом, поэтому реплика на
    границе попадает в модель дважды. Без этого шага каждое второе поручение
    в протоколе двоилось бы.

    Из двух одинаковых остаётся то, где больше сведений: если в одном
    фрагменте модель увидела задачу без срока, а в другом — со сроком, срок
    сохраняется.
    """
    best: dict[str, Task] = {}
    order: list[str] = []
    for task in tasks:
        key = task.key
        if key not in best:
            best[key] = task
            order.append(key)
            continue
        kept = best[key]
        best[key] = Task(
            what=kept.what,
            who=kept.who or task.who,
            due=kept.due or task.due,
            chunk=kept.chunk,
        )
    return [best[k] for k in order]


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
