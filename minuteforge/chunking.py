"""Нарезка стенограммы на куски, посильные модели.

Часовое совещание — это десятки тысяч токенов, и целиком в окно модели оно
не влезает. Резать приходится, но резать можно по-разному: если разорвать
реплику посередине, поручение потеряет либо исполнителя, либо срок, и в
протоколе его не будет вовсе.

Поэтому кусок собирается из целых реплик, а не из символов. Реплика, которая
одна не влезает в окно, — отдельный случай: её приходится делить, и делается
это по границам предложений, а не по счётчику символов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .blocks import Block

#: Грубая оценка: сколько символов приходится на токен русского текста.
#: Точный счёт даёт токенизатор модели, но тащить его в ядро незачем —
#: оценка нужна для нарезки, а не для биллинга, и ошибка в меньшую сторону
#: безопасна: кусок просто окажется короче разрешённого.
CHARS_PER_TOKEN = 3.0

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def estimate_tokens(text: str) -> int:
    """Оценка длины в токенах без токенизатора."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class Chunk:
    """Кусок стенограммы, который уходит в модель одним запросом."""

    blocks: list[Block]
    #: Номер куска и общее их число — модель отвечает лучше, когда знает,
    #: что видит часть, а не всё совещание.
    index: int = 0
    total: int = 1

    @property
    def text(self) -> str:
        return "\n".join(f"{b.speaker}: {b.text}" for b in self.blocks)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for block in self.blocks:
            if block.speaker not in seen:
                seen.append(block.speaker)
        return seen


def split_into_chunks(
    blocks: Sequence[Block],
    *,
    max_tokens: int = 6000,
    overlap_blocks: int = 1,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> list[Chunk]:
    """Режет стенограмму на куски не длиннее ``max_tokens``.

    :param max_tokens: сколько токенов отводится под сам текст. Это не всё
        окно модели: место нужно ещё промпту и ответу, поэтому бюджет здесь
        заведомо меньше окна — обычно две трети.
    :param overlap_blocks: сколько последних реплик повторить в начале
        следующего куска. Поручение часто разложено на две реплики — «Иван,
        подготовьте отчёт» и «до среды», — и на границе кусков срок теряется.
        Нахлёст стоит лишних токенов, но дешевле потерянного поручения.
    :param count_tokens: чем считать длину. По умолчанию оценка по символам;
        подставьте токенизатор модели, если нужна точность.

    Реплика длиннее ``max_tokens`` делится по предложениям — иначе её
    пришлось бы либо выбросить, либо отправить заведомо неподъёмный запрос.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens должен быть положительным")

    prepared: list[Block] = []
    for block in blocks:
        prepared.extend(_split_long_block(block, max_tokens, count_tokens))

    chunks: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0

    for block in prepared:
        size = count_tokens(f"{block.speaker}: {block.text}")
        if current and current_tokens + size > max_tokens:
            chunks.append(current)
            # Нахлёст: последние реплики предыдущего куска открывают следующий.
            current = current[-overlap_blocks:] if overlap_blocks else []
            current_tokens = sum(count_tokens(f"{b.speaker}: {b.text}") for b in current)
        current.append(block)
        current_tokens += size

    if current:
        chunks.append(current)

    return [
        Chunk(blocks=part, index=i, total=len(chunks)) for i, part in enumerate(chunks, 1)
    ]


def _split_long_block(
    block: Block,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[Block]:
    """Делит слишком длинную реплику по границам предложений."""
    if count_tokens(f"{block.speaker}: {block.text}") <= max_tokens:
        return [block]

    parts: list[Block] = []
    buffer: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(block.text):
        candidate = " ".join([*buffer, sentence])
        if buffer and count_tokens(f"{block.speaker}: {candidate}") > max_tokens:
            parts.append(Block(block.speaker, " ".join(buffer), block.start, block.end))
            buffer = [sentence]
        else:
            buffer.append(sentence)
    if buffer:
        parts.append(Block(block.speaker, " ".join(buffer), block.start, block.end))

    # Предложение, которое само длиннее окна, остаётся как есть: резать речь
    # человека посреди фразы — значит менять смысл, а это хуже, чем один
    # запрос сверх бюджета, который модель просто обрежет сама.
    return parts
