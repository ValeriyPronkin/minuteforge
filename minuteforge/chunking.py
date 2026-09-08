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


#: Сколько фраз вокруг взять в окно. В соседней стоит то, чего в самой
#: фразе нет: «Просьба подтвердить» — и рядом «цех компостирования на 95,7%
#: готов», «до среды», имя того, к кому обращаются.
WINDOW_CONTEXT = 1


def split_into_windows(
    blocks: Sequence[Block],
    *,
    directive: Callable[[str], bool],
    around: int = WINDOW_CONTEXT,
) -> list[Chunk]:
    """Собирает окна вокруг фраз, в которых поручение слышно.

    Другой способ нарезки, чем :func:`split_into_chunks`. Тот делит запись
    подряд, и модель читает всё совещание целиком: на записи штаба это
    30 тысяч токенов семью кусками по 4700. Мелкая модель на таком куске
    теряет середину — поручений в нём полтора десятка, а выписывает она
    первые несколько.

    Здесь в модель уходит только то, где поручение слышно: фраза и её
    соседи. Соседей ищем по всей стенограмме, а не внутри реплики: поручение
    сплошь и рядом разложено на две — «Осталась выгрузка справочников, нужен
    план работ» и в ответ «Сергей, подготовьте до пятницы». Разорви их, и в
    протоколе останется план неизвестно чего.

    Цена честная: правило ``directive`` решает, что модель увидит, а чего
    не увидит вовсе. Промах правила теперь стоит поручения. Но тем же
    правилом поручения отсеиваются и сейчас, только в конце, — так что
    теряется ровно то, что и так не доживало до протокола.

    :param directive: чем проверять фразу. Обычно
        :func:`minuteforge.tasks.worth_showing`.
    :param around: сколько фраз захватывать по сторонам.
    """
    # Стенограмма как сплошная череда фраз: чьи они, помним отдельно. Только
    # так сосед справа находится и тогда, когда он у другого говорящего.
    flat: list[tuple[Block, list[str], int]] = []
    for block in blocks:
        sentences = [s for s in _SENTENCE_SPLIT.split(block.text or "") if s.strip()]
        for position in range(len(sentences)):
            flat.append((block, sentences, position))

    wanted: set[int] = set()
    for index, (_, sentences, position) in enumerate(flat):
        if directive(sentences[position]):
            wanted.update(range(index - around, index + around + 1))
    inside = sorted(i for i in wanted if 0 <= i < len(flat))

    windows = [_window(flat, run) for run in _runs(inside)]
    return [
        Chunk(blocks=part, index=i, total=len(windows))
        for i, part in enumerate(windows, 1)
    ]


def _runs(positions: list[int]) -> list[list[int]]:
    """Идущие подряд номера — в одну группу.

    Два поручения через фразу друг от друга дают перекрывающиеся окна.
    Отправлять их двумя запросами значит дважды заплатить за один и тот же
    текст и получить один и тот же пункт дважды.
    """
    runs: list[list[int]] = []
    for position in positions:
        if runs and position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])
    return runs


def _window(
    flat: list[tuple[Block, list[str], int]],
    run: list[int],
) -> list[Block]:
    """Окно как несколько реплик: подряд идущие фразы одного говорящего вместе.

    Говорящий сохраняется у каждой части: модель должна видеть, что «нужен
    план работ» и «подготовьте до пятницы» сказаны разными людьми — иначе
    она припишет поручение тому, кто на него ответил.
    """
    parts: list[Block] = []
    for index in run:
        block, sentences, position = flat[index]
        if parts and parts[-1].speaker == block.speaker and _same_block(flat, index):
            parts[-1] = _grow(parts[-1], block, sentences, position)
            continue
        parts.append(_piece(block, sentences, position, position))
    return parts


def _same_block(flat: list[tuple[Block, list[str], int]], index: int) -> bool:
    """Та же ли это реплика, что предыдущая фраза."""
    return index > 0 and flat[index - 1][0] is flat[index][0]


def _grow(part: Block, block: Block, sentences: list[str], position: int) -> Block:
    """Дописывает фразу к части окна, сдвигая её конец."""
    grown = _piece(block, sentences, position, position)
    return Block(
        part.speaker,
        f"{part.text} {grown.text}".strip(),
        part.start,
        grown.end if grown.end is not None else part.end,
    )


def _piece(block: Block, sentences: list[str], first: int, last: int) -> Block:
    """Часть реплики со своим временем.

    Время считается по доле текста — точнее нельзя, оно известно только для
    реплики целиком. Зато отметка попадает в нужную минуту десятиминутного
    выступления, а не в его начало.
    """
    text = " ".join(sentences[first:last + 1]).strip()
    length = len(block.text or "")
    before = len(" ".join(sentences[:first]))
    start, end = block.start, block.end
    if start is None or end is None or end <= start or not length:
        return Block(block.speaker, text, start, end)
    span = end - start
    return Block(
        block.speaker,
        text,
        start + span * (min(length, before) / length),
        start + span * (min(length, before + len(text)) / length),
    )


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
