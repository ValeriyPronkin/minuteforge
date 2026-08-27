"""Реплики совещания: от сырых сегментов распознавания к читаемым блокам.

Распознавание с диаризацией отдаёт много коротких кусков: одна фраза может
разорваться на три сегмента, а метка спикера — приехать с разным написанием
(``SPEAKER_2``, ``SPEAKER 2``, ``SPE_KER_2``). Для протокола это шум: там
нужна реплика человека целиком, а не то, как модель нарезала аудио.

Модуль ничего не знает про WhisperX и LLM — на вход список сегментов, на
выходе блоки. Поэтому его можно проверять без видеокарты и без моделей.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

#: Метка неизвестного спикера. Диаризация не всегда приписывает сегмент
#: кому-то: реплика может наложиться на чужую или быть слишком короткой.
#: Терять такой текст нельзя — в нём бывает суть поручения.
UNKNOWN = "UNKNOWN"

_SPEAKER_PATTERN = re.compile(r"^SPEAKER[\s_-]*(\d+)$", re.IGNORECASE)


@dataclass
class Block:
    """Реплика одного участника.

    :param speaker: метка спикера — ``SPEAKER_00`` от диаризации либо имя
        человека, если метки уже сопоставлены с участниками.
    :param text: что сказано.
    :param start: секунда начала реплики в записи; нужна, чтобы в протоколе
        можно было вернуться к месту в видео.
    :param end: секунда окончания.
    """

    speaker: str
    text: str
    start: float | None = None
    end: float | None = None

    @property
    def duration(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return max(0.0, self.end - self.start)


def normalize_speaker(label: object) -> str:
    """Приводит метку спикера к виду ``SPEAKER_02``.

    Диаризация и последующая обработка приносят одну и ту же метку в разном
    написании, и без нормализации один человек превращается в нескольких:
    реплики не склеятся, а в протоколе появятся участники-призраки.

    Номер дополняется до двух цифр, иначе сортировка ставит ``SPEAKER_10``
    между ``SPEAKER_1`` и ``SPEAKER_2``.
    """
    if label is None:
        return UNKNOWN
    text = str(label).strip()
    if not text:
        return UNKNOWN

    match = _SPEAKER_PATTERN.match(text.replace(" ", "_"))
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    return text


def blocks_from_segments(segments: Iterable[dict]) -> list[Block]:
    """Превращает сегменты распознавания в блоки.

    Ожидаются словари с ключами ``speaker``, ``text`` и, необязательно,
    ``start`` и ``end`` — то, что отдаёт WhisperX после назначения спикеров.
    Сегмент без текста пропускается, сегмент без спикера достаётся
    :data:`UNKNOWN`, а не выбрасывается.
    """
    blocks: list[Block] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        blocks.append(
            Block(
                speaker=normalize_speaker(segment.get("speaker")),
                text=text,
                start=_as_seconds(segment.get("start")),
                end=_as_seconds(segment.get("end")),
            )
        )
    return blocks


def consolidate(blocks: Sequence[Block]) -> list[Block]:
    """Склеивает подряд идущие реплики одного спикера.

    Границы блоков после склейки — это смены говорящего, а не паузы в речи.
    Именно так протокол и читается человеком, и подаётся в модель: длинная
    связная реплика понятнее, чем два десятка обрывков.

    Время сохраняется: начало берётся у первой реплики, конец у последней.
    """
    merged: list[Block] = []
    for block in blocks:
        if merged and merged[-1].speaker == block.speaker:
            previous = merged[-1]
            previous.text = _join(previous.text, block.text)
            previous.end = block.end if block.end is not None else previous.end
            continue
        merged.append(Block(block.speaker, block.text, block.start, block.end))
    return merged


def speakers(blocks: Iterable[Block]) -> list[str]:
    """Метки спикеров в порядке первого появления.

    Порядок именно такой, а не алфавитный: первым в протоколе обычно говорит
    председатель, и человеку проще сопоставлять метки с участниками, идя по
    ходу совещания.
    """
    seen: list[str] = []
    for block in blocks:
        if block.speaker not in seen:
            seen.append(block.speaker)
    return seen


def rename_speakers(blocks: Sequence[Block], names: dict[str, str]) -> list[Block]:
    """Заменяет метки на имена участников.

    Метка, которой нет в ``names``, остаётся как есть: лучше ``SPEAKER_03``
    в протоколе, чем чужая фамилия под чужими словами.
    """
    renamed = [
        Block(names.get(b.speaker, b.speaker), b.text, b.start, b.end) for b in blocks
    ]
    # После переименования две метки могут оказаться одним человеком — если
    # диаризация разделила его на двоих, а сопоставление свело обратно.
    return consolidate(renamed)


#: Слова переклички. На совещании с десятками подключений первый час
#: уходит на «как слышно, видно» — это до половины записи, и в модель такие
#: фрагменты уходят целиком, вытесняя настоящий разговор.
_SOUNDCHECK = (
    "слышно", "слышн", "видно", "видим", "слышим", "слышу", "проверка связи",
    "микрофон", "картинк", "переименов", "видел", "виден", "не слышу",
)

#: Реплика длиннее этого уже не перекличка, а разговор: «Добрый день,
#: коллеги» умещается в несколько слов, доклад — нет.
_SOUNDCHECK_WORDS = 25

#: Совсем короткая реплика — «Здравствуйте!», «Спасибо» — сама по себе не
#: перекличка, но и не разговор: цепочку переклички она не прерывает.
_CHATTER_WORDS = 3


def is_soundcheck(text: str) -> bool:
    """Похожа ли реплика на проверку связи, а не на разговор.

    Признака два, и нужны оба: в реплике есть слово из переклички и сама она
    короткая. «Добрый день, Рязанскую область, нас видно, слышно» — это
    перекличка; «Коллеги, скажите, как слышно… а теперь по существу
    повестки» — уже нет, потому что дальше пошло содержание.
    """
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    words = re.findall(r"\w+", lowered)
    if len(words) > _SOUNDCHECK_WORDS:
        return False
    return any(mark in lowered for mark in _SOUNDCHECK)


def drop_soundcheck(blocks: Sequence[Block]) -> tuple[list[Block], int]:
    """Убирает перекличку перед началом совещания.

    Возвращает оставшиеся реплики и число убранных.

    Правило намеренно тупое: цепочка переклички идёт, пока реплики короткие,
    и обрывается на первой содержательной. Посреди часовой переклички то и
    дело мелькают «Здравствуйте!» и «Спасибо» — в них нет ни «слышно», ни
    «видно», но и разговором они не являются; останавливаться на них значит
    срезать шесть реплик вместо полутора сотен.

    Точности от этого правила ждать не стоит, и добиваться её усложнением
    незачем: когда нужно ровно, человек указывает время начала — он знает
    его точно, а угадывание всегда останется угадыванием.
    """
    start = 0
    for index, block in enumerate(blocks):
        words = len(re.findall(r"\w+", block.text))
        if is_soundcheck(block.text) or words <= _CHATTER_WORDS:
            start = index + 1
            continue
        break  # пошёл разговор
    return list(blocks[start:]), start


@dataclass
class Transcript:
    """Стенограмма целиком."""

    blocks: list[Block] = field(default_factory=list)
    #: Что пришлось уступить при распознавании: например, модель поменьше,
    #: потому что заказанная не поместилась в видеопамять. Пусто — прошло
    #: как заказано.
    notes: list[str] = field(default_factory=list)

    @property
    def duration_min(self) -> float:
        ends = [b.end for b in self.blocks if b.end is not None]
        return round(max(ends) / 60, 1) if ends else 0.0

    @property
    def speakers(self) -> list[str]:
        return speakers(self.blocks)

    def speaking_time(self) -> dict[str, float]:
        """Сколько секунд говорил каждый.

        По этому числу отделяют участников от обрывков: на длинной записи
        диаризация нарезает десятки меток, но говорят по существу пятеро, а
        у остальных набирается по несколько секунд — чужое эхо, кашель,
        реплика с чужого микрофона.
        """
        totals: dict[str, float] = {}
        for block in self.blocks:
            totals[block.speaker] = totals.get(block.speaker, 0.0) + (block.duration or 0.0)
        return totals

    def speakers_by_time(self) -> list[str]:
        """Метки от самого говорливого к самому молчаливому.

        Именно в этом порядке их и стоит показывать человеку: он опознаёт
        тех, кто вёл совещание и докладывал, а не того, чей микрофон
        случайно поймал два слова.
        """
        totals = self.speaking_time()
        return sorted(self.speakers, key=lambda label: totals.get(label, 0.0), reverse=True)

    def as_text(self, *, with_time: bool = False) -> str:
        """Стенограмма как текст: строка на реплику."""
        lines = []
        for block in self.blocks:
            stamp = f"[{_timestamp(block.start)}] " if with_time and block.start else ""
            lines.append(f"{stamp}{block.speaker}: {block.text}")
        return "\n".join(lines)


def _join(left: str, right: str) -> str:
    """Склейка двух кусков одной реплики.

    Точка в конце предыдущего куска — не конец мысли, а место, где модель
    разрезала аудио, поэтому знак препинания не добавляется: ставить его за
    говорящего значит менять сказанное.
    """
    if not left:
        return right
    return f"{left} {right}".strip()


def _as_seconds(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
