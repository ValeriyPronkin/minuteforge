"""Проверка стенограммы на выдумки распознавания.

На тишине, шуме и плохой связи модель распознавания не молчит, а сочиняет.
Выдумка узнаётся по двум приметам. Первая — повтор: одна и та же фраза
идёт подряд три, пять, десять раз. Вторая — «визитка»: «Субтитры сделал
такой-то», «Продолжение следует» — обрывки титров с видео, на которых
модель училась, и в записи совещания их не произносил никто.

Большая модель сочиняет реже, но опаснее: вместо мусора она выдаёт гладкое
осмысленное предложение, которое легко принять за сказанное и перенести в
протокол. Поэтому проверка нужна тем сильнее, чем лучше модель.

Модуль ничего не исправляет и ничего не выбрасывает — только помечает
места, на которые стоит посмотреть человеку. Стирать за модель нельзя:
среди повторов бывают и настоящие. «Челябинская область, как видно,
слышно?» ведущий переклички действительно повторял пять раз, и это часть
совещания, а не брак.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .blocks import Block, timestamp

#: Обрывки титров, которые модель приносит с обучающих видео. Ни одна из
#: этих фраз на совещании прозвучать не может, поэтому одного попадания
#: достаточно — повтора ждать незачем.
JUNK_MARKS = (
    "субтитры",
    "субтитров",
    "продолжение следует",
    "редактор субтитров",
    "amara.org",
    "dimatorzok",
    "фонд president",
)

#: Сколько раз фраза должна повториться, чтобы стать подозрительной.
ENOUGH = 3

#: В каком промежутке эти повторы должны уместиться. Смысл различения такой:
#: зациклившаяся модель повторяет фразу тут же, подряд, а перекличка на сорок
#: регионов растягивает свои «видно, слышно» на весь час. Второе — совещание,
#: первое — брак.
CLOSE_SECONDS = 120

#: Короче трёх слов фразы не проверяются: «Спасибо» и «Да» на совещании
#: звучат десятки раз, и это нормальная речь, а не выдумка.
MIN_WORDS = 3

_SENTENCE = re.compile(r"[^.!?…]+[.!?…]*")
_NOISE = re.compile(r"[^\w\s]+", re.UNICODE)


@dataclass
class Suspicion:
    """Место, на которое стоит посмотреть человеку."""

    phrase: str
    #: Секунды всех мест, где фраза встретилась.
    at: list[float] = field(default_factory=list)
    #: Почему подозрительно — словами, которые пойдут человеку.
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.at)

    @property
    def moments(self) -> list[float]:
        """Места без повторов внутри одной реплики: их время одно и то же."""
        seen: list[float] = []
        for second in self.at:
            if second not in seen:
                seen.append(second)
        return seen

    def as_line(self) -> str:
        """Одной строкой: что нашли и где смотреть."""
        where = ", ".join(timestamp(second) for second in self.moments[:5])
        if len(self.moments) > 5:
            where += " и далее"
        return f"{self.reason}: «{_short(self.phrase)}» — {where}"


def suspicious(blocks: Iterable[Block], hints: str = "") -> list[Suspicion]:
    """Ищет в стенограмме следы выдумки.

    Возвращает найденное в порядке появления в записи — так его и читают,
    сверяя с исходной записью по времени.

    :param hints: чем подсказывали распознаванию. Нужен, чтобы отличить
        обычную галлюцинацию от эха подсказки: на трудном куске Whisper
        вместо расшифровки продолжает то, что ему подали как предыдущий
        текст. На записи штаба связная подсказка так съела три четверти
        речи в получасовом окне — а в журнале это выглядело обычным
        «повторяется 20 раз подряд», и связать одно с другим человек мог
        только зная, что искать.
    """
    seen: dict[str, list[float]] = {}
    original: dict[str, str] = {}

    for block in blocks:
        for sentence in _sentences(block.text):
            key = _normalize(sentence)
            if not key:
                continue
            seen.setdefault(key, []).append(block.start if block.start is not None else 0.0)
            original.setdefault(key, sentence)

    hinted = {_normalize(sentence) for sentence in _sentences(hints)} - {""}
    found: list[Suspicion] = []
    for key, moments in seen.items():
        reason = _why(key, moments)
        if not reason:
            continue
        if key in hinted:
            reason = (
                "Модель повторяет подсказку вместо речи — уберите "
                "asr_hints_file и распознайте заново"
            )
        found.append(Suspicion(phrase=original[key], at=moments, reason=reason))

    found.sort(key=lambda item: item.at[0])
    return found


def _why(key: str, moments: list[float]) -> str:
    """Чем подозрительна фраза — или пусто, если ничем."""
    if any(mark in key for mark in JUNK_MARKS):
        return "Титры вместо речи"
    if len(key.split()) < MIN_WORDS:
        return ""
    if _burst(moments):
        return f"Повторяется {len(moments)} {_times(len(moments))} подряд"
    return ""


def _times(count: int) -> str:
    """«раз» или «раза» — иначе строка режет глаз тому, кто её читает."""
    tail = count % 10
    if count % 100 // 10 == 1 or tail in (0, 1) or tail >= 5:
        return "раз"
    return "раза"


def _burst(moments: Sequence[float]) -> bool:
    """Уместились ли хотя бы ``ENOUGH`` повторов в один короткий промежуток."""
    if len(moments) < ENOUGH:
        return False
    ordered = sorted(moments)
    return any(
        ordered[start + ENOUGH - 1] - ordered[start] <= CLOSE_SECONDS
        for start in range(len(ordered) - ENOUGH + 1)
    )


def _sentences(text: str) -> list[str]:
    return [found.group(0).strip() for found in _SENTENCE.finditer(text or "") if found.group(0).strip()]


def _normalize(sentence: str) -> str:
    """Фраза без знаков и регистра: сравниваются слова, а не пунктуация."""
    return " ".join(_NOISE.sub(" ", sentence.lower()).split())


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
