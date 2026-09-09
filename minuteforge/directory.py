"""Направления: чей вопрос сейчас разбирают.

На совещании, где разбор идёт по кругу, поручение почти никогда не адресовано
вслух. Оно адресовано разбором: ведущий объявляет «следующий — такое-то», за
четверть часа разбирают его, и всё, что за это время поручено, поручено ему.
Человек, читающий протокол, адресата берёт именно отсюда, а не из фразы, — и
это единственный способ заполнить графу, которая иначе пустует у двух третей
пунктов.

По кругу разбирают разное. У межрегионального штаба это субъекты
Федерации, у производства — площадки и цеха, у холдинга — дочерние общества,
у проектного офиса — направления работ. Инструмент ни одного из этих делений
не знает и знать не должен: список приносит тот, кто ведёт протокол, обычным
файлом рядом с настройками. Нет файла — графа остаётся пустой, и об этом
сказано вслух.

Список закрытый, и в этом всё дело. Распознавание коверкает названия:
«Носоморская область», «Синтуки», «Старопольский край». Взять услышанное как
есть — значит подписать поручение тому, кого не существует, а неверный
адресат хуже пустого: пустой заставляет уточнить перед рассылкой, неверный
уходит в рассылку. Поэтому в графу попадает только совпавшее со списком.

Совпадение ищется по основе слова, с начала: «Архангельская»,
«Архангельской», «Архангельску» — одно и то же, а морфологии для этого не
нужно.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from loguru import logger

#: Основа короче этого требует ещё и конца слова: «Чит» (Чита) находится
#: внутри «читаете» и «читать», «Уф» (Уфа) — внутри «уфимский». Длинным
#: основам это не нужно и вредно: «Ставропольского» — восемь букв сверх
#: основы, и все законные.
SHORT_MARK = 5

#: Сколько букв позволено сверх короткой основы. Ровно падежное окончание:
#: «Чита», «Чите», «Читой». Больше — и «читать» снова становится Читой.
SHORT_TAIL = 2

#: Сколько фраз вперёд искать название после объявления. Объявляют и
#: называют обычно порознь: «Следующий.» — и уже в следующей фразе «Иван
#: Иванович, Кисловодск». Двух хватает; дальше начинает попадаться
#: направление, упомянутое мимоходом.
LOOKAHEAD = 2

#: Чем объявляют очередное направление. Названия при этом может не быть:
#: «Следующий регион.» закрывает того, кого разбирали, даже если нового не
#: назвали.
_MOVE = (
    "переходим к", "перейдем к", "перейдём к", "вернемся к", "вернёмся к",
    "далее у нас", "далее регион", "далее город",
)

#: О чём переходят. Без этой проверки «переходим к фотоматериалу» и
#: «следующее заседание переселено на октябрь» считались бы объявлением — и,
#: не найдя названия, стирали бы разбор посреди него же. Список о том, как
#: называют то, что перебирают по кругу: у одних это регионы и города, у
#: других площадки, филиалы и цеха. Названия лежат в справочнике, здесь
#: только слова, которыми их окликают.
#: Слова здесь целые, а не основы, и это не педантизм: «кра» находилось в
#: «кратно усиленным мониторингом» и обрывало разбор посреди него, а
#: «объект» на стройке звучит каждую вторую фразу и обрывал бы постоянно.
_ABOUT_A_UNIT = (
    "регион", "город", "республик", "област", "округ", "субъект",
    "край", "края", "краю", "крае",
    "филиал", "площадк", "подразделен", "направлен", "предприят", "цех",
)

#: Переход не к следующему направлению, а к другому вопросу повестки:
#: «переходим к следующему вопросу», «перейдем к третьему вопросу». Разбор
#: на этом кончается, и прежнее надо стереть — иначе поручения общей части
#: достаются тому, кого разбирали до неё.
_AWAY = ("вопрос", "повестк", "итог")


def _next_pattern() -> "re.Pattern":
    """«Следующий регион», «следующая площадка» — слово и то, что за ним.

    Рядом, а не просто в одной фразе: «отправьте отчёт губернатору края…
    на следующий инцидент» — не объявление, хотя и «следующий» есть, и
    «край». Проверять соседство дешевле, чем разбираться потом, почему
    разбор оборвался посреди региона.
    """
    units = "|".join(re.escape(word) for word in _ABOUT_A_UNIT)
    return re.compile(rf"следующ\w*\s+(?:у\s+нас\s+)?(?:{units})")

#: Слова, которыми переходят, не объявляя: «Далее Иркутская область»,
#: «Начнем с первой площадки», «рассмотрим ситуацию в филиале». Сами по
#: себе они ничего не значат — «и так далее», «работы продолжаются», —
#: поэтому такой переход считается переходом, только если название при нём
#: нашлось. Не нашлось — ничего не происходит, прежнее остаётся.
_QUIET = (
    "далее", "начнем с", "начнём с", "начинаем с", "рассмотрим",
    "продолжаем", "продолжаю", "переходим к", "перейдем к", "перейдём к",
    "вернемся к", "вернёмся к",
)

_NEXT = _next_pattern()

#: Чем открывают обращение к направлению, прежде чем назвать его:
#: «Коллеги Ростовской области», «Администрация города Ессентуки».
_ADDRESS_PREFIX = re.compile(
    r"^(?:(?:коллеги|коллега|администрация|город\w*|представители|правительство)\s+)+",
    re.IGNORECASE,
)

_SENTENCES = re.compile(r"(?<=[.!?…])\s+")

#: Как называется графа, пока не сказано иначе. Слово намеренно никакое:
#: под ним одинаково лежат регион, филиал, площадка и направление работ.
DEFAULT_LABEL = "Направление"


@dataclass(frozen=True)
class Entry:
    """Одно направление: как называть в протоколе и по чему узнавать."""

    name: str
    marks: tuple[str, ...] = ()

    def patterns(self) -> list[re.Pattern]:
        return [_pattern(mark) for mark in (self.marks or (self.name,))]


@dataclass
class Directory:
    """Закрытый список того, по кому идёт разбор."""

    entries: tuple[Entry, ...] = ()
    #: Как называть графу в протоколе и в таблице поручений.
    label: str = DEFAULT_LABEL
    _compiled: list[tuple[str, re.Pattern]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._compiled = [
            (entry.name, pattern)
            for entry in self.entries
            for pattern in entry.patterns()
        ]

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def find(self, text: str) -> str:
        """Какое направление названо в тексте. Пусто — ни одно из списка.

        Если названы два, берётся первое: в фразе перехода — «Далее города
        округа, Ставропольский край» — первым идёт тот, к кому переходят.
        """
        lowered = (text or "").lower()
        best: tuple[int, str] = (len(lowered) + 1, "")
        for name, pattern in self._compiled:
            found = pattern.search(lowered)
            if found and found.start() < best[0]:
                best = (found.start(), name)
        return best[1]

    def starts_with(self, text: str) -> str:
        """Названо ли направление в самом начале фразы.

        Так к нему обращаются: «Коллеги Ростовской области, просьба
        подтвердить», «Первая площадка, начинайте». Отличать это от простого
        упоминания обязательно — «Ставропольский край поддерживает
        законопроект» не поручает краю ничего.
        """
        head = _ADDRESS_PREFIX.sub("", (text or "").strip()).lower()
        for name, pattern in self._compiled:
            found = pattern.search(head)
            if found and found.start() == 0:
                return name
        return ""

    def follow(self, blocks: Sequence[object]) -> list[tuple[float, str]]:
        """Когда какое направление начали разбирать.

        Возвращает переходы по времени записи: секунда и название. Разбор
        считается идущим до следующего перехода — так совещание и устроено,
        один за другим.
        """
        if not self.entries:
            # Пустой справочник не размечает ничего: отметки «не разобрать»
            # без единого названия — только шум в журнале.
            return []
        flat = _phrases(blocks)
        marks: list[tuple[float, str]] = []
        for position, (start, sentence) in enumerate(flat):
            announced = is_announcement(sentence, self)
            if not announced and not is_quiet_move(sentence):
                continue
            if leaves_the_round(sentence):
                # Ушли с разбора — искать название вперёд незачем: первое
                # найденное будет упомянуто в общей части мимоходом.
                name = ""
            else:
                name = self.find(sentence) or self._ahead(flat, position)
            if not announced and not name:
                # Тихий переход без названия — не переход: «и так далее»,
                # «работы продолжаются». Стирать по нему нельзя.
                continue
            # Объявление, в котором названия не разобрать, — всё равно
            # переход, и оно стирает предыдущее. «Далее города округа и
            # Синтуки»: что Ессентуки, знает человек, а список видит только
            # несовпадение. Оставить прежнее значило бы подписать поручения
            # Ессентуков Калмыкии — ровно тот случай, когда неверный
            # адресат хуже пустого.
            if not marks or marks[-1][1] != name:
                marks.append((start, name))
        return marks

    def _ahead(self, flat: Sequence[tuple[float, str]], position: int) -> str:
        """Название в ближайших фразах после объявления."""
        for step in range(1, LOOKAHEAD + 1):
            if position + step >= len(flat):
                break
            name = self.find(flat[position + step][1])
            if name:
                return name
        return ""


#: Пустой справочник: файла нет или он не задан. Разбор идёт как прежде,
#: только графа остаётся незаполненной — выдумывать в неё нечего.
EMPTY = Directory()


def read_directory(
    path: str | Path | None,
    *,
    label: str = DEFAULT_LABEL,
) -> Directory:
    """Читает справочник направлений из csv или txt.

    Ждёт две колонки: как называть и как это звучит в речи. Вторая
    необязательна — без неё узнаётся само название::

        Направление;Как звучит
        Ставропольский край;ставропольск, Ессентуки, Кисловодск
        Первая площадка;первая площадк, Северный цех

    Разделитель — точка с запятой или табуляция: так отдаёт Excel в русской
    локали, а именно им этот файл и правят. Строка заголовка распознаётся по
    словам и пропускается; если её нет, ничего не теряется.

    Файла нет — пустой справочник и предупреждение в журнал. Останавливать
    из-за этого работу нельзя: протокол собирается и без графы.
    """
    if not path:
        return Directory(label=label)
    file = Path(path).expanduser()
    if not file.exists():
        logger.warning(
            "Справочник направлений не найден: {}. Графа «{}» останется пустой.",
            file, label,
        )
        return Directory(label=label)

    try:
        text = file.read_text(encoding="utf-8-sig")
    except OSError as exc:
        logger.warning("Справочник направлений не прочитан: {}", exc)
        return Directory(label=label)

    entries = tuple(_entries(text.splitlines()))
    logger.info("Справочник направлений: {} записей из {}", len(entries), file)
    return Directory(entries=entries, label=label)


def leaves_the_round(text: str) -> bool:
    """Уводит ли эта фраза с разбора по кругу вообще.

    Только вместе с глаголом перехода: «переходим к следующему вопросу» —
    конец разбора, а «следующий вопрос уже прозвучал» — это ведущий считает
    свои вопросы к тому же самому направлению.
    """
    lowered = (text or "").lower()
    return (
        any(word in lowered for word in _MOVE)
        and any(word in lowered for word in _AWAY)
    )


def is_announcement(text: str, directory: "Directory | None" = None) -> bool:
    """Объявляют ли этой фразой следующее по списку.

    Название при этом не обязательно: «Следующий регион.» — объявление, и
    того, кого разбирали до него, оно закрывает, даже если нового не
    назвали. А вот «следующее заседание переселено на октябрь» объявлением
    не является: перебирают не заседания.
    """
    lowered = (text or "").lower()
    if _NEXT.search(lowered):
        return True
    if "следующ" in lowered and directory and directory.find(text):
        return True
    if not any(word in lowered for word in _MOVE):
        return False
    return (
        any(word in lowered for word in _ABOUT_A_UNIT)
        or leaves_the_round(text)
        or bool(directory and directory.find(text))
    )


def is_quiet_move(text: str) -> bool:
    """Похоже ли на переход без объявления.

    Считается только в начале фразы. «И так далее» — не переход, а конец
    перечисления, и на итоговой реплике оно уводило разбор туда, что
    случайно упомянуто двумя предложениями ниже.
    """
    lowered = (text or "").strip().lower().lstrip("-—…,;: ")
    return any(lowered.startswith(word) for word in _QUIET)


def at(marks: Sequence[tuple[float, str]], seconds: float | None) -> str:
    """Чей вопрос разбирали в эту секунду записи.

    До первого перехода — пусто: начало совещания это повестка и перекличка,
    и поручения там если и есть, то не по разбору.
    """
    if seconds is None:
        return ""
    current = ""
    for start, name in marks:
        if start <= seconds:
            current = name
        else:
            break
    return current


def _entries(lines: Iterable[str]) -> Iterable[Entry]:
    # Списком, а не потоком: разделитель ищется по тем же строкам, что
    # потом читаются, и одноразовый итератор здесь молча дал бы пустой файл.
    lines = [line for line in lines if line.strip()]
    rows = csv.reader(lines, delimiter=_delimiter(lines))
    for row in rows:
        cells = [cell.strip() for cell in row if cell.strip()]
        if not cells:
            continue
        name = cells[0]
        if _looks_like_header(name):
            continue
        marks = tuple(
            mark.strip().lower()
            for cell in cells[1:]
            for mark in cell.split(",")
            if mark.strip()
        )
        yield Entry(name=name, marks=marks or (name.lower(),))


#: Слова, которыми называют первую колонку. Заголовок надо пропустить, иначе
#: в справочнике заводится направление с именем «Направление».
_HEADER_WORDS = {
    "направление", "регион", "субъект", "филиал", "площадка", "подразделение",
    "название", "наименование", "кого касается",
}


def _looks_like_header(cell: str) -> bool:
    return cell.strip().lower() in _HEADER_WORDS


def _delimiter(lines: Iterable[str]) -> str:
    """Точка с запятой или табуляция — что нашлось в файле раньше."""
    for line in lines:
        if "\t" in line:
            return "\t"
        if ";" in line:
            return ";"
    return ";"


def _pattern(mark: str) -> re.Pattern:
    body = re.escape(mark.lower())
    if len(mark) < SHORT_MARK:
        return re.compile(
            rf"(?<![а-яёa-z]){body}[а-яё]{{0,{SHORT_TAIL}}}(?![а-яёa-z])"
        )
    return re.compile(rf"(?<![а-яёa-z]){body}")


def _phrases(blocks: Sequence[object]) -> list[tuple[float, str]]:
    """Стенограмма как череда фраз со временем каждой.

    Время считается по доле текста внутри реплики: реплика ведущего бывает в
    две минуты, и «Следующий регион» в её конце, отнесённое к её началу,
    отдавало следующему всё, что поручено в этой же реплике предыдущему.
    """
    flat: list[tuple[float, str]] = []
    for block in blocks:
        start = getattr(block, "start", None)
        if start is None:
            continue
        text = getattr(block, "text", "") or ""
        offset = 0
        for sentence in _SENTENCES.split(text):
            at_char = text.find(sentence, offset)
            if at_char < 0:
                at_char = offset
            offset = at_char + len(sentence)
            if sentence.strip():
                flat.append((_moment(block, at_char), sentence))
    return flat


def _moment(block, offset: int) -> float:
    """Где внутри реплики сказана эта фраза, по доле текста.

    Приблизительно, и точнее не выйдет: время известно только для реплики
    целиком.
    """
    start, end = block.start, block.end
    length = len(getattr(block, "text", "") or "")
    if not length or end is None or end <= start:
        return float(start)
    return float(start) + (end - start) * (offset / length)
