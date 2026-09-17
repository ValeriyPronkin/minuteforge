"""Сценарий заседания: кто говорит, по кому идёт разбор и кто ведёт.

К заседанию готовят технический сценарий — документ о том, кто когда берёт
слово. Выглядит он так::

    А.Н. ГОЛОВАНОВА. Презентация. Республика Марий Эл.
    ВИДЕО Республика Марий Эл
    Слово Министр строительства, архитектуры и ЖКХ Республики Марий Эл —
          Горбунов Александр Всеволодович.
    Комментарий Д.Х. ХАТУОВ.

И так по всем направлениям повестки. Оказывается, в нём лежит разом всё, что
мы до сих пор собирали руками по трём файлам: список выступающих с
должностями, привязка каждого к своему направлению, порядок разбора и имя
ведущего.

Ценность его в том, что это форма, а не факт. Сценарий говорит, кто будет
докладывать по Марий Эл, — и это верно независимо от того, что он скажет.
Чисел и сроков отсюда брать нельзя: в сценарии их нет, а если бы были, они
означали бы «как планировали», а протокол пишут о том, что решили.

Сценарий может и не сбыться: кто-то не выйдет на связь, порядок переставят.
Поэтому всё отсюда — подсказка, которую человек видит и правит, а не
решение.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from loguru import logger

from .people import Person, mentioned_people

#: Чем в сценарии объявляют очередное направление: «Презентация. Омская
#: область.», «ВИДЕО Республика Адыгея». Само название ищется справочником —
#: здесь только слова, которыми его окликают.
_ABOUT_A_UNIT = ("презентация", "видео")

#: Чем открывают очередной вопрос повестки: «II. Второй вопрос повестки».
#: Разбор на этом начинается заново, и прежнее направление надо стереть —
#: иначе федеральный докладчик, объявленный сразу после заголовка, достаётся
#: последнему региону предыдущего вопроса.
_NEW_TOPIC = ("вопрос повестки", "вопрос заседания")

#: Чем открывают строку про человека. Слово служебное и в имя не входит:
#: без этого «Доклад Карелов Евгений Александрович» даёт участника по имени
#: «Доклад Карелов Евгений».
_ROLES = ("слово", "доклад", "комментарий", "ответ на вопрос", "вопрос", "выступление")

#: Чем отделяют должность от фамилии: «Министр ЖКХ — Горбунов А.В.».
_DASH = re.compile(r"\s+[–—-]\s+")

#: Инициалы с фамилией: «Д.Х. ХАТУОВ», «А.Н. ГОЛОВАНОВА». В сценарии так
#: пишут тех, кто ведёт и кто докладывает по всей повестке.
_INITIALS = re.compile(r"\b([А-ЯЁ])\.\s?([А-ЯЁ])\.\s*([А-ЯЁ][А-ЯЁа-яё]{2,})")


@dataclass
class Scenario:
    """Что удалось прочесть из сценария заседания."""

    #: Направления в том порядке, в каком их собираются разбирать.
    units: list[str] = field(default_factory=list)
    #: Выступающие с должностями; в ``org`` стоит их направление.
    people: list[Person] = field(default_factory=list)
    #: Кто ведёт. Тот, чьи инициалы в сценарии встречаются чаще всех.
    chair: str = ""

    def __bool__(self) -> bool:
        return bool(self.units or self.people)


EMPTY = Scenario()


def read_scenario(path: str | Path | None, directory=None) -> Scenario:
    """Читает сценарий заседания: docx или обычный текст.

    :param directory: справочник направлений. Без него названия не
        опознаются: сценарий пишет их как придётся — «Республика Адыгея»,
        «Чукотский автономный округ», — а брать их как есть значит завести
        направление из опечатки.

    Файла нет или он не читается — пустой сценарий и строка в журнал.
    Останавливать из-за этого работу нельзя: всё, что он даёт, задаётся и
    руками.
    """
    if not path:
        return EMPTY
    file = Path(path).expanduser()
    if not file.exists():
        logger.warning("Сценарий заседания не найден: {}", file)
        return EMPTY
    try:
        paragraphs = _paragraphs(file)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        logger.warning("Сценарий заседания не прочитан: {}", exc)
        return EMPTY

    found = _walk(paragraphs, directory)
    logger.info(
        "Сценарий: направлений {}, выступающих {}, ведёт {}",
        len(found.units), len(found.people), found.chair or "—",
    )
    return found


def _paragraphs(file: Path) -> list[str]:
    """Абзацы документа. Word — это zip с разметкой внутри."""
    if file.suffix.lower() != ".docx":
        return [
            line.strip()
            for line in file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    with zipfile.ZipFile(file) as book:
        body = book.read("word/document.xml").decode("utf-8")
    parts = []
    for block in re.findall(r"<w:p[ >].*?</w:p>", body, re.S):
        # Внутри абзаца текст разорван на куски разметкой: Word ставит
        # границу там, где сменилось начертание, и «Карелов» приезжает
        # тремя кусками. Склеиваем и только потом читаем.
        text = " ".join("".join(
            re.findall(r"<w:t[^>]*>(.*?)</w:t>", block, re.S)
        ).split())
        if text:
            parts.append(text)
    return parts


def _walk(paragraphs: Sequence[str], directory=None) -> Scenario:
    """Идёт по сценарию сверху вниз, запоминая, чей сейчас вопрос."""
    units: list[str] = []
    people: list[Person] = []
    initials: dict[str, int] = {}
    current = ""

    for text in paragraphs:
        for one, two, surname in _INITIALS.findall(text):
            name = f"{one}.{two}. {surname.title()}"
            initials[name] = initials.get(name, 0) + 1

        if any(word in text.lower() for word in _NEW_TOPIC):
            current = ""

        named = _unit_in(text, directory)
        if named and _about_a_unit(text):
            current = named
            if named not in units:
                units.append(named)

        person = _person_in(text)
        if person is not None:
            people.append(Person(person.name, person.position, current))

    chair = _who_leads(initials)
    return Scenario(units=units, people=_without_repeats(people), chair=chair)


#: По скольким буквам фамилии считать её той же самой. Сценарий пишет
#: ведущего то в именительном, то в родительном — «Комментарий Д.Х. ХАТУОВ»
#: и «Завершающее слово Д.Х. Хатуова», — и порознь они делят голоса пополам.
_SAME_SURNAME = 6


def _who_leads(initials: dict[str, int]) -> str:
    """Кто ведёт: чьё имя звучит в сценарии чаще всех.

    Его комментарий стоит после каждого доклада, а докладчик назван один
    раз. Падежи сводятся: считаются они вместе, а в документ идёт самая
    короткая форма — именительный падеж короче родительного, и для фамилии
    это верно и в мужском роде, и в женском.
    """
    if not initials:
        return ""
    counted: dict[str, int] = {}
    forms: dict[str, set[str]] = {}
    for name, times in initials.items():
        key = name[:name.rfind(" ") + 1 + _SAME_SURNAME]
        counted[key] = counted.get(key, 0) + times
        forms.setdefault(key, set()).add(name)
    best = max(counted, key=lambda key: counted[key])
    return min(forms[best], key=len)


def _about_a_unit(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in _ABOUT_A_UNIT)


def _unit_in(text: str, directory=None) -> str:
    if directory is None:
        return ""
    found = directory.find_all(text)
    return found[0] if len(found) == 1 else ""


def _person_in(text: str) -> Person | None:
    """Человек и его должность из строки сценария.

    Строка бывает в двух видах, и оба надо разобрать::

        Слово Министр строительства Республики Марий Эл — Горбунов А.В.
        Доклад Карелов Евгений Александрович – Заместитель директора

    Где должность, а где имя, решает не порядок, а то, в какой половине
    нашёлся человек: пишут и так и так, а фамилия узнаётся одинаково.
    """
    said = _without_role(text)
    if not said:
        return None
    halves = _DASH.split(said, maxsplit=1)
    for position, half in enumerate(halves):
        people = mentioned_people(half) or mentioned_people(_tail(half))
        if len(people) == 1:
            other = halves[1 - position] if len(halves) > 1 else ""
            # Тире ставят не всегда: «Слово Первый заместитель Губернатора
            # Курганской области Воробьев Анатолий Анатольевич». Тогда
            # должность — это всё, что осталось от строки без имени.
            return Person(people[0].name, _trim(other or _besides(half, people[0].name)))
    return None


#: Сколько последних слов пробовать, когда имя в строку не далось. Фамилия
#: с именем и отчеством стоит в конце — «…министр сельского хозяйства
#: Республики Башкортостан Фазрахманов Ильшат Ильдусович», — а правило про
#: людей отказывается разбирать длинную цепочку слов с заглавной: там и
#: «Республики Башкортостан», и сам человек, и отличить их по написанию
#: нельзя.
_TAIL_WORDS = 3


def _tail(text: str) -> str:
    """Последние слова строки — там стоит ФИО, когда впереди длинная должность."""
    return " ".join(text.split()[-_TAIL_WORDS:])


def _besides(text: str, name: str) -> str:
    """Строка без имени: остальное и есть должность."""
    for word in name.split():
        text = re.sub(rf"\b{re.escape(word)}\w*", " ", text)
    return text


def _without_role(text: str) -> str:
    """Снимает служебное начало строки: «Слово», «Доклад», «Комментарий»."""
    lowered = text.lower().lstrip("0123456789. ")
    for role in _ROLES:
        if lowered.startswith(role):
            return text[len(text) - len(lowered) + len(role):].strip(" .:-—–")
    return text


def _trim(position: str) -> str:
    """Должность без хвостов разметки и без скобок с пометками."""
    clean = re.sub(r"\([^)]*\)", "", position or "")
    return " ".join(clean.split()).strip(" .,:;-—–")


def _without_repeats(people: Sequence[Person]) -> list[Person]:
    """Один человек — одна запись. В сценарии докладчик назван и в начале
    вопроса повестки, и перед каждой своей презентацией."""
    kept: list[Person] = []
    for person in people:
        same = next((known for known in kept if known.name == person.name), None)
        if same is None:
            kept.append(person)
        elif not same.position and person.position:
            kept[kept.index(same)] = person
    return kept
