"""Список участников совещания.

Отдельная сущность, и путать её с числом говорящих нельзя. В списке — все
приглашённые, включая тех, кто промолчал всё совещание; диаризации же нужно
число голосов в записи. На совещании из тридцати человек говорят обычно
пятеро, и если сказать модели «тридцать», она нарежет тридцать спикеров,
расщепив каждого реального на нескольких.

Список нужен для другого: чтобы имена в протоколе были написаны правильно,
с должностями, и чтобы их не приходилось вбивать руками при каждом разборе.

Читается он стандартной библиотекой, без pandas: списки участников — это
десятки строк, а не миллионы, и тащить ради них тяжёлую зависимость незачем.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Sequence

#: Заголовки, по которым узнаётся колонка. Реестры приходят из
#: делопроизводства, и называют их там по-разному.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name", "фио", "ф.и.о.", "ф. и. о.", "участник", "фамилия", "сотрудник"),
    "position": ("position", "должность", "роль", "title"),
    "org": ("org", "организация", "учреждение", "компания", "подразделение"),
}

#: Чем разделяют колонки. Точка с запятой первой: так сохраняет Excel в
#: русской локали, а списки участников делают именно в нём.
DELIMITERS = (";", "\t", ",")

#: Разделители между именем и должностью в текстовом списке. Встречаются оба
#: тире и запятая, нередко в одном и том же файле.
_NAME_POSITION = (" — ", " – ", " - ", ", ")


@dataclass(frozen=True)
class Person:
    """Участник совещания."""

    name: str
    position: str = ""
    org: str = ""

    @property
    def full(self) -> str:
        """Имя с должностью — так участник выглядит в шапке протокола."""
        return ", ".join(part for part in (self.name, self.position, self.org) if part)

    def __str__(self) -> str:
        return self.name


def read_people(source: str | Path | IO[bytes]) -> list[Person]:
    """Читает список участников из csv или простого текста.

    Таблица распознаётся по заголовку: если в первой строке есть что-то
    похожее на «ФИО» или «Должность», дальше читаются колонки. Иначе файл
    считается простым списком, где строка — участник, а должность отделена
    тире или запятой.

    Строки без единой буквы пропускаются: в реестрах из делопроизводства
    хватает подчёркиваний, номеров страниц и разделительных линий.
    """
    lines = _read_lines(source)
    if not lines:
        return []

    delimiter = _delimiter_of(lines)
    if delimiter and _looks_like_header(lines[0], delimiter):
        return _from_table(lines, delimiter)
    return _from_lines(lines)


def names(people: Iterable[Person]) -> list[str]:
    return [person.name for person in people]


def find(people: Iterable[Person], name: str) -> Person | None:
    """Ищет участника по имени.

    Нужен, чтобы подставить должность к тому, кого назвали в протоколе.
    Сравнение без учёта регистра и лишних пробелов — в реестрах то и другое
    в изобилии.
    """
    wanted = (name or "").strip().casefold()
    if not wanted:
        return None
    for person in people:
        if person.name.strip().casefold() == wanted:
            return person
    return None


def _read_lines(source: str | Path | IO[bytes]) -> list[str]:
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        data = source.read()
        text = data.decode("utf-8-sig") if isinstance(data, bytes) else str(data)
    return [line for line in text.splitlines() if line.strip()]


def _delimiter_of(lines: list[str]) -> str:
    first = lines[0]
    for candidate in DELIMITERS:
        if candidate in first:
            return candidate
    return ""


def _looks_like_header(line: str, delimiter: str) -> bool:
    cells = {cell.strip().strip('"').lower() for cell in line.split(delimiter)}
    known = {alias for aliases in COLUMN_ALIASES.values() for alias in aliases}
    return bool(cells & known)


def _from_table(lines: list[str], delimiter: str) -> list[Person]:
    rows = list(csv.reader(lines, delimiter=delimiter))
    header = [cell.strip().lower() for cell in rows[0]]
    index = {}
    for field, aliases in COLUMN_ALIASES.items():
        for position, cell in enumerate(header):
            if cell in aliases:
                index[field] = position
                break

    people: list[Person] = []
    for row in rows[1:]:
        person = Person(
            name=_cell(row, index.get("name", 0)),
            position=_cell(row, index.get("position")),
            org=_cell(row, index.get("org")),
        )
        if person.name:
            people.append(person)
    return people


def _from_lines(lines: Iterable[str]) -> list[Person]:
    people: list[Person] = []
    for raw in lines:
        line = raw.strip().strip("|").strip()
        if not any(character.isalpha() for character in line):
            continue
        for separator in _NAME_POSITION:
            if separator in line:
                name, _, position = line.partition(separator)
                people.append(Person(name.strip(), position.strip()))
                break
        else:
            people.append(Person(line))
    return people


def _cell(row: list[str], position: int | None) -> str:
    if position is None or position >= len(row):
        return ""
    return row[position].strip()


# ------------------------------------------------------------ подсказки

#: Отчества по-русски кончаются предсказуемо, и это самый надёжный признак
#: того, что перед нами именно ФИО, а не просто два слова с большой буквы.
_PATRONYMIC = r"[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна|инична|ична|ьевич|ьевна)"

#: Три слова с большой буквы подряд. Правило грубее, но нужное: Whisper
#: коверкает отчества — в живой расшифровке «Евгеньевна» приехала как
#: «Евгенина», и по окончанию её уже не опознать. По-русски три слова с
#: большой буквы подряд почти всегда и есть ФИО: обычные словосочетания
#: вроде «Московской области» прерываются строчной буквой.
_THREE_CAPITALS = r"[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}"

#: Два порядка написания, оба обычны на совещаниях: «Семенов Алексей
#: Валерьевич» и «Ольга Евгеньевна Белянина». Отчество — самый надёжный
#: признак, поэтому оно проверяется первым.
_FULL_NAME = re.compile(
    rf"\b(?:[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+{_PATRONYMIC}"
    rf"|[А-ЯЁ][а-яё]+\s+{_PATRONYMIC}\s+[А-ЯЁ][а-яё]+"
    rf"|{_THREE_CAPITALS})\b"
)

#: Должность обычно идёт сразу за именем, через запятую, и начинается с
#: одного из этих слов.
_POSITION_WORDS = (
    "министр", "замминистр", "заместитель", "начальник", "руководитель",
    "директор", "председатель", "глава", "советник", "специалист",
    "губернатор", "депутат", "президент", "генеральный", "первый",
)
_POSITION = re.compile(
    r"^\s*,?\s*((?:" + "|".join(_POSITION_WORDS) + r")[^.;!?]{0,80})",
    re.IGNORECASE,
)


#: Распространённые русские имена. Нужны, чтобы отличить ФИО от обычного
#: словосочетания с большой буквы: «Ставка Банка России» — три слова с
#: большой буквы подряд, но человека в них нет, а в «Ольга Евгенина
#: Белянина» есть имя, и этого достаточно.
#:
#: Список заведомо неполный — русских имён тысячи. Он покрывает то, что
#: звучит на совещаниях, и промах здесь стоит недорого: подсказка просто не
#: появится, и человек впишет имя руками.
GIVEN_NAMES = frozenset("""
александр алексей анатолий андрей антон аркадий арсений артем артём борис вадим
валентин валерий василий виктор виталий владимир владислав вячеслав геннадий
георгий григорий даниил денис дмитрий евгений егор иван игорь илья кирилл
константин лев леонид максим марк михаил никита николай олег павел пётр петр
роман руслан сергей станислав степан тимофей тимур федор фёдор эдуард юрий яков
ярослав
алевтина алина алла анастасия ангелина анна антонина валентина валерия варвара
вера вероника виктория галина дарья диана евгения екатерина елена елизавета
жанна зинаида инна ирина карина клавдия ксения лариса лидия любовь людмила
маргарита марина мария надежда наталья наталия нина оксана ольга полина раиса
регина светлана софия софья тамара татьяна ульяна юлия яна
""".split())


def is_given_name(word: str) -> bool:
    return (word or "").strip().casefold() in GIVEN_NAMES


def mentioned_people(text: str) -> list[Person]:
    """Находит в тексте ФИО и, если названа рядом, должность.

    Совещания устроены так, что список участников в них звучит вслух:
    «У нас на связи Семенов Алексей Валерьевич, первый замминистр». Это
    единственный источник имён, когда реестра участников нет, и он же
    самый точный: люди названы ровно так, как их нужно писать в протоколе.

    Порядок сохраняется, повторы убираются: одного человека за совещание
    называют несколько раз.
    """
    found: dict[str, Person] = {}
    for match in _FULL_NAME.finditer(text or ""):
        name = " ".join(match.group(0).split())
        if not _looks_like_person(name):
            continue
        tail = text[match.end():match.end() + 100]
        position = ""
        matched = _POSITION.match(tail)
        if matched:
            position = matched.group(1).strip().rstrip(",")
        key = name.casefold()
        # Из нескольких упоминаний оставляем то, где названа должность.
        if key not in found or (position and not found[key].position):
            found[key] = Person(name=name, position=position)
    return list(found.values())


def suggest_speakers(blocks: Sequence["object"]) -> dict[str, Person]:
    """Угадывает, кому какая метка принадлежит.

    Догадка строится на устройстве самих совещаний: ведущий называет
    человека и передаёт ему слово, после чего тот и говорит. Значит имя,
    прозвучавшее в конце чужой реплики, скорее всего принадлежит тому, кто
    заговорил следующим.

    Это именно подсказка, а не решение: она показывается человеку, и он
    соглашается или правит. Ошибиться тут дёшево, а вот молча подписать
    чужую фамилию под чужими словами — нет.
    """
    guesses: dict[str, Person] = {}
    for previous, current in zip(blocks, blocks[1:]):
        speaker = getattr(current, "speaker", "")
        if speaker in guesses:
            continue
        # Имя ищем в хвосте предыдущей реплики: там передают слово.
        tail = getattr(previous, "text", "")[-300:]
        people = mentioned_people(tail)
        if len(people) == 1 and getattr(previous, "speaker", "") != speaker:
            guesses[speaker] = people[0]
    return guesses


def merge_suggestions(
    roster: Sequence[Person], mentioned: Sequence[Person]
) -> list[Person]:
    """Сводит список участников и имена, прозвучавшие в записи.

    У обоих источников свои сильные стороны. В списке имена написаны верно
    и с должностями, но он есть не всегда. Из записи имена достаются даром,
    но приезжают с ошибками распознавания: «Евгеньевна» превращается в
    «Евгенину», а должность — в родительный падеж.

    Поэтому список главнее: если человек есть в обоих источниках, берётся
    написание из списка. Совпадение ищется по фамилии — она в ФИО самая
    устойчивая часть и меньше страдает от распознавания, чем отчество.
    """
    merged = list(roster)
    known = {_surname(person.name) for person in roster}
    for person in mentioned:
        if _surname(person.name) not in known:
            merged.append(person)
            known.add(_surname(person.name))
    return merged


def _looks_like_person(name: str) -> bool:
    """Похоже ли это на человека, а не на словосочетание.

    Признак — имя или отчество среди трёх слов. Без него «Ставка Банка
    России» проходит правило про три заглавные буквы и попадает в участники
    совещания.
    """
    parts = name.split()
    return any(is_given_name(part) for part in parts) or any(
        re.fullmatch(_PATRONYMIC, part) for part in parts
    )


def _surname(name: str) -> str:
    """Фамилия из ФИО, как её ни напиши.

    Она стоит либо первой, либо последней: «Семенов Алексей Валерьевич» и
    «Ольга Евгеньевна Белянина». Различить помогает имя: если первым идёт
    оно, фамилия в конце, иначе — в начале. По отчеству ориентироваться
    ненадёжно: распознавание его коверкает чаще всего.
    """
    parts = [part for part in (name or "").split() if part]
    if not parts:
        return ""
    if len(parts) < 3:
        return parts[-1].casefold() if is_given_name(parts[0]) else parts[0].casefold()
    return (parts[-1] if is_given_name(parts[0]) else parts[0]).casefold()
