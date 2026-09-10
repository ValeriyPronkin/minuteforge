"""Список участников из pdf — в csv, который читает инструмент.

    python scripts/people_from_pdf.py "Список участников 03.09.pdf" участники.csv

Список рассылают перед каждым заседанием, и он всегда одинаково устроен:
раздел «От <организации>», внутри — номер, ФАМИЛИЯ прописными, следом имя с
отчеством и должность. Набирать это руками по тридцать человек к каждому
совещанию никто не станет, а без списка в протоколе останутся SPEAKER_07 и
пустая графа исполнителя.

Разбирается по тому, на чём список и держится: фамилия — единственная
строка прописными, и она же начинается с номера; следом имя с отчеством,
потом должность.

Должность кончается там, где в документе кончается абзац. Это важнее, чем
кажется: следом идёт примечание о том, кто не смог участвовать, — «Первый
Заместитель Губернатора такой-то области Фамилия И.О. на момент заседания
находится в командировке». Без пустой строки оно приклеивается к должности
участника, и в протоколе у человека оказывается чужой пост.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

#: Номер и фамилия прописными: «12.  ИВАНОВ».
SURNAME = re.compile(r"^\d+\.\s+([А-ЯЁ][А-ЯЁ\- ]{2,})\s*$")

#: Имя и отчество следом. Отчество — самый надёжный признак, что это оно.
GIVEN = re.compile(r"^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(ович|евич|овна|евна|ична|инична)\s*$")

#: Раздел: «От Минприроды России», «От Иркутской области».
SECTION = re.compile(r"^От\s+(.+?)\s*$")

#: Примечания о тех, кто не участвует. Они стоят внутри чужой карточки, и
#: без отсева фамилия отсутствующего приезжает в должность участника.
NOTES = (
    "на момент заседания", "официальное письмо", "ответственн",
    "находится в", "принимает участие в ранее",
)

#: «Фамилия И.О.» — так в примечании называют того, кто не смог участвовать.
#: Пустая строка перед примечанием стоит не всегда, а инициалы — всегда.
_ABSENTEE = re.compile(r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.?(?![а-яё])")

#: Слова, которыми начинается должность. Примечание — это чужая должность
#: перед чужими инициалами, и отрезать его надо по началу этой должности, а
#: не по инициалам: иначе в графе останется её хвост.
_TITLE = re.compile(
    r"(Первый|Временно|Исполняющий|Заместитель|Министр|Руководитель|Директор|Глава|Губернатор)"
)

#: Мусор страницы: колонтитулы, номера, шапка списка.
JUNK = re.compile(
    r"^(\d+|СПИСОК УЧАСТНИКОВ|к Инциденту.*|\(\d{2}\.\d{2}\.\d{4}.*"
    # Пометка о способе участия стоит внутри карточки и должностью не
    # является: «(очно)», «(ВКС)», «(по видео-конференц-связи)».
    r"|\((очно|ВКС|дистанционно|по видео.*)\))$",
    re.IGNORECASE,
)


def read_lines(path: str | Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    lines = []
    for page in reader.pages:
        for line in (page.extract_text() or "").splitlines():
            line = " ".join(line.split())
            if JUNK.match(line):
                continue
            # Пустые строки не выбрасываются: ими документ отделяет карточку
            # участника от примечания, и это единственный признак границы.
            if line or (lines and lines[-1]):
                lines.append(line)
    return lines


def read_people(lines: list[str]) -> list[dict]:
    people: list[dict] = []
    org = ""
    position: list[str] = []

    def close() -> None:
        if people and not people[-1]["position"] and position:
            people[-1]["position"] = _own_position(" ".join(position))

    for line in lines:
        if not line:
            # Конец абзаца: должность дописана, дальше идёт примечание.
            close()
            position = []
            continue
        section = SECTION.match(line)
        if section and not GIVEN.match(line):
            close()
            position = []
            org = section.group(1)
            continue
        surname = SURNAME.match(line)
        if surname:
            close()
            position = []
            people.append({
                "surname": " ".join(surname.group(1).split()).title(),
                "given": "",
                "position": "",
                "org": org,
            })
            continue
        if people and GIVEN.match(line) and not people[-1]["given"]:
            people[-1]["given"] = line
            continue
        if any(note in line.lower() for note in NOTES):
            continue
        if people and not people[-1]["position"]:
            position.append(line)
    close()
    return [person for person in people if person["given"]]


def _own_position(text: str) -> str:
    """Должность самого участника, без примечания об отсутствующем."""
    # Переносы pdf: «жилищно -коммунального» — это одно слово.
    text = re.sub(r"(?<=[а-яё]) -(?=[а-яё])", "-", text).strip(" .,")
    absent = _ABSENTEE.search(text)
    if not absent:
        return text
    titles = [m.start() for m in _TITLE.finditer(text[: absent.start()])]
    # Первое вхождение — начало своей должности, последнее — начало чужой.
    cut = titles[-1] if len(titles) > 1 else absent.start()
    own = text[:cut].strip(" .,")
    # Обрубок чужой должности на конце: «Министр природных ресурсов
    # Заместитель». Именно последнее слово: «Министр» в начале — своя.
    words = own.split()
    if words and _TITLE.fullmatch(words[-1]):
        words.pop()
    return " ".join(words).strip(" .,")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("нужно: список.pdf участники.csv")
        return 2

    people = read_people(read_lines(argv[1]))
    out = Path(argv[2])
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";", lineterminator="\n")
        writer.writerow(["ФИО", "Должность", "Организация"])
        for person in people:
            writer.writerow([
                f"{person['surname']} {person['given']}",
                person["position"],
                person["org"],
            ])
    print(f"участников: {len(people)} → {out}")
    for person in people[:5]:
        print(f"  {person['surname']} {person['given']} — {person['position'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
