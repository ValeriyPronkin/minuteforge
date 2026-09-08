"""Черновик списка участников по готовой стенограмме.

    python scripts/who_is_who.py data/output/запись/стенограмма.json --out участники.csv

Список участников на штабе никто не присылает, а протоколу он нужен: по нему
фамилии исполнителей приводятся к одному написанию, и по нему же в шапке
появляются должности. Зато список звучит вслух — «у нас на связи Семенов
Алексей Валерьевич, первый замминистра», — и это самый точный источник:
люди названы ровно так, как их надо писать в документе.

Скрипт достаёт названных из стенограммы, считает, сколько раз каждого
назвали, и складывает в csv того вида, который читает приложение. Это
черновик: распознавание коверкает фамилии, и вычитать его придётся глазами.
Поэтому рядом с каждым — число упоминаний и метка голоса, если её удалось
угадать: по ним видно, кто на совещании работал, а кто мелькнул однажды.

Модель не нужна, видеокарта тоже.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.blocks import Block  # noqa: E402
from minuteforge.people import (  # noqa: E402
    Person,
    is_given_name,
    mentioned_people,
    suggest_speakers,
)

from who_gives_orders import read_blocks  # noqa: E402

#: Реже этого человек на совещании не участник, а упомянутый вскользь:
#: названный один-два раза — это чаще всего тот, о ком говорили в третьем
#: лице, или чужая фамилия, услышанная распознаванием в чужом слове.
ENOUGH = 3


#: Чем кончается глагол, приставший к имени: «Докладывает Джамбулат
#: Хизирович», «Выступает Иванов Иван». Правило поиска ФИО видит три слова
#: с большой буквы и берёт их все — а первое из них не человек.
_VERB_TAIL = ("ет", "ит", "ют", "ует", "ается", "уется", "тся")


def without_verb(name: str) -> str:
    """Отрезает глагол перед именем."""
    words = name.split()
    while len(words) > 2 and words[0].lower().endswith(_VERB_TAIL):
        words = words[1:]
    return " ".join(words)


def surname(name: str) -> str:
    """Фамилия из ФИО — по ней человека и считают.

    Порядок слов в живой речи свободный: «Павленко Андрей Васильевич» и
    «Давид Эрнестович Аванесян» — одинаково обычные. Отличается фамилия
    тем, чем не является: не имя из святцев и не отчество.
    """
    words = [w for w in re.findall(r"\w+", name) if len(w) > 2]
    for word in words:
        lowered = word.lower()
        if is_given_name(lowered):
            continue
        if lowered.endswith(("ович", "евич", "ична", "овна", "евна", "инична")):
            continue
        return word
    return words[-1] if words else name


def mentions(text: str, name: str) -> int:
    """Сколько раз фамилия звучала за совещание.

    Считается по основе, а не по слову целиком: «Аванесян», «Аванесяна» и
    «Аванесяну» — один человек, и точное совпадение их не признаёт. Это и
    есть мера участия: ведущего называют десятки раз, случайно упомянутого
    один.
    """
    stem = surname(name).lower()
    stem = stem[:-2] if len(stem) > 5 else stem
    return len(re.findall(re.escape(stem), text.lower()))


def gather(blocks: Sequence[Block]) -> list[tuple[Person, int, str]]:
    """Кого назвали, сколько раз и чьим голосом он говорил.

    Считается по всей стенограмме: человека представляют один раз, а
    называют потом ещё десять — по этому счёту и видно, кто на совещании
    работал.

    Написания сводятся по фамилии: распознавание переставляет слова и
    коверкает отчества, и один и тот же человек приезжает в список
    трижды — «Давид Эрнецович Аванесян», «Аванесян Давид Эрнестович»,
    «Аванесяна Давида».
    """
    text = " ".join(block.text for block in blocks)
    named: dict[str, Person] = {}
    for block in blocks:
        for person in mentioned_people(block.text):
            person = Person(without_verb(person.name), person.position, person.org)
            key = surname(person.name).lower()
            if not key:
                continue
            # Из нескольких написаний оставляем то, где названа должность,
            # а при прочих равных — самое длинное: в нём есть отчество.
            known = named.get(key)
            better = (
                known is None
                or (person.position and not known.position)
                or (len(person.name) > len(known.name) and not known.position)
            )
            if better:
                named[key] = person

    voices = {
        surname(person.name).lower(): label
        for label, person in suggest_speakers(blocks).items()
    }
    people = [
        (person, mentions(text, person.name), voices.get(key, ""))
        for key, person in named.items()
    ]
    # Сначала те, кого называли чаще: ведущий и докладчики. Хвост списка —
    # упомянутые вскользь, и вычёркивать придётся именно оттуда.
    return sorted(people, key=lambda item: (-item[1], item[0].name))


def write_csv(people: Sequence[tuple[Person, int, str]], path: Path) -> None:
    """Пишет тот же csv, который приложение читает обратно.

    Колонки — как в примере: ФИО, должность, организация. Число упоминаний
    и метка голоса в файл не идут: приложению они не нужны, а человеку,
    который правит список, мешают.
    """
    # utf-8 с меткой: список открывают в Excel, а он без метки показывает
    # русские фамилии крякозябрами.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(["ФИО", "Должность", "Организация"])
        for person, _, _ in people:
            writer.writerow([person.name, person.position, person.org])


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path, help="стенограмма: json или txt")
    parser.add_argument("--out", type=Path, default=None, help="куда записать csv")
    parser.add_argument(
        "--раз", dest="enough", type=int, default=ENOUGH,
        help=f"сколько раз должны назвать, чтобы попасть в список (по умолчанию {ENOUGH})",
    )
    args = parser.parse_args()

    if not args.transcript.exists():
        parser.error(f"нет такого файла: {args.transcript}")

    blocks = read_blocks(args.transcript).blocks
    everyone = gather(blocks)
    people = [item for item in everyone if item[1] >= args.enough]

    print(f"Названо людей: {len(everyone)}, из них названы не реже {args.enough} раз: {len(people)}")
    print()
    print(f"{'ФИО':<34}{'раз':>4}  {'голос':<12}должность")
    for person, times, voice in people:
        print(f"{person.name:<34}{times:>4}  {voice:<12}{person.position}")

    if len(everyone) > len(people):
        print()
        print("Названы однажды — скорее упомянутые, чем участники:")
        print("  " + ", ".join(p.name for p, times, _ in everyone if times < args.enough))

    if args.out:
        write_csv(people, args.out)
        print()
        print(f"Черновик списка: {args.out.resolve()}")
        print(
            "Вычитайте его: распознавание коверкает фамилии, и в протокол "
            "они уйдут как есть."
        )


if __name__ == "__main__":
    main()
