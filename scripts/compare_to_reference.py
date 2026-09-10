"""Сверка протокола с эталоном: что не найдено и что лишнее.

    python scripts/compare_to_reference.py эталон_поручения.csv протокол_поручения.csv

Правки разбора нельзя оценивать на глаз. «Стало лучше» на списке в полсотни
пунктов не видно: две трети совпадают всегда, а спор идёт о десятке. Поэтому
сначала мерка, потом правка — иначе следующая догадка проверяется так же, как
предыдущие две, оказавшиеся наполовину неверными.

Считается две величины, и путать их нельзя:

*   **полнота** — сколько эталонных поручений нашлось. Дороже: пропущенное
    поручение в протоколе не видно никак.
*   **точность** — сколько наших пунктов есть в эталоне. Лишнее вычёркивается
    при вычитке по цитате рядом, поэтому дешевле.

Пункты сводятся по месту в записи и похожести текста: одно и то же поручение
эталон и разбор называют разными словами, а вот время совпадает — оно взято из
той же стенограммы. Пара считается одной, когда она рядом по времени И похожа
текстом или цитатой; из всех возможных пар берутся сначала лучшие, чтобы
соседние поручения не растащили друг у друга совпадения.

У подписанного протокола времени нет и быть не может: документ не ссылается
на минуту записи. Тогда сводится по одному тексту, и порог похожести выше —
подпорки «сказано в ту же минуту» здесь нет.

Файлы с записями совещаний в репозиторий не попадают: пути передаются
аргументами, ничего не читается по умолчанию.
"""

from __future__ import annotations

import csv
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

#: Насколько далеко по времени могут стоять два описания одного поручения.
#: Эталон писал человек, и отметка у него от начала реплики, а не от фразы.
NEAR_S = 150

#: Насколько похожими должны быть тексты, чтобы счесть их одним поручением.
#: Ниже — начинают склеиваться разные поручения из одного места разговора.
ALIKE = 0.30

#: То же, когда времени нет вовсе. В подписанном протоколе его нет и быть не
#: может — документ не ссылается на минуту записи. Тогда единственный признак
#: — текст, и требовать от него приходится больше: подпорки в виде «это
#: сказано в ту же минуту» здесь нет.
ALIKE_BLIND = 0.45


def read_tasks(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    tasks = []
    for row in rows:
        clock = (row.get("Время") or "").strip()
        tasks.append({
            "at": _seconds(clock),
            "time": clock,
            "what": (row.get("Поручение") or "").strip(),
            "who": (row.get("Исполнитель") or "").strip(),
            "quote": (row.get("Цитата") or "").strip(),
        })
    return tasks


def _seconds(clock: str) -> int:
    if clock.count(":") != 2:
        return 0
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^а-яёa-z0-9 ]", " ", text.lower()).split())


def _alike(one: str, other: str) -> float:
    return SequenceMatcher(None, _plain(one), _plain(other)).ratio()


def match(reference: list[dict], ours: list[dict]) -> list[tuple[int, int]]:
    """Пары «эталонное поручение — наше». Лучшие сначала, каждое не больше раза."""
    pairs = []
    for i, want in enumerate(reference):
        for j, got in enumerate(ours):
            timed = bool(want["at"] and got["at"])
            gap = abs(want["at"] - got["at"]) if timed else 0
            if timed and gap > NEAR_S:
                continue
            score = max(_alike(want["what"], got["what"]), _alike(want["quote"], got["quote"]))
            # Время — подсказка, а не доказательство: далёкая пара должна быть
            # заметно похожее текстом, чтобы всё же считаться одной.
            score -= gap / (NEAR_S * 10)
            if score > (ALIKE if timed else ALIKE_BLIND):
                pairs.append((score, i, j))

    pairs.sort(reverse=True)
    taken_ref: set[int] = set()
    taken_our: set[int] = set()
    matched = []
    for _, i, j in pairs:
        if i in taken_ref or j in taken_our:
            continue
        taken_ref.add(i)
        taken_our.add(j)
        matched.append((i, j))
    return matched


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2

    reference = read_tasks(argv[1])
    ours = read_tasks(argv[2])
    matched = match(reference, ours)
    found = {i for i, _ in matched}
    used = {j for _, j in matched}

    print(f"эталон {len(reference)} · наш протокол {len(ours)} · сошлось {len(matched)}")
    print(f"полнота {len(matched) / max(len(reference), 1):.0%} — сколько эталонного нашли")
    print(f"точность {len(matched) / max(len(ours), 1):.0%} — сколько нашего есть в эталоне")

    print(f"\nНЕ НАЙДЕНО ({len(reference) - len(found)}) — дороже всего:")
    for i, want in enumerate(reference):
        if i not in found:
            print(f"  {want['time']}  {want['what'][:88]}")
            print(f"             «{want['quote'][:96]}»")

    print(f"\nЛИШНЕЕ ({len(ours) - len(used)}) — вычёркивается при вычитке:")
    for j, got in enumerate(ours):
        if j not in used:
            print(f"  {got['time']}  {got['what'][:88]}")

    named_ref = sum(1 for i, _ in matched if reference[i]["who"])
    named_our = sum(1 for _, j in matched if ours[j]["who"])
    print(f"\nИсполнитель на совпавших: в эталоне {named_ref}, у нас {named_our}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
