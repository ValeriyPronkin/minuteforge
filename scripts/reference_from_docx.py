"""Эталон из подписанного протокола: docx → поручения.csv и отметили.md.

    python scripts/reference_from_docx.py "Протокол 03.09.docx" data/вкс/2026-09-03/

Протокол приходит после каждого заседания, и каждый раз это новый эталон.
Переносить его руками — полдня, а нужен он ровно в двух видах: раздел
«Решили» строками для мерилки и раздел «Отметили» текстом, пункт на строку.

Разбирается по нумерации, которой документ и держится: «2.» — пункт,
«2.1.» — подпункт, «Срок:» — строка под ними. Пункт, кончающийся
двоеточием, — заголовок: он называет адресата, а поручения идут следом
подпунктами.

docx читается как zip с xml внутри: библиотека для этого не нужна, а лишняя
зависимость на офисной машине — это лишний разговор с админом.
"""

from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path

#: Начало разделов. В проектах протоколов регистр гуляет.
DECISIONS = "РЕШИЛИ"
NOTED = "ОТМЕТИЛИ"

#: Пункт: «2.» или «2.1.» в начале строки.
POINT = re.compile(r"^(\d+)\.(\d+)?\.?\s+(.+)$")
DUE = re.compile(r"^Срок:\s*(.+?)\.?$", re.IGNORECASE)

#: Чем открывается адресат. Убирается, чтобы в графе осталось «Правительству
#: такой-то области», а не оборот, который у нас и так в настройках.
FORMULA = re.compile(r"^(Рекомендовать|Поручить|Предложить)\s+", re.IGNORECASE)


def docx_lines(path: str | Path) -> list[str]:
    with zipfile.ZipFile(path) as book:
        xml = book.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return [line.strip() for line in text.splitlines() if line.strip()]


def section(lines: list[str], title: str) -> list[str]:
    """Строки раздела: от его заголовка до следующего заголовка или конца."""
    try:
        start = next(i for i, line in enumerate(lines) if line.upper().startswith(title))
    except StopIteration:
        return []
    out = []
    for line in lines[start + 1:]:
        upper = line.upper()
        if upper.startswith(DECISIONS) or upper.startswith(NOTED):
            break
        if line.startswith("Приложение:") or line.startswith("Исп.:"):
            break
        out.append(line)
    return out


def decisions(lines: list[str]) -> list[dict]:
    """Поручения строками. Заголовок с двоеточием даёт адресата подпунктам."""
    rows: list[dict] = []
    addressee = ""
    for line in lines:
        due = DUE.match(line)
        if due and rows:
            rows[-1]["due"] = due.group(1).strip()
            continue
        point = POINT.match(line)
        if not point:
            # Продолжение предыдущего пункта: в документе абзац переносят.
            if rows:
                rows[-1]["what"] = f"{rows[-1]['what']} {line}".strip()
            continue
        _, sub, text = point.groups()
        if text.endswith(":") and not sub:
            addressee = FORMULA.sub("", text.rstrip(":")).strip()
            continue
        if not sub:
            addressee = ""
        rows.append({"what": text, "addressee": addressee, "due": ""})
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("нужно: протокол.docx папка-эталона")
        return 2

    lines = docx_lines(argv[1])
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)

    noted = [line for line in section(lines, NOTED) if POINT.match(line)]
    (out / "отметили.md").write_text(
        "# Отметили\n\n" + "\n\n".join(noted) + "\n", encoding="utf-8"
    )

    rows = decisions(section(lines, DECISIONS))
    # Дежурный первый пункт — форма, а не поручение: в сверке он вечно
    # числился бы пропущенным.
    rows = [row for row in rows if "принять к сведению" not in row["what"].lower()]

    table = out / "поручения.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";", lineterminator="\n")
        writer.writerow("№;Время;Поручение;Исполнитель;Регион;Срок;Срок датой;Кто сказал;Цитата".split(";"))
        for number, row in enumerate(rows, 1):
            writer.writerow([number, "", row["what"], "", row["addressee"], "", row["due"], "", ""])

    print(f"«Отметили»: {len(noted)} пунктов → {(out / 'отметили.md').name}")
    print(f"«Решили»: {len(rows)} поручений → {table.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
