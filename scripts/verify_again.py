"""Прогнать проверочный проход по готовым поручениям другой моделью.

    python scripts/verify_again.py прогоны/..._протокол_поручения.csv --model qwen14-protocol

Сравнивать проверяющие модели по двум полным прогонам нельзя. Выписка идёт
той же моделью при `llm_temperature` выше нуля, и на одной записи она дважды
даёт разные списки: на прогонах 27 августа — 155 сырых пунктов против 156,
ответ из 90 окон против 91, и лишь 56 общих меток среди снятого правилами.
Проверки получают на вход разное, и разница в итоге — сумма двух причин,
которую не разделить.

Здесь на вход подаётся один и тот же готовый список, и меняется ровно
проверяющая модель. Распознавание и выписка не повторяются: сорок коротких
запросов вместо часа счёта.

Читается наша же выгрузка `*_поручения.csv`: в ней есть и пункт, и окно, из
которого он выписан, — больше проверке ничего не нужно. Рядом кладётся такая
же таблица с графой «Проверка», и два прогона сравниваются любым сравнением
файлов.

Файлы с записями совещаний в репозиторий не попадают: путь передаётся
аргументом, ничего не читается по умолчанию.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.config import Settings  # noqa: E402
from minuteforge.llm import LLMClient  # noqa: E402
from minuteforge.tasks import Task, verify  # noqa: E402

COLUMNS = ["№", "Время", "Поручение", "Исполнитель", "Регион", "Срок",
           "Срок датой", "Кто сказал", "Цитата", "Проверка"]


def read_tasks(path: Path) -> list[Task]:
    """Поручения из нашей выгрузки.

    Проверке нужны два поля: сам пункт и окно, из которого он взят. Окно
    лежит в графе «Цитата» — там цитата с соседними фразами, ровно то, что
    модель читала на выписке.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    return [
        Task(
            what=(row.get("Поручение") or "").strip(),
            who=(row.get("Исполнитель") or "").strip(),
            due=(row.get("Срок") or "").strip(),
            quote=(row.get("Цитата") or "").strip(),
            context=(row.get("Цитата") or "").strip(),
            said_by=(row.get("Кто сказал") or "").strip(),
            unit=(row.get("Регион") or "").strip(),
            due_date=(row.get("Срок датой") or "").strip(),
            at=_seconds(row.get("Время") or ""),
        )
        for row in rows
    ]


def _seconds(clock: str) -> float | None:
    parts = clock.strip().split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    hours, minutes, seconds = (int(p) for p in parts)
    return float(hours * 3600 + minutes * 60 + seconds)


def _clock(at: float | None) -> str:
    if at is None:
        return ""
    total = int(at)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path, help="выгрузка *_поручения.csv")
    parser.add_argument("--model", default="", help="чем проверять; по умолчанию из настроек")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="куда положить таблицу с графой «Проверка»")
    args = parser.parse_args()

    settings = Settings.load(args.config)
    model = args.model or settings.llm_verify_model or settings.llm_model
    # Проверяющая модель подставляется на место основной: клиент знает про
    # одну, а вторая живёт только в настройках прогона.
    client = LLMClient(replace(settings, llm_model=model))

    tasks = read_tasks(args.tasks)
    if not tasks:
        print("В файле нет поручений.")
        return 1
    print(f"Проверяю {len(tasks)} пунктов моделью {model}…")

    kept = verify(
        tasks, client,
        json_mode=bool(settings.llm_json_mode),
        extra=settings.verify_prompt_extra or "",
    )
    survived = {id(task) for task in kept}
    dropped = [task for task in tasks if id(task) not in survived]

    print(f"\nОставлено {len(kept)} из {len(tasks)}, снято {len(dropped)}.")
    if len(kept) == len(tasks):
        # Проверка отказывается от своего решения, когда сняла бы почти всё:
        # это признак, что модель не поняла вопроса. Молча такое выглядит
        # как «придраться не к чему».
        print("Ни один пункт не снят — либо список чист, либо проверка "
              "отвергнута целиком (смотрите предупреждение выше).")
    for task in dropped:
        print(f"  снято {_clock(task.at)} {task.what[:90]}")

    if args.out:
        with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(COLUMNS)
            for number, task in enumerate(tasks, 1):
                writer.writerow([
                    number, _clock(task.at), task.what, task.who, task.unit,
                    task.due, task.due_date, task.said_by, task.context,
                    "оставлено" if id(task) in survived else "снято",
                ])
        print(f"\nТаблица: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
